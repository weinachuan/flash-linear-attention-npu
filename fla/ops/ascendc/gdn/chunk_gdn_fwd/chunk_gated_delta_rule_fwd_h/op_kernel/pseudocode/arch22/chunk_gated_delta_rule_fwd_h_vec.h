// 仅伪代码。arch22（A2/A3）Vec 阶段：S-1、Stage1 和 Stage3。
// 该文件保留真实 op_kernel 的文件名和架构目录，具体 Ascend C 调用需在实现时补齐。

#pragma once

#include "../chunk_gated_delta_rule_fwd_h_policy.h"
#include "../chunk_gated_delta_rule_fwd_h_utils.h"

namespace fwd_h_pseudocode {

struct VecStageArgs {
    const ApiInputs* in = nullptr;
    const ApiOutputs* out = nullptr;
    const TilingPlan* tiling = nullptr;
    const RoundPlan* plan = nullptr;
    FixedMemory* memory = nullptr;
    SyncLedger* sync = nullptr;
    Stage1Variant variant = Stage1Variant::WithP;
};

struct VecStage3Args {
    const ApiInputs* in = nullptr;
    const ApiOutputs* out = nullptr;
    const TilingPlan* tiling = nullptr;
    const RoundPlan* plan = nullptr;
    FixedMemory* memory = nullptr;
    SyncLedger* sync = nullptr;
};

struct SMinusOneArgs {
    const ApiInputs* in = nullptr;
    const ApiOutputs* out = nullptr;
    const TilingPlan* tiling = nullptr;
    const SequenceSpan* sequence = nullptr;
    FixedMemory* memory = nullptr;
    SyncLedger* sync = nullptr;
    int activeHvBegin = 0;
    int activeHvCount = 0;
};

struct VecStageResult {
    bool rightOperandReady = false;
    bool alphaReady = false;
    bool nextHReady = false;
    bool finalStateWritten = false;
};

inline void RunSMinusOneArch22(const SMinusOneArgs& args)
{
    if (!args.tiling->useInitialState || args.tiling->stateType != StateType::Fp32) {
        return;
    }
    for (int local = 0; local < args.activeHvCount; ++local) {
        const int hv = args.activeHvBegin + local;
        const int hSlot = local;
        args.memory->ProduceInitialH(hSlot);

        // 唯一的 GM state 布局转换发生在这次 layout-aware 读取。
        // VF 输出是供 S0 使用的规范 BF16 [K,V]，同时写入逻辑 h0。
        const auto state = Mte2ReadStateLayoutAware(
            args.in->initialState, args.sequence->sequence, hv, args.tiling->stateLayout);
        const auto h0 = CastBf16(VfCastFp32ToBf16(state));
        Mte3WriteH0LayoutAware(*args.out, *args.tiling, *args.sequence, hv, h0);
        Mte3WriteL1Resident(L1H(args.memory->l1, hSlot), h0);
        args.memory->MarkHReady(hSlot);
        PublishInitialHReady(*args.sync, hSlot, hv);
    }
    WaitInitialPhaseDrain(*args.sync, args.sequence->sequence);
}

inline VecStageResult RunStage1Arch22(const VecStageArgs& args)
{
    VecStageResult result{};
    const RoundPlan& plan = *args.plan;
    for (int i = 0; i < plan.activeHvCount; ++i) {
        const HeadBinding& head = plan.heads[i];
        const bool hasP = plan.stage0Required;
        if (hasP) {
            args.sync->Wait(EventKind::PReady, head.roundHead, /*S1 消费者*/ 1);
        }

        // 一个 VF 覆盖本 head 的 [M,V]；尾部行使用 mask 并补零。
        const auto p = hasP ? UbP(*args.memory, head) : ZeroP(head);
        const auto v_new_fp32 = VfSubFp32(LoadU(*args.in, plan.chunk, head.hv), p);
        const auto v_new = CastBf16(v_new_fp32);
        write_v_new(*args.out, plan.chunk, head.hv, v_new);

        const bool needsRight = plan.stage2Required && args.variant != Stage1Variant::v_new_only;
        if (needsRight) {
            args.memory->AcquireRightForS1(head.hSlot);
        }
        if (plan.gateMode == GateMode::ScalarG && needsRight) {
            const auto gate = LoadG(*args.in, plan.chunk, head.hv);
            const auto v_new_g = CastBf16(
                VfMul(v_new_fp32, GateDelta(gate, plan.chunk.validTokens, args.tiling->useExp2)));
            Mte3WriteL1Right(L1Right(args.memory->l1, head.hSlot), v_new_g);
            StoreAlpha(UbAlpha(*args.memory, head),
                       LastGate(gate, plan.chunk.validTokens, args.tiling->useExp2));
            args.sync->Set(EventKind::RightReady, head.hSlot,
                           /*S1 MTE3 生产者*/ 1, /*S2 消费者*/ 2);
            result.alphaReady = true;
        } else if (plan.gateMode == GateMode::KeyWiseGk && needsRight) {
            // gk-only 在 S1 不读取 gk；相对当前 chunk 的衰减已经包含在 kg 中。
            Mte3WriteL1Right(L1Right(args.memory->l1, head.hSlot), v_new);
            args.sync->Set(EventKind::RightReady, head.hSlot,
                           /*S1 MTE3 生产者*/ 1, /*S2 消费者*/ 2);
        }

        if (needsRight) {
            result.rightOperandReady = true;
        }
        if (plan.chunk.first) {
            // 第一个 h 输出使用原生 state_v_first 地址。
            // FP32 初态已经由 S-1 写入 h0，因此 S1 不再重复读取。
            WriteH0IfNeeded(*args.in, *args.out, *args.tiling, plan, head, args.variant);
        }
        if (needsRight) {
            args.memory->MarkRightReady(head.hSlot);
        }
    }
    return result;
}

inline VecStageResult RunStage3Arch22(const VecStage3Args& args)
{
    VecStageResult result{};
    const RoundPlan& plan = *args.plan;
    if (!plan.stage3Required) {
        return result;
    }

    for (int i = 0; i < plan.activeHvCount; ++i) {
        const HeadBinding& head = plan.heads[i];
        args.sync->Wait(EventKind::DReady, head.roundHead, /*S3 消费者*/ 3);

        const auto r = LoadRollingStateCanonical(*args.in, *args.out, *args.tiling, plan, head);
        auto rNextFp32 = ToFp32(r);
        if (plan.gateMode == GateMode::ScalarG) {
            const float alpha = plan.chunk.first && !args.tiling->useInitialState
                                    ? 1.0f
                                    : LoadAlpha(UbAlpha(*args.memory, head));
            rNextFp32 = VfMul(rNextFp32, alpha);
        } else {
            const auto gkLast = LoadGkLast(*args.in, plan.chunk, head.hv);
            const auto rowGate = GateExp(gkLast, args.tiling->useExp2);
            rNextFp32 = VfRowMul(rNextFp32, rowGate);
        }
        rNextFp32 = VfAdd(rNextFp32, UbD(*args.memory, head));

        const auto rNext = CastState(rNextFp32, args.tiling->stateType);
        const auto hNext = CastBf16(rNext);
        if (!plan.chunk.last) {
            // 一个 VF 生成规范 [K,V]，GM h 和 L1 resident 共同使用该结果。
            args.memory->ProduceHForS3(head.hSlot);
            Mte3WriteHLayoutAware(*args.out, *args.tiling, plan, head, hNext);
            Mte3WriteL1Resident(L1H(args.memory->l1, head.hSlot), hNext);
            args.memory->MarkHReady(head.hSlot);
            args.sync->Set(EventKind::HReady, head.hSlot,
                           /*S3 生产者*/ 3, /*下一个 S0 消费者*/ 0);
            result.nextHReady = true;
        }
        if (plan.chunk.last && args.tiling->outputFinalState) {
            Mte3WriteFinalStateLayoutAware(*args.out, *args.tiling, plan, head, rNext);
            result.finalStateWritten = true;
        }

        // FP32 rolling state 只有在后续 S3 或公开的 final_state 会消费时才写入 GM fallback。
        // 对已经没有消费者的值不创建 state-ready 事件。
        StoreOrReleaseRollingState(*args.in, *args.out, *args.tiling, plan, head, rNext);
        args.sync->Release(EventKind::DReady, head.roundHead, /*S3 消费者*/ 3);
    }
    return result;
}

} // 命名空间 fwd_h_pseudocode
