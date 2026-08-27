// 仅伪代码。Stage1：每个完整 head 执行一次 VF，生成 V_new 和可选的 V_new_g。

#pragma once

#include "fwd_h_pseudocode_memory.h"
#include "fwd_h_pseudocode_stage0.h"

namespace fwd_h_pseudocode {

enum class Stage1Variant { WithP, NoP, v_new_only };

struct Stage1Args {
    const ApiInputs* in = nullptr;
    const ApiOutputs* out = nullptr;
    const TilingPlan* tiling = nullptr;
    const RoundPlan* plan = nullptr;
    FixedMemory* memory = nullptr;
    SyncLedger* sync = nullptr;
    Stage1Variant variant = Stage1Variant::WithP;
};

struct Stage1Result {
    bool rightOperandReady = false;
    bool alphaReady = false;
};

inline Stage1Result RunStage1(const Stage1Args& args)
{
    Stage1Result result{};
    const RoundPlan& plan = *args.plan;
    for (int i = 0; i < plan.activeHvCount; ++i) {
        const HeadBinding& head = plan.heads[i];
        const bool hasP = plan.stage0Required;
        if (hasP) {
            args.sync->Wait(EventKind::PReady, head.roundHead, /*S1 消费者*/ 1);
        }

        // 一个 RegBase VF 覆盖本 head 的 [M,V]；尾部行使用 mask 并补零。
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
            const auto v_new_g = CastBf16(VfMul(v_new_fp32, GateDelta(gate, plan.chunk.validTokens, args.tiling->useExp2)));
            Mte3WriteL1Right(L1Right(args.memory->l1, head.hSlot), v_new_g);
            StoreAlpha(UbAlpha(*args.memory, head), LastGate(gate, plan.chunk.validTokens, args.tiling->useExp2));
            args.sync->Set(EventKind::RightReady, head.hSlot, /*S1 MTE3 生产者*/ 1, /*S2 消费者*/ 2);
            result.alphaReady = true;
        } else if (plan.gateMode == GateMode::KeyWiseGk && needsRight) {
            // gk-only 在 S1 不读取 gk；相对当前 chunk 的衰减已经包含在 kg 中。
            Mte3WriteL1Right(L1Right(args.memory->l1, head.hSlot), v_new);
            args.sync->Set(EventKind::RightReady, head.hSlot, /*S1 MTE3 生产者*/ 1, /*S2 消费者*/ 2);
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

} // 命名空间 fwd_h_pseudocode
