// 仅伪代码。arch35（A5）Vec 阶段的独立 RegBase 实现。
// A5 不复用 arch22 的 Vec 入口；每个阶段都遵循 MTE2 -> 一个完整 head 的 VF -> MTE3。
// RegBase API 的写法按昇腾 Reg 矢量计算基础 API 组织，落地时再核对目标 CANN 版本的精确重载。

#pragma once

#include "../chunk_gated_delta_rule_fwd_h_policy.h"
#include "../chunk_gated_delta_rule_fwd_h_utils.h"

namespace fwd_h_pseudocode {

struct VecStageArgs {
    const ApiInputs* in = nullptr;
    const ApiOutputs* out = nullptr;
    WorkspaceRefs* workspace = nullptr;
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
    bool roundBoundaryDrained = false;
};

struct VecStageResult {
    bool rightGmReady = false;
    bool alphaReady = false;
    bool nextHReady = false;
    bool finalStateWritten = false;
    int activeTaskCount = 0;
};

namespace a5_regbase {

// RegTensor 一次处理一个 VL；下面的循环只是 VF 内部的分块，不会产生第二个 VF。
constexpr uint32_t kRegBytes = 256;
constexpr uint32_t kFp32PerReg = kRegBytes / sizeof(float);
constexpr uint32_t kBf16PerReg = kRegBytes / sizeof(bfloat16_t);
constexpr uint32_t kStateElements = kKeyDim * kValueDim;
constexpr float kLn2 = 0.6931471805599453f;

inline void RegBaseCastFp32ToBf16(AscendC::Reg::RegTensor<bfloat16_t>& dst,
                                  AscendC::Reg::RegTensor<float>& src,
                                  AscendC::Reg::MaskReg& mask)
{
    // 伪代码：对应目标 CANN 的 Reg::Cast FP32 -> BF16 重载。
    AscendC::Reg::Cast(dst, src, mask);
}

inline void RegBaseCastBf16ToFp32(AscendC::Reg::RegTensor<float>& dst,
                                  AscendC::Reg::RegTensor<bfloat16_t>& src,
                                  AscendC::Reg::MaskReg& mask)
{
    // 伪代码：对应目标 CANN 的 Reg::Cast BF16 -> FP32 重载。
    AscendC::Reg::Cast(dst, src, mask);
}

inline void RegBaseLoadGate(AscendC::Reg::RegTensor<float>& gate,
                            __ubuf__ const float* gateUb,
                            uint32_t tokenIndex,
                            AscendC::Reg::MaskReg& mask)
{
    // g 是每 token 一个标量；RegBase 通过加载加广播扩展到该 token 的 V 维。
    AscendC::Reg::LoadAlign(gate, gateUb + tokenIndex);
    AscendC::Reg::Duplicate(gate, gate[0], mask);
}

inline void RegBaseLoadGkLast(AscendC::Reg::RegTensor<float>& gk,
                              __ubuf__ const float* gkUb,
                              uint32_t keyIndex,
                              AscendC::Reg::MaskReg& mask)
{
    // state 的 canonical 顺序是 [K,V]，所以每个 K 行复用一个 gk[M-1,k] 标量。
    // 尾 chunk 的无效 token 已由 MTE2 清零；这里用 RegBase 广播到该 K 行的 V 元素。
    AscendC::Reg::LoadAlign(gk, gkUb + keyIndex);
    AscendC::Reg::Duplicate(gk, gk[0], mask);
}

inline void RegBaseLoadAlpha(AscendC::Reg::RegTensor<float>& alpha,
                             __ubuf__ const float* alphaUb,
                             AscendC::Reg::MaskReg& mask)
{
    // alpha 是每个 head 一个标量，Stage3 对整个 [K,V] 广播同一个 lambda。
    AscendC::Reg::LoadAlign(alpha, alphaUb);
    AscendC::Reg::Duplicate(alpha, alpha[0], mask);
}

inline void RegBaseApplyGateDelta(AscendC::Reg::RegTensor<float>& value,
                                  AscendC::Reg::RegTensor<float>& gate,
                                  bool useExp2,
                                  AscendC::Reg::MaskReg& mask)
{
    // 伪代码：按 use_exp2 选择 exp/exp2，计算 value *= (1 - exp(g))。
    // 这些 Reg 算子必须在同一 VF 内完成，不能拆成额外的 vector pass。
    AscendC::Reg::RegTensor<float> one;
    AscendC::Reg::RegTensor<float> decay;
    AscendC::Reg::Duplicate(one, 1.0f, mask);
    if (useExp2) {
        AscendC::Reg::Muls(gate, gate, kLn2, mask);
        AscendC::Reg::Exp(decay, gate, mask);
    } else {
        AscendC::Reg::Exp(decay, gate, mask);
    }
    AscendC::Reg::Sub(decay, one, decay, mask);
    AscendC::Reg::Mul(value, value, decay, mask);
}

inline void RegBaseApplyStateGate(AscendC::Reg::RegTensor<float>& state,
                                  AscendC::Reg::RegTensor<float>& gate,
                                  bool useExp2,
                                  AscendC::Reg::MaskReg& mask)
{
    // Stage3 使用 lambda = exp(g_last)，不是 Stage1 的 delta = 1 - exp(g)。
    if (useExp2) {
        AscendC::Reg::Muls(gate, gate, kLn2, mask);
        AscendC::Reg::Exp(gate, gate, mask);
    } else {
        AscendC::Reg::Exp(gate, gate, mask);
    }
    AscendC::Reg::Mul(state, state, gate, mask);
}

inline void RegBaseApplyStateScale(AscendC::Reg::RegTensor<float>& state,
                                   AscendC::Reg::RegTensor<float>& alpha,
                                   AscendC::Reg::MaskReg& mask)
{
    // g-only 的 alpha 已在 Stage1 VF 中变成 lambda，Stage3 不能再次做 exp。
    AscendC::Reg::Mul(state, state, alpha, mask);
}

inline void RegBaseWriteLastAlpha(__ubuf__ float* alphaUb,
                                  __ubuf__ const float* gateUb,
                                  uint32_t validTokens,
                                  bool useExp2)
{
    // alpha 仅供同一 head 的 S3 使用，按最后有效 token 写入一个 FP32 标量。
    RegBaseWriteAlphaScalar(alphaUb, gateUb + validTokens - 1, useExp2);
}

// S-1 的唯一 VF：FP32 initial -> BF16 H0。内部始终按 canonical [K,V] 处理。
__simd_vf__ inline void SMinusOneFp32ToBf16Vf(__ubuf__ const float* srcUb,
                                              __ubuf__ bfloat16_t* dstUb,
                                              uint32_t elementCount,
                                              bool stateVFirst)
{
    uint32_t remaining = elementCount;
    uint32_t offset = 0;
    while (remaining > 0) {
        auto fp32Mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::RegTensor<float> srcReg;
        AscendC::Reg::RegTensor<bfloat16_t> dstReg;
        AscendC::Reg::LoadAlign(srcReg, srcUb + offset);
        RegBaseCastFp32ToBf16(dstReg, srcReg, fp32Mask);
        auto bf16Mask = AscendC::Reg::CreateMask<bfloat16_t>();
        AscendC::Reg::StoreAlign(dstUb + offset, dstReg, bf16Mask);
        offset += kFp32PerReg;
        remaining = remaining > kFp32PerReg ? remaining - kFp32PerReg : 0;
    }
    // state_v_first 只由 MTE3 layout-aware 描述符处理，VF 内不做转置。
    (void)stateVFirst;
}

// Stage1 的唯一 VF：U-P、BF16 转换、g/gk 分支派生、alpha，以及首 chunk H0。
__simd_vf__ inline void Stage1FullHeadVf(
    __ubuf__ const bfloat16_t* uUb,
    __ubuf__ const float* pUb,
    __ubuf__ const float* gateUb,
    __ubuf__ const bfloat16_t* initialStateUb,
    __ubuf__ float* vNewFp32Ub,
    __ubuf__ bfloat16_t* vNewBf16Ub,
    __ubuf__ bfloat16_t* rightBf16Ub,
    __ubuf__ float* alphaUb,
    __ubuf__ bfloat16_t* h0Ub,
    uint32_t validTokens,
    bool hasP,
    bool scalarG,
    bool writeRight,
    bool writeH0,
    bool zeroH0,
    bool useExp2)
{
    if (writeH0) {
        uint32_t stateRemaining = kStateElements;
        uint32_t stateOffset = 0;
        while (stateRemaining > 0) {
            auto stateMask = AscendC::Reg::UpdateMask<bfloat16_t>(stateRemaining);
            AscendC::Reg::RegTensor<bfloat16_t> stateReg;
            if (zeroH0) {
                AscendC::Reg::Duplicate(stateReg, static_cast<bfloat16_t>(0), stateMask);
            } else {
                AscendC::Reg::LoadAlign(stateReg, initialStateUb + stateOffset);
            }
            AscendC::Reg::StoreAlign(h0Ub + stateOffset, stateReg, stateMask);
            stateOffset += kBf16PerReg;
            stateRemaining = stateRemaining > kBf16PerReg
                                 ? stateRemaining - kBf16PerReg : 0;
        }
    }

    uint32_t remaining = validTokens * kValueDim;
    uint32_t offset = 0;
    while (remaining > 0) {
        auto bf16Mask = AscendC::Reg::UpdateMask<bfloat16_t>(remaining);
        auto fp32Mask = AscendC::Reg::CreateMask<float>();
        AscendC::Reg::RegTensor<bfloat16_t> uBf16;
        AscendC::Reg::RegTensor<float> uFp32;
        AscendC::Reg::RegTensor<float> pFp32;
        AscendC::Reg::RegTensor<float> vFp32;
        AscendC::Reg::RegTensor<bfloat16_t> vBf16;
        AscendC::Reg::LoadAlign(uBf16, uUb + offset);
        RegBaseCastBf16ToFp32(uFp32, uBf16, bf16Mask);
        if (hasP) {
            AscendC::Reg::LoadAlign(pFp32, pUb + offset);
        } else {
            AscendC::Reg::Duplicate(pFp32, 0.0f, fp32Mask);
        }
        AscendC::Reg::Sub(vFp32, uFp32, pFp32, fp32Mask);
        AscendC::Reg::StoreAlign(vNewFp32Ub + offset, vFp32, fp32Mask);
        RegBaseCastFp32ToBf16(vBf16, vFp32, fp32Mask);
        AscendC::Reg::StoreAlign(vNewBf16Ub + offset, vBf16, bf16Mask);

        if (writeRight) {
            if (scalarG) {
                AscendC::Reg::RegTensor<float> gateReg;
                RegBaseLoadGate(gateReg, gateUb, offset / kValueDim, fp32Mask);
                RegBaseApplyGateDelta(vFp32, gateReg, useExp2, fp32Mask);
                RegBaseCastFp32ToBf16(vBf16, vFp32, fp32Mask);
            }
            AscendC::Reg::StoreAlign(rightBf16Ub + offset, vBf16, bf16Mask);
        }
        offset += kFp32PerReg;
        remaining = remaining > kFp32PerReg ? remaining - kFp32PerReg : 0;
    }
    if (writeRight && scalarG) {
        RegBaseWriteLastAlpha(alphaUb, gateUb, validTokens, useExp2);
    }
}

// Stage3 的唯一 VF：gate(state)、加 D、状态写回和下一 H 的 BF16 派生。
__simd_vf__ inline void Stage3FullHeadVf(
    __ubuf__ bfloat16_t* stateBf16Ub,
    __ubuf__ float* stateFp32Ub,
    __ubuf__ const float* dUb,
    __ubuf__ const float* alphaUb,
    __ubuf__ const float* gkLastUb,
    __ubuf__ bfloat16_t* hNextUb,
    uint32_t elementCount,
    StateType stateType,
    GateMode gateMode,
    bool writeHNext,
    bool useExp2,
    bool stateVFirst)
{
    uint32_t remaining = elementCount;
    uint32_t offset = 0;
    while (remaining > 0) {
        auto fp32Mask = AscendC::Reg::UpdateMask<float>(remaining);
        auto bf16Mask = AscendC::Reg::CreateMask<bfloat16_t>();
        AscendC::Reg::RegTensor<float> stateReg;
        AscendC::Reg::RegTensor<float> dReg;
        AscendC::Reg::RegTensor<float> gateReg;
        AscendC::Reg::RegTensor<float> resultReg;
        AscendC::Reg::RegTensor<bfloat16_t> resultBf16;
        if (stateType == StateType::Bf16) {
            AscendC::Reg::RegTensor<bfloat16_t> stateBf16;
            AscendC::Reg::LoadAlign(stateBf16, stateBf16Ub + offset);
            RegBaseCastBf16ToFp32(stateReg, stateBf16, bf16Mask);
        } else {
            AscendC::Reg::LoadAlign(stateReg, stateFp32Ub + offset);
        }
        AscendC::Reg::LoadAlign(dReg, dUb + offset);
        if (gateMode == GateMode::ScalarG) {
            RegBaseLoadAlpha(gateReg, alphaUb, fp32Mask);
            RegBaseApplyStateScale(stateReg, gateReg, fp32Mask);
        } else {
            RegBaseLoadGkLast(gateReg, gkLastUb, offset / kValueDim, fp32Mask);
            RegBaseApplyStateGate(stateReg, gateReg, useExp2, fp32Mask);
        }
        AscendC::Reg::Add(resultReg, stateReg, dReg, fp32Mask);
        if (stateType == StateType::Bf16) {
            RegBaseCastFp32ToBf16(resultBf16, resultReg, fp32Mask);
            AscendC::Reg::StoreAlign(stateBf16Ub + offset, resultBf16, bf16Mask);
            if (writeHNext) {
                AscendC::Reg::StoreAlign(hNextUb + offset, resultBf16, bf16Mask);
            }
        } else {
            AscendC::Reg::StoreAlign(stateFp32Ub + offset, resultReg, fp32Mask);
            if (writeHNext) {
                RegBaseCastFp32ToBf16(resultBf16, resultReg, fp32Mask);
                AscendC::Reg::StoreAlign(hNextUb + offset, resultBf16, bf16Mask);
            }
        }
        offset += kFp32PerReg;
        remaining = remaining > kFp32PerReg ? remaining - kFp32PerReg : 0;
    }
    // 仅用于保护 VF 内同一 UB 槽的先存后读；MTE2/MTE3 依赖由外层事件负责。
    LocalMemBar<VEC_STORE, VEC_LOAD>();
    (void)stateVFirst;
}

} // namespace a5_regbase

// ------------------------------- S-1 -------------------------------

inline void A5SMinusOneLoadInitial(const SMinusOneArgs& args, const HeadBinding& head)
{
    const int bank = FixedMemory::LocalBank(head);
    auto& input = args.memory->initialInput[bank];
    if (input.generation > 0 && !args.roundBoundaryDrained) {
        args.sync->Wait(EventKind::InitialInputFree, bank, /*S-1 MTE2*/ 0);
    }
    args.memory->AcquireInitialInput(head);
    Mte2StateToFp32UbAsync(InitialStateAt(*args.in, args.sequence->sequence, head.hv),
                           UbInitialInput(*args.memory, head), args.tiling->stateLayout);
    args.memory->MarkInitialInputReady(head);
    args.sync->Set(EventKind::InitialInputReady, bank, /*S-1 MTE2*/ 0, /*S-1 VF*/ 1);
}

inline void A5SMinusOneConvertAndWriteH0(const SMinusOneArgs& args,
                                         const HeadBinding& head)
{
    const int bank = FixedMemory::LocalBank(head);
    args.sync->Wait(EventKind::InitialInputReady, bank, /*S-1 VF*/ 1);
    args.memory->AcquireInitialHOutput(head);
    args.memory->ProduceInitialH(head.hSlot);
    // A5 S-1 的 vector 计算只有这一处 VF 调用。
    a5_regbase::SMinusOneFp32ToBf16Vf(
        UbInitialInputRaw(*args.memory, head),
        UbInitialHOutputRaw(*args.memory, head),
        a5_regbase::kStateElements,
        args.tiling->stateLayout == StateLayout::VK);
    PipeBarrierVForA5RegBase();
    args.memory->ReleaseInitialInput(head);
    args.sync->Release(EventKind::InitialInputFree, bank, /*S-1 VF*/ 1);
    Mte3WriteH0LayoutAwareAsync(*args.out, *args.tiling, *args.sequence, head.hv,
                                UbInitialHOutput(*args.memory, head));
    wait_mte3_h0_done(bank);
    args.memory->MarkHReady(head.hSlot);
    args.memory->MarkInitialHOutputReady(head);
}

inline void RunSMinusOneArch35(const SMinusOneArgs& args)
{
    if (!args.tiling->useInitialState || args.tiling->stateType != StateType::Fp32) {
        return;
    }
    for (int local = 0; local < args.activeHvCount; ++local) {
        HeadBinding head{};
        head.roundHead = local;
        head.hv = args.activeHvBegin + local;
        head.hSlot = local;
        head.aiv = local % kAivCount;
        head.localSlot = local / kAivCount;
        A5SMinusOneLoadInitial(args, head);
        A5SMinusOneConvertAndWriteH0(args, head);
    }
    args.sync->Set(EventKind::InitialPhaseDrain, args.sequence->sequence,
                   /*S-1 MTE3*/ 1, /*调度器*/ -1);
    for (int local = 0; local < args.activeHvCount; ++local) {
        args.sync->Set(EventKind::HGmReady, local, /*S-1 phase*/ 1, /*S0 MTE2*/ 0);
        HeadBinding head{};
        head.roundHead = local;
        head.hv = args.activeHvBegin + local;
        head.hSlot = local;
        head.aiv = local % kAivCount;
        head.localSlot = local / kAivCount;
        args.memory->ReleaseInitialHOutput(head);
    }
}

// ------------------------------- Stage1 -------------------------------

inline void A5Stage1PrepareInitialOrZeroState(const VecStageArgs& args,
                                              const HeadBinding& head)
{
    const RoundPlan& plan = *args.plan;
    if (!plan.chunk.first || args.tiling->stateType != StateType::Bf16) {
        return;
    }
    if (plan.finalVNewOnly) {
        // 最终 v_new-only 不建立 rolling state；H0 只写到独立 scratch，供本次 MTE3 消费。
        if (args.tiling->useInitialState) {
            Mte2StateToBf16Ub(InitialStateAt(*args.in, plan.sequence, head.hv),
                              UbH0Scratch(*args.memory, head), args.tiling->stateLayout);
        } else {
            Mte2InitConstValueAsync(UbH0Scratch(*args.memory, head), /*zero*/ 0,
                                    /*bytes*/ 32 * 1024);
        }
        args.sync->Set(EventKind::StateReady, FixedMemory::LocalBank(head),
                       /*S1 MTE2*/ 0, /*S1 VF*/ 1);
        return;
    }
    args.memory->InitializeBf16StateInS1(head);
    if (args.tiling->useInitialState) {
        Mte2StateToBf16Ub(InitialStateAt(*args.in, plan.sequence, head.hv),
                          UbBf16State(*args.memory, head), args.tiling->stateLayout);
    } else {
        Mte2InitConstValueAsync(UbBf16State(*args.memory, head), /*zero*/ 0,
                                /*bytes*/ 32 * 1024);
    }
    args.sync->Set(EventKind::StateReady, FixedMemory::LocalBank(head),
                   /*S1 MTE2*/ 0, /*S1 VF*/ 1);
}

inline void A5Stage1LoadUAndGate(const VecStageArgs& args, const HeadBinding& head)
{
    const int bank = FixedMemory::LocalBank(head);
    auto& ticket = args.memory->vNewWork[bank];
    if (ticket.generation > 0 && !args.plan->roundBoundaryDrained) {
        args.sync->Wait(EventKind::VNewWorkFree, bank, /*S1 MTE2*/ 0);
    }
    args.memory->AcquireVNewWorkForS1(head);
    Mte2InitConstValueAsync(UbVNewWork(*args.memory, head, args.tiling->stateType),
                            /*zero*/ 0, /*bytes*/ 16 * 1024);
    Mte2CopyValidRowsAsync(UAt(*args.in, args.plan->chunk, head.hv),
                           UbVNewWork(*args.memory, head, args.tiling->stateType),
                           args.plan->chunk.validTokens);
    if (args.plan->gateMode == GateMode::ScalarG &&
        args.variant != Stage1Variant::v_new_only) {
        Mte2CopyGateValidRowsAsync(GAt(*args.in, args.plan->chunk, head.hv),
                                   UbGate(*args.memory, head),
                                   args.plan->chunk.validTokens);
    }
    WaitMte2ToVForStage1(head);
    args.memory->MarkVNewWorkReady(head);
}

inline void A5Stage1ComputeAndWrite(const VecStageArgs& args, const HeadBinding& head)
{
    const RoundPlan& plan = *args.plan;
    const bool hasP = plan.stage0Required;
    if (hasP) {
        args.sync->Wait(EventKind::PReady, head.roundHead, /*S1 VF*/ 1);
    }
    const bool writeBf16H0 = plan.chunk.first && args.tiling->stateType == StateType::Bf16;
    const bool writeFp32ZeroH0 = plan.chunk.first &&
                                 args.tiling->stateType == StateType::Fp32 &&
                                 !args.tiling->useInitialState;
    const bool writeH0 = writeBf16H0 || writeFp32ZeroH0;
    const bool zeroH0 = writeH0 && !args.tiling->useInitialState;
    const bool writeRight = plan.stage2Required;
    const bool scalarG = plan.gateMode == GateMode::ScalarG;
    if (writeFp32ZeroH0) {
        args.memory->AcquireLocalDataForH0(head);
    }
    if (writeBf16H0) {
        args.sync->Wait(EventKind::StateReady, FixedMemory::LocalBank(head),
                        /*S1 VF*/ 1);
    }
    const auto* stateBf16 = args.tiling->stateType == StateType::Bf16
                                ? (plan.finalVNewOnly
                                       ? UbH0ScratchRaw(*args.memory, head)
                                       : UbBf16StateRaw(*args.memory, head))
                                : nullptr;
    const auto* h0Target = writeFp32ZeroH0
                               ? UbH0WriteTargetRaw(*args.memory, head)
                               : (plan.finalVNewOnly ? UbH0ScratchRaw(*args.memory, head)
                                                     : UbBf16StateRaw(*args.memory, head));

    // A5 Stage1 的所有 vector 计算集中在一个完整 head VF。
    a5_regbase::Stage1FullHeadVf(
        UbUbf16Raw(*args.memory, head),
        hasP ? UbPRaw(*args.memory, head) : nullptr,
        scalarG ? UbGateRaw(*args.memory, head) : nullptr,
        stateBf16,
        UbVNewFp32Raw(*args.memory, head),
        UbVNewBf16Raw(*args.memory, head),
        writeRight ? UbRightBf16Raw(*args.memory, head) : nullptr,
        scalarG ? UbAlphaRaw(*args.memory, head) : nullptr,
        writeH0 ? h0Target : nullptr,
        plan.chunk.validTokens, hasP, scalarG, writeRight, writeH0, zeroH0,
        args.tiling->useExp2);
    PipeBarrierVForA5RegBase();

    write_v_new_from_ub(*args.out, plan.chunk, head.hv,
                        UbVNewBf16(*args.memory, head, args.tiling->stateType));
    if (writeRight) {
        // UB right 是 ND；Stage1 只写 roundHead 对应的 GM scratch，禁止直接 UB -> L1。
        const int rightSlot = head.roundHead;
        if (args.memory->rightGm[rightSlot].generation > 0 && !plan.roundBoundaryDrained) {
            args.sync->Wait(EventKind::RightGmFree, rightSlot, /*下一 chunk S1 MTE3*/ 1);
        }
        args.memory->AcquireRightGmForS1(rightSlot);
        Mte3WriteRightOperandGmNdAsync(
            args.workspace->rightOperandGm, rightSlot, plan.chunk,
            UbRightBf16(*args.memory, head), plan.chunk.validTokens);
        wait_mte3_right_gm_done(rightSlot);
        args.memory->MarkRightGmReady(rightSlot);
        args.sync->Set(EventKind::RightGmReady, rightSlot, /*S1 MTE3*/ 1, /*S2 MTE2*/ 0);
    }
    if (hasP) {
        args.memory->ReleasePAfterS1(head);
        args.sync->Release(EventKind::PFree, head.roundHead, /*S1 VF*/ 1);
    }
    if (writeH0) {
        Mte3WriteH0LayoutAwareAsync(*args.out, *args.tiling, plan, head,
                                    writeFp32ZeroH0
                                        ? UbH0WriteTarget(*args.memory, head)
                                        : (plan.finalVNewOnly
                                               ? UbH0Scratch(*args.memory, head)
                                               : UbBf16State(*args.memory, head)));
        wait_mte3_h0_done(FixedMemory::LocalBank(head));
        if (writeFp32ZeroH0) {
            args.memory->ReleaseH0AfterMte3(head);
            if (plan.stage2Required) {
                args.sync->Set(EventKind::LocalDataFree, head.roundHead,
                               /*S1 H0 MTE3*/ 1, /*本 chunk S2*/ 2);
            } else if (plan.nextRoundStartsWithS0) {
                args.sync->Set(EventKind::UnionFree, head.localSlot,
                               /*S1 H0 MTE3*/ 1, /*下一 round S0*/ 0);
            } else {
                PipeBarrierVForH0Union(head);
            }
        }
    }
}

inline void A5Stage1ReleaseWork(const VecStageArgs& args, const HeadBinding& head)
{
    const int bank = FixedMemory::LocalBank(head);
    wait_mte3_vnew_done(args.plan->chunk, head.hv);
    args.memory->ReleaseVNewWorkAfterMte3(head);
    if (args.plan->hasNextChunk || args.plan->hasNextHeadRound) {
        args.sync->Set(EventKind::VNewWorkFree, bank, /*S1 MTE3*/ 1, /*下一 producer*/ -1);
    }
}

inline VecStageResult RunStage1Arch35(const VecStageArgs& args)
{
    VecStageResult result{};
    const RoundPlan& plan = *args.plan;
    for (int i = 0; i < plan.activeHvCount; ++i) {
        const HeadBinding& head = plan.heads[i];
        A5Stage1PrepareInitialOrZeroState(args, head);
        A5Stage1LoadUAndGate(args, head);
        A5Stage1ComputeAndWrite(args, head);
        A5Stage1ReleaseWork(args, head);
        ++result.activeTaskCount;
        result.rightGmReady = result.rightGmReady || plan.stage2Required;
        result.alphaReady = result.alphaReady ||
                            (plan.gateMode == GateMode::ScalarG && plan.stage2Required);
    }
    return result;
}

// ------------------------------- Stage3 -------------------------------

inline void A5Stage3PrepareState(const VecStage3Args& args, const HeadBinding& head)
{
    const RoundPlan& plan = *args.plan;
    if (args.tiling->stateType == StateType::Bf16) {
        auto& state = args.memory->bf16State[FixedMemory::LocalBank(head)];
        if (state.owner == StateOwner::RNextMte3) {
            args.sync->Wait(EventKind::StateToVFree, FixedMemory::LocalBank(head),
                            /*下一 S3 VF*/ 3);
            args.memory->MarkBf16StateConsumedByNextVf(head);
        }
        return;
    }
    if (args.memory->fp32StateScratch.owner == StateOwner::RNextMte3) {
        args.sync->Wait(EventKind::StateToMte2Free, 0, /*state MTE2*/ 0);
        args.memory->ReleaseFp32StateScratch();
    }
    args.memory->AcquireFp32StateScratchForS3();
    if (plan.chunk.first && !args.tiling->useInitialState) {
        Mte2InitConstValueAsync(UbFp32StateScratch(*args.memory), /*zero*/ 0,
                                /*bytes*/ 64 * 1024);
    } else {
        Mte2ReadRollingStateLayoutAware(
            RollingStateSource(*args.in, *args.out, *args.tiling, plan, head),
            UbFp32StateScratch(*args.memory), args.tiling->stateLayout);
    }
    args.memory->MarkFp32StateReady();
    args.sync->Set(EventKind::StateReady, 0, /*state MTE2*/ 0, /*S3 VF*/ 3);
}

inline void A5Stage3LoadGate(const VecStage3Args& args, const HeadBinding& head)
{
    if (args.plan->gateMode != GateMode::KeyWiseGk) {
        return;
    }
    Mte2CopyGkLastAsync(GkAtLast(*args.in, args.plan->chunk, head.hv),
                        UbGkLast(*args.memory, head));
    WaitMte2ToVForStage3(head);
}

inline void A5Stage3ComputeAndWrite(const VecStage3Args& args,
                                    const HeadBinding& head)
{
    const RoundPlan& plan = *args.plan;
    args.sync->Wait(EventKind::DReady, head.roundHead, /*S3 VF*/ 3);
    if (args.tiling->stateType == StateType::Fp32 || plan.chunk.first) {
        args.sync->Wait(EventKind::StateReady,
                        args.tiling->stateType == StateType::Bf16
                            ? FixedMemory::LocalBank(head) : 0,
                        /*S3 VF*/ 3);
    }

    // A5 Stage3 的 gate、D、状态更新、H 派生全部由一个 RegBase VF 完成。
    a5_regbase::Stage3FullHeadVf(
        args.tiling->stateType == StateType::Bf16
            ? UbBf16StateRaw(*args.memory, head) : nullptr,
        args.tiling->stateType == StateType::Fp32
            ? UbFp32StateScratchRaw(*args.memory) : nullptr,
        UbDRaw(*args.memory, head),
        plan.gateMode == GateMode::ScalarG ? UbAlphaRaw(*args.memory, head) : nullptr,
        plan.gateMode == GateMode::KeyWiseGk ? UbGkLastRaw(*args.memory, head) : nullptr,
        plan.chunk.last ? nullptr : UbHWriteTargetRaw(*args.memory, head),
        a5_regbase::kStateElements, args.tiling->stateType, plan.gateMode,
        !plan.chunk.last, args.tiling->useExp2,
        args.tiling->stateLayout == StateLayout::VK);
    PipeBarrierVForA5RegBase();

    if (!plan.chunk.last) {
        args.memory->ProduceHForS3(head.hSlot);
        Mte3WriteHLayoutAwareAsync(*args.out, *args.tiling, plan, head,
                                   UbHWriteTarget(*args.memory, head));
        wait_mte3_hnext_done(head.hSlot);
        args.memory->MarkHReady(head.hSlot);
        args.sync->Set(EventKind::HGmReady, head.hSlot, /*S3 MTE3*/ 1,
                       /*下一 chunk S0 MTE2*/ 0);
    } else if (args.tiling->outputFinalState) {
        Mte3WriteFinalStateLayoutAwareAsync(
            *args.out, *args.tiling, plan, head,
            args.tiling->stateType == StateType::Bf16
                ? UbBf16State(*args.memory, head)
                : UbFp32StateScratch(*args.memory));
        wait_mte3_final_state_done(head.hv);
    }
}

inline void A5Stage3ReleaseStateAndD(const VecStage3Args& args,
                                     const HeadBinding& head)
{
    const RoundPlan& plan = *args.plan;
    if (args.tiling->stateType == StateType::Bf16) {
        if (plan.hasNextChunk) {
            args.memory->MarkBf16StateMte3InFlight(head);
            args.sync->Set(EventKind::StateToVFree, FixedMemory::LocalBank(head),
                           /*state MTE3*/ 1, /*下一 S3 VF*/ 3);
        } else if (plan.nextRoundStartsWithS0) {
            args.memory->MarkBf16StateMte3InFlight(head);
            args.sync->Set(EventKind::StateToMte2Free, FixedMemory::LocalBank(head),
                           /*state MTE3*/ 1, /*下一 S1 MTE2*/ 0);
        } else if (plan.nextRoundStartsWithS1NoP) {
            args.memory->MarkBf16StateMte3InFlight(head);
            args.sync->Set(EventKind::StateToVFree, FixedMemory::LocalBank(head),
                           /*state MTE3*/ 1, /*下一 S1 VF*/ 1);
        } else {
            args.memory->ReleaseBf16StateAtTerminal(head);
        }
    } else {
        const bool nextHeadUsesScratch = head.roundHead + 1 < plan.activeHvCount;
        if (plan.hasNextChunk || plan.hasNextHeadRound || nextHeadUsesScratch) {
            args.memory->MarkFp32StateMte3InFlight();
            args.sync->Set(EventKind::StateToMte2Free, 0, /*state MTE3*/ 1,
                           /*下一 state MTE2*/ -1);
        } else {
            args.memory->ReleaseFp32StateScratch();
        }
    }
    args.memory->ReleaseDAfterS3(head);
    args.sync->Release(EventKind::DFree, head.roundHead, /*S3 VF*/ 3);
    if (plan.gateMode == GateMode::KeyWiseGk) {
        ReleaseUbGkLast(*args.memory, head);
    }
}

inline VecStageResult RunStage3Arch35(const VecStage3Args& args)
{
    VecStageResult result{};
    const RoundPlan& plan = *args.plan;
    if (!plan.stage3Required) {
        return result;
    }
    for (int i = 0; i < plan.activeHvCount; ++i) {
        const HeadBinding& head = plan.heads[i];
        A5Stage3PrepareState(args, head);
        A5Stage3LoadGate(args, head);
        A5Stage3ComputeAndWrite(args, head);
        A5Stage3ReleaseStateAndD(args, head);
        ++result.activeTaskCount;
        result.nextHReady = result.nextHReady || !plan.chunk.last;
        result.finalStateWritten = result.finalStateWritten ||
                                   (plan.chunk.last && args.tiling->outputFinalState);
    }
    return result;
}

} // 命名空间 fwd_h_pseudocode
