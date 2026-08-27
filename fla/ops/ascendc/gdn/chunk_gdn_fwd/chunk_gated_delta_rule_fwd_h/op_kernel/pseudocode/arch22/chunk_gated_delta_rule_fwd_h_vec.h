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
    bool roundBoundaryDrained = false;
};

struct VecStageResult {
    bool rightOperandReady = false;
    bool alphaReady = false;
    bool nextHReady = false;
    bool finalStateWritten = false;
    int activeTaskCount = 0;
};

// ------------------------------- S-1 helpers -------------------------------

inline void SMinusOneLoadInitial(const SMinusOneArgs& args, const HeadBinding& head)
{
    const int bank = FixedMemory::LocalBank(head);
    auto& input = args.memory->initialInput[bank];
    if (input.generation > 0 && !args.roundBoundaryDrained) {
        args.sync->Wait(EventKind::InitialInputFree, bank, /*S-1 MTE2*/ 0);
    }
    args.memory->AcquireInitialInput(head);

    // FP32 initial 只完整搬入一次；layout-aware 描述符按 state_v_first 选择 GM 访问顺序。
    Mte2StateToFp32UbAsync(
        /*源=*/InitialStateAt(*args.in, args.sequence->sequence, head.hv),
        /*目的=*/UbInitialInput(*args.memory, head), args.tiling->stateLayout);
    args.memory->MarkInitialInputReady(head);
    args.sync->Set(EventKind::InitialInputReady, bank, /*S-1 MTE2*/ 0,
                   /*S-1 VF*/ 1);
}

inline void SMinusOneConvertAndWriteH0(const SMinusOneArgs& args, const HeadBinding& head)
{
    const int bank = FixedMemory::LocalBank(head);
    args.sync->Wait(EventKind::InitialInputReady, bank, /*S-1 VF*/ 1);
    args.memory->AcquireInitialHOutput(head);
    args.memory->ProduceInitialH(head.hSlot);

    // 一次完整 VF：FP32 initial -> BF16 H0，内部 canonical [K,V]，不拆行或增加 pass。
    const auto h0 = VfCastFp32ToBf16Full(UbInitialInput(*args.memory, head));
    VfLastReadInitialInput(head);
    args.memory->ReleaseInitialInput(head);
    args.sync->Release(EventKind::InitialInputFree, bank, /*S-1 VF*/ 1);

    // 同一份 h0 同时写公开 h0 和 L1 resident；两路 MTE3 完成前不能释放输出 bank。
    Mte3WriteH0LayoutAwareAsync(*args.out, *args.tiling, *args.sequence, head.hv, h0);
    Mte3WriteL1ResidentAsync(L1H(args.memory->l1, head.hSlot), h0);
    wait_mte3_h0_done(bank);
    args.memory->MarkHReady(head.hSlot);
    args.memory->MarkInitialHOutputReady(head);
}

inline void RunSMinusOneArch22(const SMinusOneArgs& args)
{
    if (!args.tiling->useInitialState || args.tiling->stateType != StateType::Fp32) {
        // BF16 initial 由首个 S0 直接读取；无 initial 不建立 S-1 task。
        return;
    }

    for (int local = 0; local < args.activeHvCount; ++local) {
        HeadBinding head{};
        head.roundHead = local;
        head.hv = args.activeHvBegin + local;
        head.hSlot = local;
        head.aiv = local % kAivCount;
        head.localSlot = local / kAivCount;
        SMinusOneLoadInitial(args, head);
        SMinusOneConvertAndWriteH0(args, head);
    }

    // 所有 active head 的初态输入/H0 两路搬运都 drain 后，才统一向 AIC 发布 HReady。
    args.sync->Set(EventKind::InitialPhaseDrain, args.sequence->sequence,
                   /*S-1 MTE3*/ 1, /*调度器*/ -1);
    for (int local = 0; local < args.activeHvCount; ++local) {
        args.sync->Set(EventKind::HReady, local, /*S-1 phase*/ 1, /*S0 MTE1*/ 1);
        args.memory->ReleaseInitialHOutput(
            HeadBinding{local, args.activeHvBegin + local, -1, -1, local,
                        local, local % kAivCount, local / kAivCount, true});
    }
}

// ------------------------------- Stage 1 helpers -------------------------------

inline void Stage1PrepareInitialOrZeroState(const VecStageArgs& args, const HeadBinding& head)
{
    const RoundPlan& plan = *args.plan;
    if (!plan.chunk.first) {
        // 后续 chunk 的 BF16 R_c 已经由前一个 S3 驻留在同一个 UB state bank。
        return;
    }

    if (args.tiling->stateType == StateType::Bf16) {
        if (plan.finalVNewOnly) {
            // 最终 v_new-only 没有 S3 消费者：首 chunk 的 H0 只用临时源写出，不建立
            // 跨 chunk 的 R resident，也不发布 StateReady/StateFree。
            if (args.tiling->useInitialState) {
                Mte2StateToBf16Ub(
                    /*源=*/InitialStateAt(*args.in, plan.sequence, head.hv),
                    /*目的=*/UbH0Scratch(*args.memory, head), args.tiling->stateLayout);
                Mte3WriteH0LayoutAwareAsync(*args.out, *args.tiling, plan, head,
                                            UbH0Scratch(*args.memory, head));
                wait_mte3_h0_done(FixedMemory::LocalBank(head));
            } else {
                Mte3WriteH0LayoutAwareAsync(*args.out, *args.tiling, plan, head,
                                            VfZeroH0Bf16(head));
                wait_mte3_h0_done(FixedMemory::LocalBank(head));
            }
            return;
        }
        args.memory->InitializeBf16StateInS1(head);
        if (args.tiling->useInitialState) {
            Mte2StateToBf16Ub(
                /*源=*/InitialStateAt(*args.in, plan.sequence, head.hv),
                /*目的=*/UbBf16State(*args.memory, head), args.tiling->stateLayout);
            args.sync->Set(EventKind::StateReady, FixedMemory::LocalBank(head), /*S1 MTE2*/ 0,
                           /*S1 VF*/ 1);
        } else {
            // 同一次 no-P VF 内置零 R0；不读取未初始化的 GM rolling state。
            VfZeroBf16State(UbBf16State(*args.memory, head));
            args.sync->Set(EventKind::StateReady, FixedMemory::LocalBank(head), /*S1 VF*/ 1,
                           /*S3 VF*/ 3);
        }
        return;
    }

    // FP32 state 不在 S1 占用 shared state scratch；无 initial 的 H0 使用独立 zero-state 输出。
    if (!args.tiling->useInitialState && plan.chunk.first) {
        args.memory->AcquireLocalDataForH0(head);
        const auto h0 = VfZeroH0Bf16(head);
        Mte3WriteH0LayoutAwareAsync(*args.out, *args.tiling, plan, head,
                                    UbH0WriteTarget(*args.memory, head), h0);
        wait_mte3_h0_done(FixedMemory::LocalBank(head));
        args.memory->ReleaseH0AfterMte3(head);
        if (plan.stage2Required) {
            // 只有本 chunk 确实进入 S2 时，H0 才需要向 D producer 交接 local data。
            args.sync->Set(EventKind::LocalDataFree, head.roundHead, /*S1 H0 MTE3*/ 1,
                           /*本 chunk S2 Fixpipe*/ 2);
        }
        if (plan.nextRoundStartsWithS0) {
            // 只有下一 round 真正存在 AIC S0 写者时，才发布跨核 unionFree；下一 round 无
            // initial 时由同一 AIV 的 no-P VF 按 V pipe 程序顺序接管，不生成无消费者 token。
            args.sync->Set(EventKind::UnionFree, head.localSlot, /*S1 VF*/ 1,
                           /*下一 round S0*/ 0);
        } else {
            PipeBarrierVForH0Union(head);
        }
    }
}

inline void Stage1LoadUAndGate(const VecStageArgs& args, const HeadBinding& head)
{
    const int bank = FixedMemory::LocalBank(head);
    auto& ticket = args.memory->vNewWork[bank];
    if (ticket.generation > 0 && !args.plan->roundBoundaryDrained) {
        // 上一 chunk 的 V_new GM/L1 MTE3 仍可能读取该 bank；必须按代际等待 free。
        args.sync->Wait(EventKind::VNewWorkFree, bank, /*S1 MTE2*/ 0);
    }
    args.memory->AcquireVNewWorkForS1(head);

    Mte2InitConstValueAsync(UbVNewWork(*args.memory, head, args.tiling->stateType), /*zero*/ 0,
                            /*bytes*/ 16 * 1024);
    Mte2CopyValidRowsAsync(
        /*源=*/UAt(*args.in, args.plan->chunk, head.hv),
        /*目的=*/UbVNewWork(*args.memory, head, args.tiling->stateType),
        args.plan->chunk.validTokens);
    if (args.plan->gateMode == GateMode::ScalarG && args.variant != Stage1Variant::v_new_only) {
        Mte2CopyGateValidRowsAsync(
            /*源=*/GAt(*args.in, args.plan->chunk, head.hv),
            /*目的=*/UbGate(*args.memory, head), args.plan->chunk.validTokens);
    }
    WaitMte2ToVForStage1(head);
    args.memory->MarkVNewWorkReady(head);
}

inline void Stage1ComputeAndWriteVNew(const VecStageArgs& args, const HeadBinding& head)
{
    const RoundPlan& plan = *args.plan;
    const bool hasP = plan.stage0Required;
    if (hasP) {
        args.sync->Wait(EventKind::PReady, head.roundHead, /*S1 VF*/ 1);
    }

    // 一个 VF 覆盖本 head 的 [M,V]；P 不存在时用编译期 zero-P 语义，不能伪造 W @ 0。
    const auto p = hasP ? UbP(*args.memory, head) : ZeroP(head);
    auto vNewFp32 = VfSubFp32Full(
        /*u=*/UbVNewWork(*args.memory, head, args.tiling->stateType),
        /*p=*/p, args.plan->chunk.validTokens);
    const auto vNew = CastBf16(vNewFp32);
    write_v_new_from_ub(*args.out, plan.chunk, head.hv,
                        UbVNewWork(*args.memory, head, args.tiling->stateType), vNew);

    // 子模块 Stage1WriteRightOperand：仅在确有 S2 消费者时，把右操作数写入 L1 zN。
    if (plan.gateMode == GateMode::ScalarG && plan.stage2Required) {
        // g-only：同一次 VF 从 V_new_fp32 派生 V_new_g 和 alpha，避免重复读 u/P/g。
        const auto gate = UbGate(*args.memory, head);
        const auto vNewG = CastBf16(
            VfMulFull(vNewFp32, GateDeltaFromUb(gate, plan.chunk.validTokens,
                                                args.tiling->useExp2)));
        args.memory->AcquireRightForS1(head.hSlot);
        Mte3WriteL1RightZnAsync(L1Right(args.memory->l1, head.hSlot), vNewG,
                                plan.chunk.validTokens);
        StoreAlpha(UbAlpha(*args.memory, head),
                   LastGateFromUb(gate, plan.chunk.validTokens, args.tiling->useExp2));
        wait_mte3_right_done(head.hSlot);
        args.memory->MarkRightReady(head.hSlot);
        args.sync->Set(EventKind::RightReady, head.hSlot, /*S1 MTE3*/ 1,
                       /*S2 MTE1*/ 2);
    } else if (plan.gateMode == GateMode::KeyWiseGk && plan.stage2Required) {
        // gk-only：相对衰减已经吸收到输入 kg，S1 不搬 gk、不生成 V_new_g。
        args.memory->AcquireRightForS1(head.hSlot);
        Mte3WriteL1RightZnAsync(L1Right(args.memory->l1, head.hSlot), vNew,
                                plan.chunk.validTokens);
        wait_mte3_right_done(head.hSlot);
        args.memory->MarkRightReady(head.hSlot);
        args.sync->Set(EventKind::RightReady, head.hSlot, /*S1 MTE3*/ 1,
                       /*S2 MTE1*/ 2);
    }

    // P 的最后一次读取在 VF 内完成；g-only 还要等 V_new_g MTE3，RightReady 才能继续 S2。
    if (hasP) {
        VfLastReadP(head);
        args.memory->ReleasePAfterS1(head);
        args.sync->Release(EventKind::PFree, head.roundHead, /*S1 VF*/ 1);
    }
}

inline void Stage1WriteH0IfNeeded(const VecStageArgs& args, const HeadBinding& head)
{
    const RoundPlan& plan = *args.plan;
    if (!plan.chunk.first || plan.finalVNewOnly || args.tiling->stateType == StateType::Fp32) {
        // FP32 initial 的 h0 已由 S-1 写回；无 initial 的 FP32 h0 在 zero-state 分支单独写回。
        return;
    }

    args.sync->Wait(EventKind::StateReady, FixedMemory::LocalBank(head), /*S1 VF*/ 1);
    Mte3WriteH0LayoutAwareAsync(*args.out, *args.tiling, plan, head,
                                UbBf16State(*args.memory, head));
    // 首个 S3 将原位覆写同一 state bank 前，必须确认 H0 MTE3 已不再读取该源。
    wait_mte3_h0_done(FixedMemory::LocalBank(head));
}

inline void Stage1ReleaseVNewWork(const VecStageArgs& args, const HeadBinding& head)
{
    const int bank = FixedMemory::LocalBank(head);
    // v_new GM 是所有分支都存在的最后一条 MTE3；gk-only 还要计入 L1 右操作数 MTE3。
    wait_mte3_vnew_done(args.plan->chunk, head.hv);
    args.memory->ReleaseVNewWorkAfterMte3(head);
    if (args.plan->hasNextChunk || args.plan->hasNextHeadRound) {
        args.sync->Set(EventKind::VNewWorkFree, bank, /*S1 MTE3*/ 1,
                       /*下一 S1/S-1 调度者*/ -1);
    }
}

inline VecStageResult RunStage1Arch22(const VecStageArgs& args)
{
    VecStageResult result{};
    const RoundPlan& plan = *args.plan;
    for (int i = 0; i < plan.activeHvCount; ++i) {
        const HeadBinding& head = plan.heads[i];
        Stage1PrepareInitialOrZeroState(args, head);
        Stage1LoadUAndGate(args, head);
        Stage1ComputeAndWriteVNew(args, head);
        Stage1WriteH0IfNeeded(args, head);
        Stage1ReleaseVNewWork(args, head);
        ++result.activeTaskCount;
        result.rightOperandReady = result.rightOperandReady || plan.stage2Required;
        result.alphaReady = result.alphaReady ||
                            (plan.gateMode == GateMode::ScalarG && plan.stage2Required);
    }
    return result;
}

// ------------------------------- Stage 3 helpers -------------------------------

inline void Stage3PrepareState(const VecStage3Args& args, const HeadBinding& head)
{
    const RoundPlan& plan = *args.plan;
    if (args.tiling->stateType == StateType::Bf16) {
        auto& state = args.memory->bf16State[FixedMemory::LocalBank(head)];
        if (state.owner == StateOwner::RNextMte3) {
            args.sync->Wait(EventKind::StateToVFree, FixedMemory::LocalBank(head),
                            /*下一个 S3 VF*/ 3);
            args.memory->MarkBf16StateConsumedByNextVf(head);
        }
        // 首 chunk 的 R0 已由 S1 初始化；后续 chunk 直接使用 canonical UB resident。
        (void)plan;
        return;
    }

    // FP32 state 共享 [160,224) scratch；先按上一 owner 类型等待，再一次完整 MTE2 搬入。
    if (args.memory->fp32StateScratch.owner == StateOwner::RNextMte3) {
        args.sync->Wait(EventKind::StateToMte2Free, 0, /*下一 head/chunk 的 state MTE2*/ 0);
        args.memory->ReleaseFp32StateScratch();
    }
    args.memory->AcquireFp32StateScratchForS3();
    if (plan.chunk.first && !args.tiling->useInitialState) {
        VfZeroFp32State(UbFp32StateScratch(*args.memory));
    } else {
        Mte2ReadRollingStateLayoutAware(
            /*源=*/RollingStateSource(*args.in, *args.out, *args.tiling, plan, head),
            /*目的=*/UbFp32StateScratch(*args.memory), args.tiling->stateLayout);
    }
    args.memory->MarkFp32StateReady();
    args.sync->Set(EventKind::StateReady, 0, /*state MTE2*/ 0, /*S3 VF*/ 3);
}

inline void Stage3LoadGate(const VecStage3Args& args, const HeadBinding& head)
{
    const RoundPlan& plan = *args.plan;
    if (plan.gateMode != GateMode::KeyWiseGk) {
        // g-only 直接复用 S1 在 alpha UB slot 中保留的最后 gate。
        return;
    }
    // 只搬最后一个有效 token 的 gk[M-1,:]；尾 chunk 严禁读取 BT-1 padding。
    Mte2CopyGkLastAsync(
        /*源=*/GkAtLast(*args.in, plan.chunk, head.hv),
        /*目的=*/UbGkLast(*args.memory, head));
    WaitMte2ToVForStage3(head);
}

inline void Stage3ComputeRNext(const VecStage3Args& args, const HeadBinding& head)
{
    const RoundPlan& plan = *args.plan;
    args.sync->Wait(EventKind::DReady, head.roundHead, /*S3 VF*/ 3);
    if (args.tiling->stateType == StateType::Fp32 || plan.chunk.first) {
        args.sync->Wait(EventKind::StateReady,
                        args.tiling->stateType == StateType::Bf16 ? FixedMemory::LocalBank(head) : 0,
                        /*S3 VF*/ 3);
    }

    const auto state = args.tiling->stateType == StateType::Bf16
                           ? UbBf16State(*args.memory, head)
                           : UbFp32StateScratch(*args.memory);
    const auto d = UbD(*args.memory, head);
    const auto gatedState = plan.gateMode == GateMode::ScalarG
                                ? VfScaleStateByAlpha(state, UbAlpha(*args.memory, head),
                                                      plan.chunk.first && !args.tiling->useInitialState)
                                : VfScaleStateByGkLast(state, UbGkLast(*args.memory, head),
                                                       args.tiling->useExp2,
                                                       plan.chunk.first && !args.tiling->useInitialState);

    // 一次完整 VF：Rnext = gated(R_c) + D_c。每行 D 先进入寄存器，末次读取后才允许
    // 该行的固定地址转交 H 写者；不做 UB copy、compact 或第二次 VF。
    const auto rNextFp32 = VfAddStateAndD(gatedState, d);
    const auto rNext = CastState(rNextFp32, args.tiling->stateType);
    const auto hNext = CastBf16(rNext);

    if (args.tiling->stateType == StateType::Fp32) {
        // FP32 分支的 H 目标是本 head local data bank 的固定低 32 KiB；VF 内逐行完成
        // D 的末次读取后再写入 BF16 H，不覆盖尚未读取的 D 行。
        VfReadDAndWriteBf16HInFixedSubregion(d, hNext, UbHWriteTarget(*args.memory, head));
        args.memory->BeginHWriteAfterDRow(head);
    }

    // 子模块 Stage3WriteStateOutputs：只写有真实消费者的 Hnext 或 final_state。
    if (!plan.chunk.last) {
        // H_{c+1} 同时写公开 h 和 L1 resident；两路 MTE3 完成前不能发布 HReady。
        args.memory->ProduceHForS3(head.hSlot);
        Mte3WriteHLayoutAwareAsync(*args.out, *args.tiling, plan, head, hNext);
        Mte3WriteL1ResidentAsync(L1H(args.memory->l1, head.hSlot), hNext);
        wait_mte3_hnext_done(head.hSlot);
        args.memory->MarkHReady(head.hSlot);
        args.sync->Set(EventKind::HReady, head.hSlot, /*S3 MTE3*/ 1,
                       /*下一 chunk S0 MTE1*/ 1);
    }

    if (plan.chunk.last && args.tiling->outputFinalState) {
        // final_state 的物理末两轴由 state_v_first 决定，内部 rNext 始终是 canonical [K,V]。
        Mte3WriteFinalStateLayoutAwareAsync(*args.out, *args.tiling, plan, head, rNext);
        wait_mte3_final_state_done(head.hv);
    }
}

inline void Stage3ReleaseStateAndD(const VecStage3Args& args, const HeadBinding& head)
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
                           /*state MTE3*/ 1, /*下一 round S1 MTE2*/ 0);
        } else if (plan.nextRoundStartsWithS1NoP) {
            args.memory->MarkBf16StateMte3InFlight(head);
            args.sync->Set(EventKind::StateToVFree, FixedMemory::LocalBank(head),
                           /*state MTE3*/ 1, /*下一 round S1 VF*/ 1);
        } else {
            args.memory->ReleaseBf16StateAtTerminal(head);
        }
    } else {
        const bool nextHeadUsesScratch = head.roundHead + 1 < plan.activeHvCount;
        if (plan.hasNextChunk || plan.hasNextHeadRound || nextHeadUsesScratch) {
            args.memory->MarkFp32StateMte3InFlight();
            args.sync->Set(EventKind::StateToMte2Free, 0, /*state MTE3*/ 1,
                           /*下一 state MTE2/调度器*/ -1);
        } else if (plan.chunk.last && args.tiling->outputFinalState) {
            // final_state MTE3 是本 sequence 的最后一个状态消费者；只交给 terminal drain，
            // 不生成没有消费者的 free token。
            args.memory->MarkFp32StateMte3InFlight();
        } else {
            args.memory->ReleaseFp32StateScratch();
        }
    }

    // D 的最后一次读取与（FP32 state 时）H 派生写出均在同一次 VF 内完成。
    args.memory->ReleaseDAfterS3(head);
    args.sync->Release(EventKind::DFree, head.roundHead, /*S3 VF*/ 3);
    if (plan.gateMode == GateMode::KeyWiseGk) {
        ReleaseUbGkLast(*args.memory, head);
    }
}

inline VecStageResult RunStage3Arch22(const VecStage3Args& args)
{
    VecStageResult result{};
    const RoundPlan& plan = *args.plan;
    if (!plan.stage3Required) {
        // final chunk 且不输出 final_state 时没有 D/state 消费者，S3 整体跳过。
        return result;
    }

    for (int i = 0; i < plan.activeHvCount; ++i) {
        const HeadBinding& head = plan.heads[i];
        Stage3PrepareState(args, head);
        Stage3LoadGate(args, head);
        Stage3ComputeRNext(args, head);
        Stage3ReleaseStateAndD(args, head);
        ++result.activeTaskCount;
        result.nextHReady = result.nextHReady || !plan.chunk.last;
        result.finalStateWritten = result.finalStateWritten ||
                                   (plan.chunk.last && args.tiling->outputFinalState);
    }
    return result;
}

} // 命名空间 fwd_h_pseudocode
