// 仅伪代码。arch22（A2/A3）Vec 阶段：S-1、Stage1 和 Stage3。
// 该文件保留真实 op_kernel 的文件名和架构目录，具体 Ascend C 调用需在实现时补齐。

#ifndef FLA_FWD_H_PSEUDOCODE_ARCH22_CHUNK_GATED_DELTA_RULE_FWD_H_VEC_H_
#define FLA_FWD_H_PSEUDOCODE_ARCH22_CHUNK_GATED_DELTA_RULE_FWD_H_VEC_H_

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
    int aivId = 0;
};

struct VecStage3Args {
    const ApiInputs* in = nullptr;
    const ApiOutputs* out = nullptr;
    const TilingPlan* tiling = nullptr;
    const RoundPlan* plan = nullptr;
    FixedMemory* memory = nullptr;
    SyncLedger* sync = nullptr;
    int aivId = 0;
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
    int aivId = 0;
};

struct VecStageResult {
    bool rightGmReady = false;
    bool alphaReady = false;
    bool nextHReady = false;
    bool finalStateWritten = false;
    int activeTaskCount = 0;
};

// ------------------------------- S-1 helpers -------------------------------

inline void SMinusOneLoadInitial(const SMinusOneArgs& args, const HeadBinding& head)
{
    // S-1 输入公式：R_0,h = layout_decode(initial_state[n,h,:,:])，内部统一为 FP32 [K,V]。
    const int eventSlot = head.localSlot;
    args.memory->AcquireInitialInput(head);

    // FP32 initial 只完整搬入一次；layout-aware 描述符按 state_v_first 选择 GM 访问顺序。
    Mte2StateToFp32UbAsync(
        /*源=*/InitialStateAt(*args.in, args.sequence->sequence, head.hv),
        /*目的=*/UbInitialInput(*args.memory, head), args.tiling->stateLayout);
    args.memory->MarkInitialInputReady(head);
    args.sync->Set(EventKind::InitialInputReady, eventSlot, /*S-1 MTE2*/ 0,
                   /*S-1 VF*/ 1);
}

inline void SMinusOneConvertAndWriteH0(const SMinusOneArgs& args, const HeadBinding& head)
{
    // S-1 计算公式：H_0,h = cast_BF16(R_0,h)，保持 canonical [K,V] 后写 GM。
    const int eventSlot = head.localSlot;
    args.sync->Wait(EventKind::InitialInputReady, eventSlot, /*S-1 VF*/ 1);
    args.memory->AcquireInitialHOutput(head);
    args.memory->ProduceInitialH(head.hSlot);

    // 一次完整 VF：FP32 initial -> BF16 H0，内部 canonical [K,V]，不拆行或增加 pass。
    const auto h0 = VfCastFp32ToBf16Full(UbInitialInput(*args.memory, head));
    VfLastReadInitialInput(head);
    args.memory->ReleaseInitialInput(head);

    // H0 只写公开 GM 的 layout-aware 形式；S0 随后用 MTE2 搬成 L1 NZ，避免 UB -> L1 直通。
    const int pipelineSlot = AivPipelineSlot(head.localSlot);
    args.sync->Set(EventKind::SMinusVToMte3Ready, pipelineSlot,
                   /*S-1 VF*/ 1, /*S-1 MTE3*/ 3);
    args.sync->Wait(EventKind::SMinusVToMte3Ready, pipelineSlot, /*S-1 MTE3*/ 3);
    Mte3WriteH0LayoutAwareAsync(*args.out, *args.tiling, *args.sequence, head.hv, h0);
    args.memory->MarkHReady(head.hSlot);
    args.memory->MarkInitialHOutputReady(head);
}

inline void RunSMinusOneArch22(const SMinusOneArgs& args)
{
    // S-1 阶段公式：对当前 round 的每个 head 执行 H_0,h = cast_BF16(initial_state_h)。
    if (!args.tiling->useInitialState || args.tiling->stateType != StateType::Fp32) {
        // BF16 initial 由首个 S0 直接读取；无 initial 不建立 S-1 task。
        return;
    }

    const int coreHeadCount =
        args.activeHvCount > args.aivId
            ? (args.activeHvCount - args.aivId + kAivCount - 1) / kAivCount
            : 0;
    for (int coreHeadId = 0; coreHeadId < coreHeadCount; ++coreHeadId) {
        const int roundHead = args.aivId + coreHeadId * kAivCount;
        HeadBinding head{};
        head.roundHead = roundHead;
        head.hv = args.activeHvBegin + roundHead;
        head.hSlot = roundHead;
        head.aiv = args.aivId;
        head.localSlot = coreHeadId;
        SMinusOneLoadInitial(args, head);
        // 先下发当前 head 的 MTE2，再让 VF 等前一个 head 的独立 ready EventID；
        // 当前 head 的 MTE2 不会被前一个 head 的 VF WaitFlag 阻塞。
        if (coreHeadId > 0) {
            const int previousRoundHead = args.aivId + (coreHeadId - 1) * kAivCount;
            HeadBinding previous{};
            previous.roundHead = previousRoundHead;
            previous.hv = args.activeHvBegin + previousRoundHead;
            previous.hSlot = previousRoundHead;
            previous.aiv = args.aivId;
            previous.localSlot = coreHeadId - 1;
            SMinusOneConvertAndWriteH0(args, previous);
        }
    }
    if (coreHeadCount > 0) {
        const int coreHeadId = coreHeadCount - 1;
        const int roundHead = args.aivId + coreHeadId * kAivCount;
        HeadBinding last{};
        last.roundHead = roundHead;
        last.hv = args.activeHvBegin + roundHead;
        last.hSlot = roundHead;
        last.aiv = args.aivId;
        last.localSlot = coreHeadId;
        SMinusOneConvertAndWriteH0(args, last);
    }

    // 每个 head 的 HGmReady 都排在本 AIV 对应 H0 MTE3 之后；不生成无消费者的统一 phase token。
    for (int coreHeadId = 0; coreHeadId < coreHeadCount; ++coreHeadId) {
        const int roundHead = args.aivId + coreHeadId * kAivCount;
        args.sync->Set(EventKind::HGmReady, roundHead, /*S-1 phase*/ 1, /*S0 MTE2*/ 0);
        args.memory->ReleaseInitialHOutput(
            HeadBinding{roundHead, args.activeHvBegin + roundHead, -1, -1, roundHead,
                        roundHead, args.aivId, coreHeadId, true});
    }
}

// ------------------------------- Stage 1 helpers -------------------------------

inline void Stage1PrepareInitialOrZeroState(const VecStageArgs& args, const HeadBinding& head)
{
    // Stage1 初态公式：R_0,h = initial_state_h（有初态）或 0（无初态）；v_new-only 仅生成 H0 scratch。
    const RoundPlan& plan = *args.plan;
    if (!plan.chunk.first) {
        // 后续 chunk 的 BF16 R_c 已经由前一个 S3 驻留在同一个 UB state bank。
        return;
    }

    if (args.tiling->stateType == StateType::Bf16) {
        if (plan.finalVNewOnly) {
            // 最终 v_new-only 没有 S3 消费者：首 chunk 的 H0 只用临时源写出，不建立
            // 跨 chunk 的 R resident，也不发布面向 S3 的 state free 事件。
            if (args.tiling->useInitialState) {
                Mte2StateToBf16Ub(
                    /*源=*/InitialStateAt(*args.in, plan.sequence, head.hv),
                    /*目的=*/UbH0Scratch(*args.memory, head), args.tiling->stateLayout);
                args.sync->Set(EventKind::StateInitToH0Ready, head.localSlot,
                               /*S1 MTE2*/ 0, /*S1 MTE3*/ 3);
            } else {
                VfZeroBf16State(UbH0Scratch(*args.memory, head));
                args.sync->Set(EventKind::StateInitToH0Ready, head.localSlot,
                               /*S1 VF*/ 1, /*S1 MTE3*/ 3);
            }
            return;
        }
        args.memory->InitializeBf16StateInS1(head);
        if (args.tiling->useInitialState) {
            Mte2StateToBf16Ub(
                /*源=*/InitialStateAt(*args.in, plan.sequence, head.hv),
                /*目的=*/UbBf16State(*args.memory, head), args.tiling->stateLayout);
            args.sync->Set(EventKind::StateInitToH0Ready, head.localSlot, /*S1 MTE2*/ 0,
                           /*S1 H0 MTE3*/ 3);
        } else {
            // 同一次 no-P VF 内置零 R0；不读取未初始化的 GM rolling state。
            VfZeroBf16State(UbBf16State(*args.memory, head));
            args.sync->Set(EventKind::StateInitToH0Ready, head.localSlot, /*S1 VF*/ 1,
                           /*S1 H0 MTE3*/ 3);
        }
        return;
    }

    // FP32 state 不在 S1 占用 shared state scratch；无 initial 的 H0 使用独立 zero-state 输出。
    if (!args.tiling->useInitialState && plan.chunk.first) {
        args.memory->AcquireLocalDataForH0(head);
        const auto h0 = VfZeroH0Bf16(head);
        Mte3WriteH0LayoutAwareAsync(*args.out, *args.tiling, plan, head,
                                    UbH0WriteTarget(*args.memory, head), h0);
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

inline void Stage1LoadUAndGate(const VecStageArgs& args, const HeadBinding& head,
                               int pipelineSlot)
{
    // Stage1 输入公式：u_c,h = u[h,0:M,:]；g_c,h = g[h,0:M]（仅 g-only）。
    const int bank = FixedMemory::LocalBank(head);
    auto& ticket = args.memory->vNewWork[bank];
    if (ticket.generation > 0 && !args.plan->roundBoundaryDrained) {
        args.sync->Wait(EventKind::S1Mte3ToMte2Free, pipelineSlot, /*S1 MTE2*/ 0);
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
    // 当前 slot 的最后一条 MTE2 后立即 SetFlag<MTE2_V>。必须在下一个
    // coreHeadId 的 MTE2 之前下发，保证 ping ready 不包含 pong 的搬运。
    args.sync->Set(EventKind::S1Mte2ToVReady, pipelineSlot,
                   /*S1 MTE2*/ 0, /*S1 VF*/ 1);
    args.memory->MarkVNewWorkReady(head);
}

inline void Stage1ComputeAndWriteVNew(const VecStageArgs& args, const HeadBinding& head,
                                      int pipelineSlot)
{
    // Stage1 计算公式：V_new_fp32,c,h = fp32(u_c,h) - fp32(P_c,h)，V_new_c,h = cast_BF16(V_new_fp32,c,h)；无 P 分支取 P_c,h=0。
    // g-only 额外计算 V_new_g,c,h[i,:] = cast_BF16(E(g_last-g_i) * V_new_fp32,c,h[i,:])；E 由 useExp2 决定。
    const RoundPlan& plan = *args.plan;
    args.sync->Wait(EventKind::S1Mte2ToVReady, pipelineSlot, /*S1 VF*/ 1);
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
    // VF 的最后一次 UB 写之后发布本 slot 的 V_MTE3 事件；MTE3 只等待该 slot。
    args.sync->Set(EventKind::S1VToMte3Ready, pipelineSlot,
                   /*S1 VF*/ 1, /*S1 MTE3*/ 3);
    args.sync->Wait(EventKind::S1VToMte3Ready, pipelineSlot, /*S1 MTE3*/ 3);
    write_v_new_from_ub(*args.out, plan.chunk, head.hv,
                        UbVNewWork(*args.memory, head, args.tiling->stateType), vNew);

    // 子模块 Stage1WriteRightOperandToGm：UB 中是 ND，先写入每个 roundHead
    // 对应的 GM scratch；这里禁止 UB -> L1，因为该路径不能一次完成 ND -> NZ。
    if (plan.gateMode == GateMode::ScalarG && plan.stage2Required) {
        // g-only：同一次 VF 从 V_new_fp32 派生 V_new_g 和 alpha，避免重复读 u/P/g。
        const auto gate = UbGate(*args.memory, head);
        const auto vNewG = CastBf16(
            VfMulFull(vNewFp32, GateDeltaFromUb(gate, plan.chunk.validTokens,
                                                args.tiling->useExp2)));
        const int rightSlot = head.roundHead;
        if (args.memory->rightGm[rightSlot].generation > 0 && !plan.roundBoundaryDrained) {
            args.sync->Wait(EventKind::RightGmFree, rightSlot, /*下一 chunk S1 MTE3*/ 1);
        }
        args.memory->AcquireRightGmForS1(rightSlot);
        Mte3WriteRightOperandGmNdAsync(
            args.workspace->rightOperandGm, rightSlot, plan.chunk,
            vNewG, plan.chunk.validTokens);
        StoreAlpha(UbAlpha(*args.memory, head),
                   LastGateFromUb(gate, plan.chunk.validTokens, args.tiling->useExp2));
        args.memory->MarkRightGmReady(rightSlot);
        args.sync->Set(EventKind::RightGmReady, rightSlot, /*S1 MTE3*/ 1,
                       /*S2 MTE2*/ 0);
    } else if (plan.gateMode == GateMode::KeyWiseGk && plan.stage2Required) {
        // gk-only：相对衰减已经吸收到输入 kg，S1 不搬 gk、不生成 V_new_g。
        const int rightSlot = head.roundHead;
        if (args.memory->rightGm[rightSlot].generation > 0 && !plan.roundBoundaryDrained) {
            args.sync->Wait(EventKind::RightGmFree, rightSlot, /*下一 chunk S1 MTE3*/ 1);
        }
        args.memory->AcquireRightGmForS1(rightSlot);
        Mte3WriteRightOperandGmNdAsync(
            args.workspace->rightOperandGm, rightSlot, plan.chunk,
            vNew, plan.chunk.validTokens);
        args.memory->MarkRightGmReady(rightSlot);
        args.sync->Set(EventKind::RightGmReady, rightSlot, /*S1 MTE3*/ 1,
                       /*S2 MTE2*/ 0);
    }

    // P 的最后一次读取在 VF 内完成；右操作数先落 GM，S2 再负责 GM -> L1 NZ。
    if (hasP) {
        VfLastReadP(head);
        args.memory->ReleasePAfterS1(head);
        args.sync->Release(EventKind::PFree, head.roundHead, /*S1 VF*/ 1);
    }

}

inline void Stage1WriteH0IfNeeded(const VecStageArgs& args, const HeadBinding& head)
{
    // Stage1 H0 输出公式：首 chunk 且需要 H0 时写 H_0,h = R_0,h（无初态则为零矩阵）。
    const RoundPlan& plan = *args.plan;
    if (!plan.chunk.first || args.tiling->stateType == StateType::Fp32) {
        // FP32 initial 的 h0 已由 S-1 写回；无 initial 的 FP32 h0 在 zero-state 分支单独写回。
        return;
    }

    args.sync->Wait(EventKind::StateInitToH0Ready, head.localSlot, /*S1 H0 MTE3*/ 3);
    Mte3WriteH0LayoutAwareAsync(*args.out, *args.tiling, plan, head,
                                plan.finalVNewOnly
                                    ? UbH0Scratch(*args.memory, head)
                                    : UbBf16State(*args.memory, head));
    if (plan.stage3Required) {
        args.memory->MarkBf16StateMte3InFlight(head);
        args.sync->Set(EventKind::StateToVFree, head.localSlot,
                       /*S1 H0 MTE3*/ 3, /*本 chunk S3 VF*/ 3);
    }
}

inline void Stage1ReleaseVNewWork(const VecStageArgs& args, const HeadBinding& head,
                                  int pipelineSlot, bool hasNextSameSlot)
{
    // Stage1 释放公式：V_new_c,h 已写公开 GM，右操作数已写 GM scratch；仅回收对应 work bank。
    // v_new GM 是所有分支都存在的 MTE3；这里只把反向事件排到本 slot
    // 最后一条 MTE3 后面，不做同步 wait，因而可以和下一 head 的 VF 重叠。
    args.memory->ReleaseVNewWorkAfterMte3(head);
    if (hasNextSameSlot || args.plan->hasNextChunk) {
        // 放在本 slot 最后一条 MTE3 之后，仅在存在下一代同 slot MTE2 时发布。
        args.sync->Set(EventKind::S1Mte3ToMte2Free, pipelineSlot,
                       /*S1 MTE3*/ 3, /*下一 S1 MTE2*/ 0);
    }
}

inline void Stage1ConsumeHead(const VecStageArgs& args, const HeadBinding& head,
                              int pipelineSlot, bool hasNextSameSlot)
{
    Stage1ComputeAndWriteVNew(args, head, pipelineSlot);
    Stage1WriteH0IfNeeded(args, head);
    Stage1ReleaseVNewWork(args, head, pipelineSlot, hasNextSameSlot);
}

inline VecStageResult RunStage1Arch22(const VecStageArgs& args)
{
    // Stage1 阶段公式：逐 head 完成 V_new_c,h = cast_BF16(fp32(u_c,h)-fp32(P_c,h)) 及可选右操作数派生；无 P 时取零。
    VecStageResult result{};
    const RoundPlan& plan = *args.plan;
    const int coreHeadCount = AivCoreHeadCount(plan, args.aivId);
    for (int coreHeadId = 0; coreHeadId < coreHeadCount; ++coreHeadId) {
        const CoreHeadBinding current = BindAivCoreHead(plan, args.aivId, coreHeadId);
        const HeadBinding& head = plan.heads[current.roundHead];
        Stage1PrepareInitialOrZeroState(args, head);
        Stage1LoadUAndGate(args, head, current.pipelineSlot);
        if (coreHeadId > 0) {
            const CoreHeadBinding previous =
                BindAivCoreHead(plan, args.aivId, coreHeadId - 1);
            Stage1ConsumeHead(args, plan.heads[previous.roundHead], previous.pipelineSlot,
                              /*hasNextSameSlot*/
                              previous.coreHeadId + kAivPipelineSlotCount < coreHeadCount);
        }
        ++result.activeTaskCount;
        result.rightGmReady = result.rightGmReady || plan.stage2Required;
        result.alphaReady = result.alphaReady ||
                            (plan.gateMode == GateMode::ScalarG && plan.stage2Required);
    }
    if (coreHeadCount > 0) {
        const CoreHeadBinding last =
            BindAivCoreHead(plan, args.aivId, coreHeadCount - 1);
        Stage1ConsumeHead(args, plan.heads[last.roundHead], last.pipelineSlot,
                          /*hasNextSameSlot*/ false);
    }
    return result;
}

// ------------------------------- Stage 3 helpers -------------------------------

inline void Stage3PrepareState(const VecStage3Args& args, const HeadBinding& head)
{
    // Stage3 初态公式：R_c,h = rolling state（BF16 resident 或 FP32 GM scratch），首 chunk 无初态时为零。
    const RoundPlan& plan = *args.plan;
    if (args.tiling->stateType == StateType::Bf16) {
        auto& state = args.memory->bf16State[FixedMemory::LocalBank(head)];
        if (state.owner == StateOwner::RNextMte3) {
            args.sync->Wait(EventKind::StateToVFree, head.localSlot,
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
        args.sync->Set(EventKind::StateMte2ToStage3VReady, 0,
                       /*state MTE2*/ 0, /*S3 VF*/ 3);
    }
    args.memory->MarkFp32StateReady();
}

inline void Stage3LoadGate(const VecStage3Args& args, const HeadBinding& head,
                           int pipelineSlot)
{
    // Stage3 门控公式：g-only 使用 alpha_c = E(g_last)，gk-only 使用 gk_last,h = gk_c,h[M-1,:]；E 由 useExp2 决定。
    const RoundPlan& plan = *args.plan;
    if (plan.gateMode != GateMode::KeyWiseGk) {
        // g-only 直接复用 S1 在 alpha UB slot 中保留的最后 gate。
        return;
    }
    // 只搬最后一个有效 token 的 gk[M-1,:]；尾 chunk 严禁读取 BT-1 padding。
    Mte2CopyGkLastAsync(
        /*源=*/GkAtLast(*args.in, plan.chunk, head.hv),
        /*目的=*/UbGkLast(*args.memory, head));
    args.sync->Set(EventKind::S3Mte2ToVReady, pipelineSlot,
                   /*S3 MTE2*/ 0, /*S3 VF*/ 3);
}

inline void Stage3ComputeRNext(const VecStage3Args& args, const HeadBinding& head,
                               int pipelineSlot, bool publishGateFree)
{
    // Stage3 计算公式：R_{c+1,h} = gate(R_c,h) + D_c,h；H_{c+1,h} = cast_BF16(R_{c+1,h})。
    const RoundPlan& plan = *args.plan;
    if (plan.gateMode == GateMode::KeyWiseGk) {
        // 只有 gk-only 真正搬入 gk_last；g-only 没有 S3 gate MTE2，也不能等待伪造事件。
        args.sync->Wait(EventKind::S3Mte2ToVReady, pipelineSlot, /*S3 VF*/ 3);
    }
    args.sync->Wait(EventKind::DReady, head.roundHead, /*S3 VF*/ 3);
    const bool fp32StateFromMte2 = args.tiling->stateType == StateType::Fp32 &&
                                   !(plan.chunk.first && !args.tiling->useInitialState);
    if (fp32StateFromMte2) {
        args.sync->Wait(EventKind::StateMte2ToStage3VReady, 0, /*S3 VF*/ 3);
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
    if (publishGateFree) {
        // gk_last 的末次读取已经完成；下一代同槽 gate MTE2 只等 V，不应扩大到 MTE3。
        args.sync->Set(EventKind::S3VToMte2Free, pipelineSlot,
                       /*S3 VF*/ 3, /*下一 S3 MTE2*/ 0);
    }

    if (args.tiling->stateType == StateType::Fp32) {
        // FP32 分支的 H 目标是本 head local data bank 的固定低 32 KiB；VF 内逐行完成
        // D 的末次读取后再写入 BF16 H，不覆盖尚未读取的 D 行。
        VfReadDAndWriteBf16HInFixedSubregion(d, hNext, UbHWriteTarget(*args.memory, head));
        args.memory->BeginHWriteAfterDRow(head);
    }

    // VF 完整写完 Rnext/Hnext 后仅发布本 slot 的 V_MTE3；MTE3 等待本 slot，
    // V pipe 可以继续处理另一 slot。
    args.sync->Set(EventKind::S3VToMte3Ready, pipelineSlot,
                   /*S3 VF*/ 3, /*S3 MTE3*/ 1);
    args.sync->Wait(EventKind::S3VToMte3Ready, pipelineSlot, /*S3 MTE3*/ 1);

    // 子模块 Stage3WriteStateOutputs：只写有真实消费者的 Hnext 或 final_state。
    if (!plan.chunk.last) {
        // H_{c+1} 只写公开 GM 的 layout-aware 形式；下一 chunk 的 S0 再用 MTE2 转成 L1 NZ。
        args.memory->ProduceHForS3(head.hSlot);
        Mte3WriteHLayoutAwareAsync(*args.out, *args.tiling, plan, head, hNext);
        args.memory->MarkHReady(head.hSlot);
        args.sync->Set(EventKind::HGmReady, head.hSlot, /*S3 MTE3*/ 1,
                       /*下一 chunk S0 MTE2*/ 0);
    }

    if (plan.chunk.last && args.tiling->outputFinalState) {
        // final_state 的物理末两轴由 state_v_first 决定，内部 rNext 始终是 canonical [K,V]。
        Mte3WriteFinalStateLayoutAwareAsync(*args.out, *args.tiling, plan, head, rNext);
    }
}

inline void Stage3ReleaseStateAndD(const VecStage3Args& args, const HeadBinding& head,
                                   bool hasNextCoreHead)
{
    // Stage3 释放公式：D_c,h 末次读取、R_{c+1,h}/final_state 写出后，按下一 owner 释放 state 与 D。
    const RoundPlan& plan = *args.plan;
    if (args.tiling->stateType == StateType::Bf16) {
        if (plan.nextChunkNeedsStage3) {
            args.memory->MarkBf16StateMte3InFlight(head);
            args.sync->Set(EventKind::StateToVFree, head.localSlot,
                           /*state MTE3*/ 1, /*下一 S3 VF*/ 3);
        } else if (plan.hasNextChunk) {
            // 下一 chunk 是 final v_new-only：H 写回仍由 MTE3 完成，但不存在 state 消费者。
            args.memory->MarkBf16StateMte3InFlight(head);
        } else if (plan.nextRoundStartsWithS0) {
            args.memory->MarkBf16StateMte3InFlight(head);
            args.sync->Set(EventKind::StateToMte2Free, head.localSlot,
                           /*state MTE3*/ 1, /*下一 round S1 MTE2*/ 0);
        } else if (plan.nextRoundStartsWithS1NoP) {
            args.memory->MarkBf16StateMte3InFlight(head);
            args.sync->Set(EventKind::StateToVFree, head.localSlot,
                           /*state MTE3*/ 1, /*下一 round S1 VF*/ 1);
        } else {
            args.memory->ReleaseBf16StateAtTerminal(head);
        }
    } else {
        if (plan.nextChunkNeedsStage3 || plan.hasNextHeadRound || hasNextCoreHead) {
            args.memory->MarkFp32StateMte3InFlight();
            args.sync->Set(EventKind::StateToMte2Free, 0, /*state MTE3*/ 1,
                           /*下一 state MTE2/调度器*/ -1);
        } else if (plan.hasNextChunk) {
            // 下一 chunk 跳过 S3；只由 chunk/round drain 等待当前 H MTE3，不生成 state token。
            args.memory->MarkFp32StateMte3InFlight();
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

inline void Stage3PrefetchHead(const VecStage3Args& args, const HeadBinding& head,
                               int pipelineSlot, bool prefetchFp32State)
{
    // BF16 state 按 head 独立驻留。FP32 state 只有一个 shared scratch：核内首个 head
    // 可以在 ping 阶段预取，后续 head 必须等前一 head 的 StateToMte2Free。
    if (args.tiling->stateType == StateType::Bf16 || prefetchFp32State) {
        Stage3PrepareState(args, head);
    }
    Stage3LoadGate(args, head, pipelineSlot);
}

inline void Stage3ConsumeHead(const VecStage3Args& args, const HeadBinding& head,
                              int pipelineSlot, bool hasNextCoreHead,
                              bool hasNextSameSlot, bool fp32StatePrefetched)
{
    if (args.tiling->stateType == StateType::Fp32 && !fp32StatePrefetched) {
        Stage3PrepareState(args, head);
    }
    const bool publishGateFree = args.plan->gateMode == GateMode::KeyWiseGk &&
                                 (hasNextSameSlot || args.plan->hasNextChunk);
    Stage3ComputeRNext(args, head, pipelineSlot, publishGateFree);
    Stage3ReleaseStateAndD(args, head, hasNextCoreHead);
}

inline VecStageResult RunStage3Arch22(const VecStage3Args& args)
{
    // Stage3 阶段公式：逐 head 完成 R_{c+1,h} = gate(R_c,h) + D_c,h，并按分支写 H/final_state。
    VecStageResult result{};
    const RoundPlan& plan = *args.plan;
    if (!plan.stage3Required) {
        // final chunk 且不输出 final_state 时没有 D/state 消费者，S3 整体跳过。
        return result;
    }

    const int coreHeadCount = AivCoreHeadCount(plan, args.aivId);
    for (int coreHeadId = 0; coreHeadId < coreHeadCount; ++coreHeadId) {
        const CoreHeadBinding current = BindAivCoreHead(plan, args.aivId, coreHeadId);
        const bool reusesGateSlot = coreHeadId >= kAivPipelineSlotCount ||
                                    (plan.chunk.chunk > 0 &&
                                     coreHeadId < kAivPipelineSlotCount);
        if (plan.gateMode == GateMode::KeyWiseGk && reusesGateSlot) {
            args.sync->Wait(EventKind::S3VToMte2Free, current.pipelineSlot,
                            /*当前 S3 MTE2*/ 0);
        }
        Stage3PrefetchHead(args, plan.heads[current.roundHead], current.pipelineSlot,
                           /*prefetchFp32State*/ coreHeadId == 0);
        if (coreHeadId > 0) {
            const CoreHeadBinding previous =
                BindAivCoreHead(plan, args.aivId, coreHeadId - 1);
            Stage3ConsumeHead(args, plan.heads[previous.roundHead], previous.pipelineSlot,
                              /*hasNextCoreHead*/ true,
                              /*hasNextSameSlot*/
                              previous.coreHeadId + kAivPipelineSlotCount < coreHeadCount,
                              /*fp32StatePrefetched*/ previous.coreHeadId == 0);
        }
        ++result.activeTaskCount;
        result.nextHReady = result.nextHReady || !plan.chunk.last;
        result.finalStateWritten = result.finalStateWritten ||
                                   (plan.chunk.last && args.tiling->outputFinalState);
    }
    if (coreHeadCount > 0) {
        const CoreHeadBinding last =
            BindAivCoreHead(plan, args.aivId, coreHeadCount - 1);
        Stage3ConsumeHead(args, plan.heads[last.roundHead], last.pipelineSlot,
                          /*hasNextCoreHead*/ false, /*hasNextSameSlot*/ false,
                          /*fp32StatePrefetched*/ last.coreHeadId == 0);
    }
    return result;
}

} // 命名空间 fwd_h_pseudocode

#endif // FLA_FWD_H_PSEUDOCODE_ARCH22_CHUNK_GATED_DELTA_RULE_FWD_H_VEC_H_
