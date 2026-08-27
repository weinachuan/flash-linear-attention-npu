// 仅伪代码。arch22（A2/A3）Cube 阶段：Stage0 和 Stage2。
// 该文件保留真实 op_kernel 的文件名和架构目录，具体 Ascend C 调用需在实现时补齐。

#pragma once

#include "../chunk_gated_delta_rule_fwd_h_policy.h"
#include "../chunk_gated_delta_rule_fwd_h_utils.h"

namespace fwd_h_pseudocode {

struct CubeStage0Args {
    const ApiInputs* in = nullptr;
    const ApiOutputs* out = nullptr;
    const TilingPlan* tiling = nullptr;
    const RoundPlan* plan = nullptr;
    FixedMemory* memory = nullptr;
    SyncLedger* sync = nullptr;
};

struct CubeStage2Args {
    const ApiInputs* in = nullptr;
    const TilingPlan* tiling = nullptr;
    const RoundPlan* plan = nullptr;
    WorkspaceRefs* workspace = nullptr;
    FixedMemory* memory = nullptr;
    SyncLedger* sync = nullptr;
};

struct CubeStageResult {
    bool produced = false;
    int activeTaskCount = 0;
    int kgLoadCount = 0;
};

// ------------------------------- Stage 0 helpers -------------------------------

inline void Stage0LoadH(const CubeStage0Args& args, const HeadBinding& head)
{
    // Stage0 搬运公式：H_c,h = layout_decode(H_source,h)，并由 MTE2 将 GM 布局转为 L1 NZ。
    const RoundPlan& plan = *args.plan;
    const bool bf16Initial = plan.chunk.first && args.tiling->useInitialState &&
                             args.tiling->stateType == StateType::Bf16;

    if (bf16Initial) {
        // BF16 initial 由本 AIC 的 MTE2 直接写 canonical H[K,V]，不等待不存在的 AIV producer。
        if (args.memory->h[head.hSlot].generation > 0 && !plan.roundBoundaryDrained) {
            args.sync->Wait(EventKind::HFree, head.hSlot, /*S0 MTE2*/ 0);
        }
        args.memory->AcquireHForS0(head.hSlot);
        Mte2StateToCanonicalH(*args.in, head.hv, args.tiling->stateLayout,
                              L1H(args.memory->l1, head.hSlot));
        wait_mte2_h_l1_done(head.hSlot);
        args.memory->MarkHReady(head.hSlot);
        args.sync->Set(EventKind::HReady, head.hSlot, /*S0 MTE2*/ 0,
                       /*S0 MTE1*/ 1);
        return;
    }

    // FP32 initial 来自 S-1；后续 chunk 来自前一个 chunk 的 S3。producer 先把 layout-aware H
    // 写到 GM，S0 再用 MTE2 搬成 L1 NZ；禁止把 UB 数据直接写入 L1。
    args.sync->Wait(EventKind::HGmReady, head.hSlot, /*S0 MTE2*/ 0);
    Mte2CopyHFromGmLayoutAwareToL1NzAsync(
        *args.out, *args.tiling, *args.plan, head,
        L1H(args.memory->l1, head.hSlot));
    wait_mte2_h_l1_done(head.hSlot);
    args.sync->Set(EventKind::HReady, head.hSlot, /*S0 MTE2*/ 0,
                   /*S0 MTE1*/ 1);
    args.memory->BeginHReadFromS3(head.hSlot);
}

inline void Stage0LoadW(const CubeStage0Args& args, const HeadBinding& head)
{
    // Stage0 输入公式：W_c,h[i,:] = w[h,i,:]（0 <= i < M），无效尾行补零后驻留 L1。
    if (args.memory->w[head.wSlot].generation > 0 && !args.plan->roundBoundaryDrained) {
        args.sync->Wait(EventKind::WFree, head.wSlot, /*S0 MTE2*/ 0);
    }
    args.memory->AcquireWForS0(head.wSlot);

    // 当前 head 的 w_c 一次完整搬入；tail 行在 L1 内先清零再覆盖有效 M 行。
    Mte2InitConstValueAsync(L1W(args.memory->l1, head.wSlot), /*zero*/ 0,
                            /*bytes*/ 16 * 1024);
    Mte2CopyValidRowsAsync(Mte2WSource(*args.in, args.plan->chunk, head.hv),
                           L1W(args.memory->l1, head.wSlot), args.plan->chunk.validTokens);
    args.memory->MarkWReady(head.wSlot);
    args.sync->Set(EventKind::WReady, head.wSlot, /*S0 MTE2*/ 0,
                   /*S0 MTE1*/ 1);
}

inline void Stage0ComputeP(const CubeStage0Args& args, const HeadBinding& head)
{
    // Stage0 计算公式：P_c,h = W_c,h @ H_c,h，矩阵乘为 BF16 x BF16 -> FP32 累加。
    // S0 MTE1 只读取本 Stage 的 W/H 输入；P 由 Fixpipe 写入本 head 的 local data bank。
    args.sync->Wait(EventKind::WReady, head.wSlot, /*S0 MTE1*/ 1);
    args.sync->Wait(EventKind::HReady, head.hSlot, /*S0 MTE1*/ 1);
    Mte1Load(L1W(args.memory->l1, head.wSlot), L0A(head));
    Mte1Load(L1H(args.memory->l1, head.hSlot), L0B(head));
    Mte1ToCubeReady(/*W/H 已全部进入 L0*/ head);

    const auto& localData = args.memory->localData[FixedMemory::LocalBank(head)];
    if (localData.generation > 0 && !args.plan->roundBoundaryDrained) {
        const EventKind freeEvent = localData.previousOwner == LocalDataOwner::D ||
                                            localData.previousOwner == LocalDataOwner::HWrite
                                        ? EventKind::DFree
                                        : localData.previousOwner == LocalDataOwner::H0
                                            ? EventKind::LocalDataFree
                                            : EventKind::PFree;
        args.sync->Wait(freeEvent, head.roundHead, /*S0 Fixpipe*/ 0);
    }
    args.memory->AcquireLocalDataForP(head);
    const auto pAcc = MmadBf16AccFp32(L0A(head), L0B(head), args.plan->chunk.validTokens);
    // StateT=BF16 使用 F322BF16，StateT=FP32 使用 NoQuant；P 不经过 GM。
    FixpipePByStateType(pAcc, UbP(*args.memory, head), args.tiling->stateType);
    args.memory->MarkPReady(head);
    args.sync->Set(EventKind::PReady, head.roundHead, /*S0 Fixpipe*/ 0,
                   /*S1 VF*/ 1);

    // W/H 的最后一次消费者是本次 MTE1；释放后下一个 Stage 才能复用 L1 slot。
    args.memory->ReleaseWAfterS0Mte1(head.wSlot);
    args.memory->ReleaseHAfterS0Mte1(head.hSlot);
    args.sync->Release(EventKind::WFree, head.wSlot, /*S0 MTE1*/ 1);
    args.sync->Release(EventKind::HFree, head.hSlot, /*S0 MTE1*/ 1);
}

inline CubeStageResult RunStage0Arch22(const CubeStage0Args& args)
{
    // Stage0 阶段公式：对每个 active head 执行 P_c,h = W_c,h @ H_c,h，并将 P 写入 UB。
    CubeStageResult result{};
    const RoundPlan& plan = *args.plan;
    if (!plan.stage0Required) {
        // 首 chunk 无 initial 时，S0 不执行 W @ 0，也不发布 PReady。
        return result;
    }

    for (int i = 0; i < plan.activeHvCount; ++i) {
        const HeadBinding& head = plan.heads[i];
        Stage0LoadH(args, head);
        Stage0LoadW(args, head);
        Stage0ComputeP(args, head);
        ++result.activeTaskCount;
    }
    result.produced = true;
    return result;
}

// ------------------------------- Stage 2 helpers -------------------------------

inline void Stage2LoadKgForRound(const CubeStage2Args& args)
{
    // Stage2 左操作数公式：kg_c,kh = k_raw_c,kh（g-only）或 kg_c,kh（gk-only），按 distinct kh 装载。
    const RoundPlan& plan = *args.plan;
    // Stage2 自己负责当前 chunk 的全部 distinct kg 搬运；Stage0 不触碰 kg。
    for (int i = 0; i < plan.requiredKhCount; ++i) {
        const kg_binding& binding = plan.kg[i];
        auto& ticket = args.memory->kg[binding.slot];

        // 本 chunk/round 的 slot 只能在上一代最后一个 S2 MTE1 完成后覆写。
        if (ticket.generation > 0 && !plan.roundBoundaryDrained) {
            args.sync->Wait(EventKind::kg_overwrite_safe, binding.slot,
                            /*当前 Stage2 MTE2 生产者*/ 0);
        }
        args.memory->acquire_kg(binding.slot);

        // 先把整个 16 KiB entry 置零，再只覆盖 M 个有效 token；尾部不读取 GM padding。
        Mte2InitConstValueAsync(l1_kg(args.memory->l1, binding.slot), /*zero*/ 0,
                                /*bytes*/ 16 * 1024);
        Mte2CopyValidRowsAsync(
            /*源=*/KeyPayload(*args.in, plan.chunk, binding.kh, binding.payload),
            /*目的=*/l1_kg(args.memory->l1, binding.slot), plan.chunk.validTokens);
        // MTE2 先异步发起；Stage2 的首个 mapped head 等待该 slot 完成后再做 MTE1。
    }
}

inline void Stage2EnsureKgReady(const CubeStage2Args& args, const kg_binding& binding)
{
    // Stage2 kg 就绪语义：kg_c,kh[M,K] 完整搬入 L1 后，才允许映射到该 slot 的 head 做 MTE1。
    auto& ticket = args.memory->kg[binding.slot];
    if (ticket.state != SlotState::Loading) {
        // 同一 kg slot 的后续 mapped head 直接复用已经 ready 的 entry。
        return;
    }
    // Stage2LoadKgForRound 已经发起本代 MTE2；这里只等待一次，不重复读取 GM。
    wait_mte2_kg_done(binding.slot);
    args.memory->mark_kg_ready(binding.slot);
    args.sync->Set(EventKind::kg_ready, binding.slot, /*kg MTE2*/ 0,
                   /*S2 MTE1*/ 2);
}

inline void Stage2PrefetchRightToL1Nz(const CubeStage2Args& args, const HeadBinding& head)
{
    // Stage2 右操作数预取公式：g-only 搬 V_new_g,c,h，gk-only 搬 V_new_c,h；两者均由 ND 转为 NZ。
    const int rightGmSlot = head.roundHead;
    // Stage1 已经把 UB ND 写入 GM；先等 GM 写出，再异步发起 Stage2 的 ND -> NZ 搬运。
    args.sync->Wait(EventKind::RightGmReady, rightGmSlot, /*S2 MTE2*/ 0);
    args.memory->AcquireRightForS2Mte2(head.hSlot);
    Mte2CopyRightOperandGmNdToL1NzAsync(
        args.workspace->rightOperandGm, rightGmSlot, args.plan->chunk,
        L1Right(args.memory->l1, head.hSlot), args.plan->chunk.validTokens);
}

inline void Stage2EnsureRightL1Ready(const CubeStage2Args& args, const HeadBinding& head)
{
    // Stage2 右操作数就绪公式：L1Right_c,h[NZ] = convert_ND_to_NZ(V_new_g/V_new[GM])。
    const int rightGmSlot = head.roundHead;
    // 预取已经发起；每个 head 的 MTE1 只在本 head 的 L1 NZ 完成后继续。
    wait_mte2_right_l1_done(head.hSlot);
    args.memory->MarkRightL1Ready(head.hSlot);
    // GM scratch 在 MTE2 完成后已经没有消费者，允许下一 chunk/round 的 Stage1 复用。
    args.memory->ReleaseRightGmAfterS2Mte2(rightGmSlot);
    args.sync->Set(EventKind::RightGmFree, rightGmSlot, /*S2 MTE2*/ 0,
                   /*下一 S1 MTE3*/ 1);
    args.sync->Set(EventKind::RightL1Ready, head.hSlot, /*S2 MTE2*/ 0,
                   /*S2 MTE1*/ 2);
}

inline void Stage2ComputeDForHead(const CubeStage2Args& args, const HeadBinding& head)
{
    // Stage2 计算公式：g-only 为 D_c,h = k_raw_c,kh^T @ V_new_g,c,h；gk-only 为 D_c,h = kg_c,kh^T @ V_new_c,h。
    const kg_binding& binding = args.plan->kg[head.kgSlot];
    Stage2EnsureKgReady(args, binding);

    // 一个 kg slot 的 ready token 只由第一个映射 head 消费；后续 head 直接复用 valid entry。
    if (head.roundHead == binding.firstConsumerRoundHead) {
        args.sync->Wait(EventKind::kg_ready, binding.slot, /*S2 MTE1*/ 2);
    }
    // 子模块 Stage2EnsureRightL1Ready：确认预取的 GM ND -> L1 NZ 已完成。
    Stage2EnsureRightL1Ready(args, head);
    // L1 NZ 已经由本 head 的 Stage2 MTE2 生成，之后才允许 MTE1 读右操作数。
    args.sync->Wait(EventKind::RightL1Ready, head.hSlot, /*S2 MTE1*/ 2);
    const auto& localData = args.memory->localData[FixedMemory::LocalBank(head)];
    if (localData.generation > 0 && !args.plan->roundBoundaryDrained) {
        const EventKind freeEvent = localData.previousOwner == LocalDataOwner::P ||
                                            localData.previousOwner == LocalDataOwner::VNewWork
                                        ? EventKind::PFree
                                        : localData.previousOwner == LocalDataOwner::H0
                                            ? EventKind::LocalDataFree
                                            : EventKind::DFree;
        args.sync->Wait(freeEvent, head.roundHead, /*S2 Fixpipe*/ 2);
    }
    args.memory->AcquireLocalDataForD(head);

    Mte1Load(l1_kg(args.memory->l1, binding.slot), L0A(head));
    Mte1Load(L1Right(args.memory->l1, head.hSlot), L0B(head));
    Mte1ToCubeReady(/*kg/right 已全部进入 L0*/ head);

    // g-only 物理 kg slot 保存 k_raw，右操作数是 V_new_g；gk-only 保存 prepared kg，右操作数是 V_new。
    const auto dAcc = MmadBf16AccFp32(L0A(head), L0B(head), args.plan->chunk.validTokens);
    FixpipeDNoQuant(dAcc, UbD(*args.memory, head));
    args.memory->MarkDReady(head);
    args.sync->Set(EventKind::DReady, head.roundHead, /*S2 Fixpipe*/ 2,
                   /*S3 VF*/ 3);

    // 右操作数和 kg 均按最后一个 MTE1 消费者释放，不能按 head loop 结束猜测。
    args.memory->ReleaseRightAfterS2Mte1(head.hSlot);
    args.sync->Release(EventKind::RightFree, head.hSlot, /*S2 MTE1*/ 2);
    if (binding.lastConsumerRoundHead == head.roundHead) {
        args.memory->release_kg_after_last_s2_mte1(binding.slot);
        args.sync->Release(EventKind::kg_overwrite_safe, binding.slot, /*S2 MTE1*/ 2);
    }
}

inline CubeStageResult RunStage2Arch22(const CubeStage2Args& args)
{
    // Stage2 阶段公式：先预取 Nkg_round 个 kg 和全部 V_new_g/V_new，再逐 head 按分支计算 D_c,h。
    CubeStageResult result{};
    const RoundPlan& plan = *args.plan;
    if (!plan.stage2Required) {
        // v_new-only 最终 chunk 没有 kg/right/D 消费者，S2 整体跳过，也不搬运 kg。
        return result;
    }

    Stage2LoadKgForRound(args);
    // 先为本 round 的所有 value head 发起右操作数预取；后续每个 head 在 MTE1 前确认就绪。
    for (int i = 0; i < plan.activeHvCount; ++i) {
        Stage2PrefetchRightToL1Nz(args, plan.heads[i]);
    }
    // 每个 active value head 执行一次完整逻辑转置 MMAD；共享 kh 的 head 共享 L1 kg slot。
    for (int i = 0; i < plan.activeHvCount; ++i) {
        Stage2ComputeDForHead(args, plan.heads[i]);
        ++result.activeTaskCount;
    }
    result.kgLoadCount = plan.requiredKhCount;
    result.produced = true;
    return result;
}

} // 命名空间 fwd_h_pseudocode
