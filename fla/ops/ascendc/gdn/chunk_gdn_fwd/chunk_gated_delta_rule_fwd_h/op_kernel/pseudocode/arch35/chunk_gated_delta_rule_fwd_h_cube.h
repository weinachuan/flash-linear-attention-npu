// 仅伪代码。arch35（A5）Cube 阶段：Stage0 和 Stage2。
// 本文件不包含 arch22 Cube 头文件；A5 的 MTE2/MTE1/MMAD/Fixpipe 路径在此独立描述。

#pragma once

#include "../chunk_gated_delta_rule_fwd_h_policy.h"
#include "../chunk_gated_delta_rule_fwd_h_utils.h"

namespace fwd_h_pseudocode {

// ------------------------------- Stage 0 helpers -------------------------------

inline void Stage0LoadHArch35(const CubeStage0Args& args, const HeadBinding& head)
{
    // Stage0 搬运公式：H_c,h = layout_decode(H_source,h)，A5 MTE2 将 GM 布局转为 L1 NZ。
    const RoundPlan& plan = *args.plan;
    const bool bf16Initial = plan.chunk.first && args.tiling->useInitialState &&
                             args.tiling->stateType == StateType::Bf16;

    if (bf16Initial) {
        // BF16 initial 由 A5 AIC 的 MTE2 直接写 canonical H[K,V]。
        if (args.memory->h[head.hSlot].generation > 0 && !plan.roundBoundaryDrained) {
            args.sync->Wait(EventKind::HFree, head.hSlot, /*A5 S0 MTE2*/ 0);
        }
        args.memory->AcquireHForS0(head.hSlot);
        Mte2StateToCanonicalHArch35(*args.in, head.hv, args.tiling->stateLayout,
                                    L1H(args.memory->l1, head.hSlot));
        wait_mte2_h_l1_done_arch35(head.hSlot);
        args.memory->MarkHReady(head.hSlot);
        args.sync->Set(EventKind::HReady, head.hSlot, /*A5 S0 MTE2*/ 0,
                       /*A5 S0 MTE1*/ 1);
        return;
    }

    // FP32 initial 来自 S-1，后续 chunk 来自前一个 S3；GM layout-aware H 再转为 L1 NZ。
    args.sync->Wait(EventKind::HGmReady, head.hSlot, /*A5 S0 MTE2*/ 0);
    Mte2CopyHFromGmLayoutAwareToL1NzAsyncArch35(
        *args.out, *args.tiling, *args.plan, head,
        L1H(args.memory->l1, head.hSlot));
    wait_mte2_h_l1_done_arch35(head.hSlot);
    args.sync->Set(EventKind::HReady, head.hSlot, /*A5 S0 MTE2*/ 0,
                   /*A5 S0 MTE1*/ 1);
    args.memory->BeginHReadFromS3(head.hSlot);
}

inline void Stage0LoadWArch35(const CubeStage0Args& args, const HeadBinding& head)
{
    // Stage0 输入公式：W_c,h[i,:] = w[h,i,:]（0 <= i < M），无效尾行补零。
    if (args.memory->w[head.wSlot].generation > 0 && !args.plan->roundBoundaryDrained) {
        args.sync->Wait(EventKind::WFree, head.wSlot, /*A5 S0 MTE2*/ 0);
    }
    args.memory->AcquireWForS0(head.wSlot);
    Mte2InitConstValueAsyncArch35(L1W(args.memory->l1, head.wSlot), /*zero*/ 0,
                                  /*bytes*/ 16 * 1024);
    Mte2CopyValidRowsAsyncArch35(Mte2WSource(*args.in, args.plan->chunk, head.hv),
                                 L1W(args.memory->l1, head.wSlot),
                                 args.plan->chunk.validTokens);
    args.memory->MarkWReady(head.wSlot);
    args.sync->Set(EventKind::WReady, head.wSlot, /*A5 S0 MTE2*/ 0,
                   /*A5 S0 MTE1*/ 1);
}

inline void Stage0ComputePArch35(const CubeStage0Args& args, const HeadBinding& head)
{
    // Stage0 计算公式：P_c,h = W_c,h @ H_c,h，BF16 x BF16 -> FP32 L0C。
    // A5 支持 L0C -> UB，P 不经过 GM；Fixpipe 按 StateT 选择量化格式。
    args.sync->Wait(EventKind::WReady, head.wSlot, /*A5 S0 MTE1*/ 1);
    args.sync->Wait(EventKind::HReady, head.hSlot, /*A5 S0 MTE1*/ 1);
    Mte1LoadArch35(L1W(args.memory->l1, head.wSlot), L0AArch35(head));
    Mte1LoadArch35(L1H(args.memory->l1, head.hSlot), L0BArch35(head));
    Mte1ToCubeReadyArch35(/*W/H 已全部进入 L0*/ head);

    const auto& localData = args.memory->localData[FixedMemory::LocalBank(head)];
    if (localData.generation > 0 && !args.plan->roundBoundaryDrained) {
        const EventKind freeEvent = localData.previousOwner == LocalDataOwner::D ||
                                            localData.previousOwner == LocalDataOwner::HWrite
                                        ? EventKind::DFree
                                        : localData.previousOwner == LocalDataOwner::H0
                                            ? EventKind::LocalDataFree
                                            : EventKind::PFree;
        args.sync->Wait(freeEvent, head.roundHead, /*A5 S0 Fixpipe*/ 0);
    }
    args.memory->AcquireLocalDataForP(head);
    const auto pAcc = MmadBf16AccFp32Arch35(L0AArch35(head), L0BArch35(head),
                                                args.plan->chunk.validTokens);
    FixpipePByStateTypeArch35(pAcc, UbP(*args.memory, head), args.tiling->stateType);
    args.memory->MarkPReady(head);
    args.sync->Set(EventKind::PReady, head.roundHead, /*A5 S0 Fixpipe*/ 0,
                   /*S1 VF*/ 1);

    args.memory->ReleaseWAfterS0Mte1(head.wSlot);
    args.memory->ReleaseHAfterS0Mte1(head.hSlot);
    args.sync->Release(EventKind::WFree, head.wSlot, /*A5 S0 MTE1*/ 1);
    args.sync->Release(EventKind::HFree, head.hSlot, /*A5 S0 MTE1*/ 1);
}

inline CubeStageResult RunStage0Arch35(const CubeStage0Args& args)
{
    // Stage0 阶段公式：对每个 active head 执行 P_c,h = W_c,h @ H_c,h，并由 A5 Fixpipe 写 UB。
    CubeStageResult result{};
    const RoundPlan& plan = *args.plan;
    if (!plan.stage0Required) {
        // 首 chunk 无 initial 时，S0 不执行 W @ 0，也不发布 PReady。
        return result;
    }
    for (int i = 0; i < plan.activeHvCount; ++i) {
        const HeadBinding& head = plan.heads[i];
        Stage0LoadHArch35(args, head);
        Stage0LoadWArch35(args, head);
        Stage0ComputePArch35(args, head);
        ++result.activeTaskCount;
    }
    result.produced = true;
    return result;
}

// ------------------------------- Stage 2 helpers -------------------------------

inline void Stage2LoadKgForRoundArch35(const CubeStage2Args& args)
{
    // Stage2 左操作数公式：kg_c,kh = k_raw_c,kh（g-only）或 kg_c,kh（gk-only）。
    const RoundPlan& plan = *args.plan;
    for (int i = 0; i < plan.requiredKhCount; ++i) {
        const kg_binding& binding = plan.kg[i];
        auto& ticket = args.memory->kg[binding.slot];
        if (ticket.generation > 0 && !plan.roundBoundaryDrained) {
            args.sync->Wait(EventKind::kg_overwrite_safe, binding.slot,
                            /*A5 Stage2 MTE2*/ 0);
        }
        args.memory->acquire_kg(binding.slot);
        Mte2InitConstValueAsyncArch35(l1_kg(args.memory->l1, binding.slot), /*zero*/ 0,
                                      /*bytes*/ 16 * 1024);
        Mte2CopyValidRowsAsyncArch35(
            /*源=*/KeyPayload(*args.in, plan.chunk, binding.kh, binding.payload),
            /*目的=*/l1_kg(args.memory->l1, binding.slot), plan.chunk.validTokens);
        // A5 MTE2 只在此处发起一次；首个 mapped head 等待 ready 后再做 MTE1。
    }
}

inline void Stage2EnsureKgReadyArch35(const CubeStage2Args& args, const kg_binding& binding)
{
    // kg_c,kh 完整进入 A5 L1 后，才允许映射到该 slot 的 head 做 MTE1。
    auto& ticket = args.memory->kg[binding.slot];
    if (ticket.state != SlotState::Loading) {
        return;
    }
    wait_mte2_kg_done_arch35(binding.slot);
    args.memory->mark_kg_ready(binding.slot);
    args.sync->Set(EventKind::kg_ready, binding.slot, /*A5 kg MTE2*/ 0,
                   /*A5 S2 MTE1*/ 2);
}

inline void Stage2PrefetchRightToL1NzArch35(const CubeStage2Args& args,
                                            const HeadBinding& head)
{
    // Stage2 右操作数公式：g-only 搬 V_new_g，gk-only 搬 V_new；GM ND 转为 L1 NZ。
    const int rightGmSlot = head.roundHead;
    args.sync->Wait(EventKind::RightGmReady, rightGmSlot, /*A5 S2 MTE2*/ 0);
    args.memory->AcquireRightForS2Mte2(head.hSlot);
    Mte2CopyRightOperandGmNdToL1NzAsyncArch35(
        args.workspace->rightOperandGm, rightGmSlot, args.plan->chunk,
        L1Right(args.memory->l1, head.hSlot), args.plan->chunk.validTokens);
}

inline void Stage2EnsureRightL1ReadyArch35(const CubeStage2Args& args,
                                            const HeadBinding& head)
{
    // L1Right_c,h[NZ] 只有在 A5 MTE2 完成后才能被 MTE1 读取。
    const int rightGmSlot = head.roundHead;
    wait_mte2_right_l1_done_arch35(head.hSlot);
    args.memory->MarkRightL1Ready(head.hSlot);
    args.memory->ReleaseRightGmAfterS2Mte2(rightGmSlot);
    args.sync->Set(EventKind::RightGmFree, rightGmSlot, /*A5 S2 MTE2*/ 0,
                   /*下一 S1 MTE3*/ 1);
    args.sync->Set(EventKind::RightL1Ready, head.hSlot, /*A5 S2 MTE2*/ 0,
                   /*A5 S2 MTE1*/ 2);
}

inline void Stage2ComputeDForHeadArch35(const CubeStage2Args& args, const HeadBinding& head)
{
    // Stage2 计算公式：g-only 为 D_c,h = k_raw_c,kh^T @ V_new_g,c,h；gk-only 为 D_c,h = kg_c,kh^T @ V_new_c,h。
    const kg_binding& binding = args.plan->kg[head.kgSlot];
    Stage2EnsureKgReadyArch35(args, binding);
    if (head.roundHead == binding.firstConsumerRoundHead) {
        args.sync->Wait(EventKind::kg_ready, binding.slot, /*A5 S2 MTE1*/ 2);
    }
    Stage2EnsureRightL1ReadyArch35(args, head);
    args.sync->Wait(EventKind::RightL1Ready, head.hSlot, /*A5 S2 MTE1*/ 2);

    const auto& localData = args.memory->localData[FixedMemory::LocalBank(head)];
    if (localData.generation > 0 && !args.plan->roundBoundaryDrained) {
        const EventKind freeEvent = localData.previousOwner == LocalDataOwner::P ||
                                            localData.previousOwner == LocalDataOwner::VNewWork
                                        ? EventKind::PFree
                                        : localData.previousOwner == LocalDataOwner::H0
                                            ? EventKind::LocalDataFree
                                            : EventKind::DFree;
        args.sync->Wait(freeEvent, head.roundHead, /*A5 S2 Fixpipe*/ 2);
    }
    args.memory->AcquireLocalDataForD(head);
    Mte1LoadArch35(l1_kg(args.memory->l1, binding.slot), L0AArch35(head));
    Mte1LoadArch35(L1Right(args.memory->l1, head.hSlot), L0BArch35(head));
    Mte1ToCubeReadyArch35(/*kg/right 已全部进入 L0*/ head);
    const auto dAcc = MmadBf16AccFp32Arch35(L0AArch35(head), L0BArch35(head),
                                                args.plan->chunk.validTokens);
    // A5 支持 L0C -> UB，D 保持 FP32 NoQuant，不经过 GM。
    FixpipeDNoQuantArch35(dAcc, UbD(*args.memory, head));
    args.memory->MarkDReady(head);
    args.sync->Set(EventKind::DReady, head.roundHead, /*A5 S2 Fixpipe*/ 2,
                   /*S3 VF*/ 3);

    args.memory->ReleaseRightAfterS2Mte1(head.hSlot);
    args.sync->Release(EventKind::RightFree, head.hSlot, /*A5 S2 MTE1*/ 2);
    if (binding.lastConsumerRoundHead == head.roundHead) {
        args.memory->release_kg_after_last_s2_mte1(binding.slot);
        args.sync->Release(EventKind::kg_overwrite_safe, binding.slot,
                           /*A5 S2 MTE1*/ 2);
    }
}

inline CubeStageResult RunStage2Arch35(const CubeStage2Args& args)
{
    // Stage2 阶段公式：先预取 Nkg_round 个 kg 和全部右操作数，再逐 head 计算 D_c,h。
    CubeStageResult result{};
    const RoundPlan& plan = *args.plan;
    if (!plan.stage2Required) {
        // 最终 v_new-only 没有 kg/right/D 消费者，S2 整体跳过。
        return result;
    }
    Stage2LoadKgForRoundArch35(args);
    for (int i = 0; i < plan.activeHvCount; ++i) {
        Stage2PrefetchRightToL1NzArch35(args, plan.heads[i]);
    }
    for (int i = 0; i < plan.activeHvCount; ++i) {
        Stage2ComputeDForHeadArch35(args, plan.heads[i]);
        ++result.activeTaskCount;
    }
    result.kgLoadCount = plan.requiredKhCount;
    result.produced = true;
    return result;
}

} // namespace fwd_h_pseudocode
