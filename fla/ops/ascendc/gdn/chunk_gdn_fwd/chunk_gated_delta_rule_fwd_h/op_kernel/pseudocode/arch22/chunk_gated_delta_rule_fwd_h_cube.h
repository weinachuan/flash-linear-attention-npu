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
    FixedMemory* memory = nullptr;
    SyncLedger* sync = nullptr;
};

struct CubeStageResult {
    bool produced = false;
    int activeTaskCount = 0;
    int kgPrefetchCount = 0;
};

// ------------------------------- Stage 0 helpers -------------------------------

inline void Stage0PrefetchKgAsync(const CubeStage0Args& args)
{
    const RoundPlan& plan = *args.plan;
    if (!plan.stage2Required) {
        // 最终 chunk 且不输出 final_state 时，S2 不存在，不为无消费者的 kg 发起搬运。
        return;
    }

    for (int i = 0; i < plan.requiredKhCount; ++i) {
        const kg_binding& binding = plan.kg[i];
        auto& ticket = args.memory->kg[binding.slot];

        // 本 chunk/round 的 slot 只能在上一代最后一个 S2 MTE1 完成后覆写。
        if (ticket.generation > 0 && !plan.roundBoundaryDrained) {
            args.sync->Wait(EventKind::kg_overwrite_safe, binding.slot,
                            /*当前 Stage0 的 MTE2 生产者*/ 0);
        }
        args.memory->acquire_kg(binding.slot);

        // 先把整个 16 KiB entry 置零，再只覆盖 M 个有效 token；尾部不读取 GM padding。
        Mte2InitConstValueAsync(l1_kg(args.memory->l1, binding.slot), /*zero*/ 0,
                                /*bytes*/ 16 * 1024);
        Mte2CopyValidRowsAsync(
            /*源=*/KeyPayload(*args.in, plan.chunk, binding.kh, binding.payload),
            /*目的=*/l1_kg(args.memory->l1, binding.slot), plan.chunk.validTokens);
        // 这里故意不 wait_mte2_done：kg 在 S1 期间保持 Loading，S2 的首个消费者负责等待。
    }
}

inline void Stage0LoadH(const CubeStage0Args& args, const HeadBinding& head)
{
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
        args.memory->MarkHReady(head.hSlot);
        args.sync->Set(EventKind::HReady, head.hSlot, /*S0 MTE2*/ 0,
                       /*S0 MTE1*/ 1);
        return;
    }

    // FP32 initial 来自 S-1；后续 chunk 来自前一个 chunk 的 S3。
    // 两者都必须由对应 producer 发布 HReady 后，S0 才能消费。
    args.sync->Wait(EventKind::HReady, head.hSlot, /*S0 MTE1*/ 1);
    args.memory->BeginHReadFromS3(head.hSlot);
}

inline void Stage0LoadW(const CubeStage0Args& args, const HeadBinding& head)
{
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
    CubeStageResult result{};
    const RoundPlan& plan = *args.plan;
    if (!plan.stage0Required) {
        // 首 chunk 无 initial 时，S0 不执行 W @ 0，也不发布 PReady。
        return result;
    }

    // kg 预取与 H/W 搬运同属 S0，但 kg 只异步发起，不能在这里等待其完成。
    Stage0PrefetchKgAsync(args);
    for (int i = 0; i < plan.activeHvCount; ++i) {
        const HeadBinding& head = plan.heads[i];
        Stage0LoadH(args, head);
        Stage0LoadW(args, head);
        Stage0ComputeP(args, head);
        ++result.activeTaskCount;
    }
    result.kgPrefetchCount = plan.stage2Required ? plan.requiredKhCount : 0;
    result.produced = true;
    return result;
}

// ------------------------------- Stage 2 helpers -------------------------------

inline void Stage2EnsureKgReady(const CubeStage2Args& args, const kg_binding& binding)
{
    auto& ticket = args.memory->kg[binding.slot];
    if (ticket.state == SlotState::Ready) {
        return;
    }

    if (ticket.state == SlotState::Loading) {
        // S0 已经发起异步 MTE2；这里只等待该 slot 的本代完成，不重复 GM 读取。
        wait_mte2_kg_done(binding.slot);
        args.memory->mark_kg_ready(binding.slot);
        args.sync->Set(EventKind::kg_ready, binding.slot, /*kg MTE2*/ 0,
                       /*S2 MTE1*/ 2);
        return;
    }

    // S0 被跳过或未预取时，S2 只补齐这个缺失的 distinct (chunk, kh) entry。
    if (ticket.generation > 0 && !args.plan->roundBoundaryDrained) {
        args.sync->Wait(EventKind::kg_overwrite_safe, binding.slot,
                        /*S2 的 MTE2 生产者*/ 0);
    }
    args.memory->acquire_kg(binding.slot);
    Mte2InitConstValue(l1_kg(args.memory->l1, binding.slot), /*zero*/ 0,
                       /*bytes*/ 16 * 1024);
    Mte2CopyValidRows(
        /*源=*/KeyPayload(*args.in, args.plan->chunk, binding.kh, binding.payload),
        /*目的=*/l1_kg(args.memory->l1, binding.slot), args.plan->chunk.validTokens);
    args.memory->mark_kg_ready(binding.slot);
    args.sync->Set(EventKind::kg_ready, binding.slot, /*kg MTE2*/ 0,
                   /*S2 MTE1*/ 2);
}

inline void Stage2ComputeDForHead(const CubeStage2Args& args, const HeadBinding& head)
{
    const kg_binding& binding = args.plan->kg[head.kgSlot];
    Stage2EnsureKgReady(args, binding);

    // 一个 kg slot 的 ready token 只由第一个映射 head 消费；后续 head 直接复用 valid entry。
    if (head.roundHead == binding.firstConsumerRoundHead) {
        args.sync->Wait(EventKind::kg_ready, binding.slot, /*S2 MTE1*/ 2);
    }
    // 子模块 Stage2WaitRightAndAcquireD：RightReady 同时证明 S1 右操作数 MTE3
    // 已完成、P owner 已释放，随后才能获取 D UB bank。
    args.sync->Wait(EventKind::RightReady, head.hSlot, /*S2 MTE1*/ 2);
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
    CubeStageResult result{};
    const RoundPlan& plan = *args.plan;
    if (!plan.stage2Required) {
        // v_new-only 最终 chunk 没有 kg/right/D 消费者，S2 整体跳过。
        return result;
    }

    // 每个 active value head 执行一次完整逻辑转置 MMAD；共享 kh 的 head 共享 L1 kg slot。
    for (int i = 0; i < plan.activeHvCount; ++i) {
        Stage2ComputeDForHead(args, plan.heads[i]);
        ++result.activeTaskCount;
    }
    result.produced = true;
    return result;
}

} // 命名空间 fwd_h_pseudocode
