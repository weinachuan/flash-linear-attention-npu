// 仅伪代码。arch22（A2/A3）Cube 阶段：Stage0 和 Stage2。
// 该文件保留真实 op_kernel 的文件名和架构目录，具体 Ascend C 调用需在实现时补齐。

#ifndef FLA_FWD_H_PSEUDOCODE_ARCH22_CHUNK_GATED_DELTA_RULE_FWD_H_CUBE_H_
#define FLA_FWD_H_PSEUDOCODE_ARCH22_CHUNK_GATED_DELTA_RULE_FWD_H_CUBE_H_

#include "../chunk_gated_delta_rule_fwd_h_policy.h"
#include "../chunk_gated_delta_rule_fwd_h_utils.h"

namespace fwd_h_pseudocode {

inline TensorRef& Arch22PScratch(WorkspaceRefs& workspace, int roundHead)
{
    // A2/A3 仅在本架构文件内把扩展槽 0 解释为 P 的 GM 中转。
    return workspace.cubeScratch0[roundHead];
}

inline TensorRef& Arch22DScratch(WorkspaceRefs& workspace, int roundHead)
{
    // A2/A3 仅在本架构文件内把扩展槽 1 解释为 D 的 GM 中转。
    return workspace.cubeScratch1[roundHead];
}

inline void PrepareArch22CubeWorkspace(WorkspaceRefs& workspace, const TilingPlan& tiling)
{
    // 只有 A2/A3 路径申请 L0C->GM->MTE2->UB 所需的中转区；A5 不调用此函数。
    for (int slot = 0; slot < kMaxRoundHeads; ++slot) {
        workspace.cubeScratch0[slot].shape = {kBatchTokens, kValueDim, 0, 0, 0};
        workspace.cubeScratch0[slot].dtype = tiling.stateType;
        workspace.cubeScratch0[slot].present = true;
        workspace.cubeScratch1[slot].shape = {kKeyDim, kValueDim, 0, 0, 0};
        workspace.cubeScratch1[slot].dtype = StateType::Fp32;
        workspace.cubeScratch1[slot].present = true;
    }
}

// A2/A3 不提供 L0C -> UB。Cube 结果必须先由 Fixpipe 写入 GM，再由 MTE2 搬回配对 AIV 的 UB。
inline void Arch22Mte2PFromGmToUb(const CubeStage0Args& args, const HeadBinding& head)
{
    Mte2CopyPFromGmToUbArch22(Arch22PScratch(*args.workspace, head.roundHead),
                              UbP(*args.memory, head), args.plan->chunk,
                              args.tiling->stateType);
}

inline void Arch22Mte2DFromGmToUb(const CubeStage2Args& args, const HeadBinding& head)
{
    Mte2CopyDFromGmToUbArch22(Arch22DScratch(*args.workspace, head.roundHead),
                              UbD(*args.memory, head), args.plan->chunk);
}

// ------------------------------- Stage 0 helpers -------------------------------

inline void Stage0LoadHArch22(const CubeStage0Args& args, const HeadBinding& head)
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
        Mte2StateToCanonicalHArch22(*args.in, head.hv, args.tiling->stateLayout,
                                    L1H(args.memory->l1, head.hSlot));
        args.memory->MarkHReady(head.hSlot);
        args.sync->Set(EventKind::HReady, head.hSlot, /*S0 MTE2*/ 0,
                       /*S0 MTE1*/ 1);
        return;
    }

    // FP32 initial 来自 S-1；后续 chunk 来自前一个 chunk 的 S3。producer 先把 layout-aware H
    // 写到 GM，S0 再用 MTE2 搬成 L1 NZ；禁止把 UB 数据直接写入 L1。
    args.sync->Wait(EventKind::HGmReady, head.hSlot, /*S0 MTE2*/ 0);
    if (plan.chunk.chunk > 0) {
        // L1 H 与上一 chunk 的 right 使用同一物理 slot；只等本 head 的最后一次 right MTE1。
        args.sync->Wait(EventKind::RightFree, head.hSlot, /*S0 MTE2*/ 0);
    }
    Mte2CopyHFromGmLayoutAwareToL1NzAsyncArch22(
        *args.out, *args.tiling, *args.plan, head,
        L1H(args.memory->l1, head.hSlot));
    args.sync->Set(EventKind::HReady, head.hSlot, /*S0 MTE2*/ 0,
                   /*S0 MTE1*/ 1);
    args.memory->BeginHReadFromS3(head.hSlot);
}

inline void Stage0LoadWArch22(const CubeStage0Args& args, const HeadBinding& head)
{
    // Stage0 输入公式：W_c,h[i,:] = w[h,i,:]（0 <= i < M），无效尾行补零后驻留 L1。
    if (args.memory->w[head.wSlot].generation > 0 && !args.plan->roundBoundaryDrained) {
        args.sync->Wait(EventKind::WFree, head.wSlot, /*S0 MTE2*/ 0);
    }
    args.memory->AcquireWForS0(head.wSlot);

    // 当前 head 的 w_c 一次完整搬入；tail 行在 L1 内先清零再覆盖有效 M 行。
    Mte2InitConstValueAsyncArch22(L1W(args.memory->l1, head.wSlot), /*zero*/ 0,
                                  /*bytes*/ 16 * 1024);
    Mte2CopyValidRowsAsyncArch22(Mte2WSource(*args.in, args.plan->chunk, head.hv),
                                 L1W(args.memory->l1, head.wSlot), args.plan->chunk.validTokens);
    args.memory->MarkWReady(head.wSlot);
    args.sync->Set(EventKind::WReady, head.wSlot, /*S0 MTE2*/ 0,
                   /*S0 MTE1*/ 1);
}

inline void Stage0ComputePArch22(const CubeStage0Args& args, const HeadBinding& head,
                                 int pipelineSlot)
{
    // Stage0 计算公式：P_c,h = W_c,h @ H_c,h，矩阵乘为 BF16 x BF16 -> FP32 累加。
    // S0 MTE1 只读取本 Stage 的 W/H 输入；P 经过 A2 GM scratch 中转后进入本 head 的 local data bank。
    if (args.plan->chunk.chunk > 0) {
        // AIC 四槽在本 round 内互不复用；这里只接收上一 chunk Stage2 对同一物理槽的 free。
        // 跨 round 已由动态 CubeRoundDrain 收口，不能等待上一 round 未必生成的槽级 token。
        args.sync->Wait(EventKind::S2CubeToMte1Free, pipelineSlot, /*S0 MTE1*/ 1);
    }
    args.sync->Wait(EventKind::S0Mte2ToMte1Ready, pipelineSlot, /*S0 MTE1*/ 1);
    args.sync->Wait(EventKind::WReady, head.wSlot, /*S0 MTE1*/ 1);
    args.sync->Wait(EventKind::HReady, head.hSlot, /*S0 MTE1*/ 1);
    Mte1LoadArch22(L1W(args.memory->l1, head.wSlot), L0AArch22(head));
    Mte1LoadArch22(L1H(args.memory->l1, head.hSlot), L0BArch22(head));
    args.sync->Set(EventKind::S0Mte1ToCubeReady, pipelineSlot,
                   /*S0 MTE1*/ 1, /*S0 Cube*/ 2);
    args.sync->Wait(EventKind::S0Mte1ToCubeReady, pipelineSlot, /*S0 Cube*/ 2);
    if (args.plan->chunk.chunk > 0) {
        args.sync->Wait(EventKind::S2FixpipeToCubeFree, pipelineSlot, /*S0 Cube*/ 2);
    }

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
    const auto pAcc = MmadBf16AccFp32Arch22(L0AArch22(head), L0BArch22(head),
                                               args.plan->chunk.validTokens);
    if (args.plan->stage2Required) {
        // MMAD 已完成 L0A/L0B 末读；仅为当前 chunk Stage2 复用同一 AIC 槽发布 free。
        args.sync->Set(EventKind::S0CubeToMte1Free, pipelineSlot,
                       /*S0 Cube*/ 2, /*当前 S2 MTE1*/ 1);
    }
    args.sync->Set(EventKind::S0CubeToFixpipeReady, pipelineSlot,
                   /*S0 Cube*/ 2, /*S0 Fixpipe*/ 3);
    args.sync->Wait(EventKind::S0CubeToFixpipeReady, pipelineSlot, /*S0 Fixpipe*/ 3);
    // A2/A3 没有 L0C -> UB，P 走 L0C -> GM -> MTE2 -> UB；StateT 决定 GM 中转时的量化格式。
    wait_arch22_p_gm_slot_free(Arch22PScratch(*args.workspace, head.roundHead));
    FixpipePByStateTypeToGmArch22(pAcc, Arch22PScratch(*args.workspace, head.roundHead),
                                  args.plan->chunk, args.tiling->stateType);
    args.sync->Set(EventKind::S0FixpipeToMte2Ready, pipelineSlot,
                   /*S0 Fixpipe*/ 3, /*S0 MTE2*/ 0);
    if (args.plan->stage2Required) {
        // Fixpipe 已完成 L0C 末读；仅解锁当前 chunk Stage2 对同一 AIC 槽 L0C 的覆盖。
        args.sync->Set(EventKind::S0FixpipeToCubeFree, pipelineSlot,
                       /*S0 Fixpipe*/ 3, /*当前 S2 Cube*/ 2);
    }
    args.sync->Wait(EventKind::S0FixpipeToMte2Ready, pipelineSlot, /*S0 MTE2*/ 0);
    Arch22Mte2PFromGmToUb(args, head);
    args.memory->MarkPReady(head);
    args.sync->Set(EventKind::PReady, head.roundHead, /*A2 S0 MTE2 GM->UB*/ 0,
                   /*S1 VF*/ 1);
    args.sync->Release(EventKind::PgmFree, head.roundHead, /*A2 S0 MTE2 GM->UB*/ 0);

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
    PrepareArch22CubeWorkspace(*args.workspace, *args.tiling);

    for (int coreHeadId = 0; coreHeadId < plan.activeHvCount; ++coreHeadId) {
        const CoreHeadBinding current = BindAicCoreHead(plan, coreHeadId);
        const HeadBinding& head = plan.heads[current.roundHead];
        Stage0LoadHArch22(args, head);
        Stage0LoadWArch22(args, head);
        args.sync->Set(EventKind::S0Mte2ToMte1Ready, current.pipelineSlot,
                       /*S0 MTE2*/ 0, /*S0 MTE1*/ 1);
        if (coreHeadId > 0) {
            const CoreHeadBinding previous = BindAicCoreHead(plan, coreHeadId - 1);
            Stage0ComputePArch22(args, plan.heads[previous.roundHead],
                                 previous.pipelineSlot);
        }
        ++result.activeTaskCount;
    }
    if (plan.activeHvCount > 0) {
        const int lastCoreHeadId = plan.activeHvCount - 1;
        const CoreHeadBinding last = BindAicCoreHead(plan, lastCoreHeadId);
        Stage0ComputePArch22(args, plan.heads[last.roundHead], last.pipelineSlot);
    }
    result.produced = true;
    return result;
}

// ------------------------------- Stage 2 helpers -------------------------------

inline void Stage2LoadKgForRoundArch22(const CubeStage2Args& args)
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
        Mte2InitConstValueAsyncArch22(l1_kg(args.memory->l1, binding.slot), /*zero*/ 0,
                                      /*bytes*/ 16 * 1024);
        Mte2CopyValidRowsAsyncArch22(
            /*源=*/KeyPayload(*args.in, plan.chunk, binding.kh, binding.payload),
            /*目的=*/l1_kg(args.memory->l1, binding.slot), plan.chunk.validTokens);
        args.memory->mark_kg_ready(binding.slot);
        args.sync->Set(EventKind::kg_ready, binding.slot, /*kg MTE2*/ 0,
                       /*S2 MTE1*/ 2);
    }
}

inline void Stage2EnsureKgReadyArch22(const CubeStage2Args& args, const kg_binding& binding)
{
    // Stage2 kg 就绪语义：kg_c,kh[M,K] 完整搬入 L1 后，才允许映射到该 slot 的 head 做 MTE1。
    // ready 已紧跟本 kg slot 的 MTE2 发布，不能在消费点等待全部 MTE2 后才 Set。
    (void)args;
    (void)binding;
}

inline void Stage2PrefetchRightToL1NzArch22(const CubeStage2Args& args,
                                            const HeadBinding& head,
                                            int pipelineSlot)
{
    // Stage2 右操作数预取公式：g-only 搬 V_new_g,c,h，gk-only 搬 V_new_c,h；两者均由 ND 转为 NZ。
    const int rightGmSlot = head.roundHead;
    // Stage1 已经把 UB ND 写入 GM；先等 GM 写出，再异步发起 Stage2 的 ND -> NZ 搬运。
    args.sync->Wait(EventKind::RightGmReady, rightGmSlot, /*S2 MTE2*/ 0);
    if (args.plan->stage0Required) {
        // 当前 chunk 的 H MTE1 末读后，才允许 right MTE2 覆盖同一 L1 slot。
        args.sync->Wait(EventKind::HFree, head.hSlot, /*S2 MTE2*/ 0);
    }
    args.memory->AcquireRightForS2Mte2(head.hSlot);
    Mte2CopyRightOperandGmNdToL1NzAsyncArch22(
        args.workspace->rightOperandGm, rightGmSlot, args.plan->chunk,
        L1Right(args.memory->l1, head.hSlot), args.plan->chunk.validTokens);
    args.memory->MarkRightL1Ready(head.hSlot);
    args.sync->Set(EventKind::RightL1Ready, head.hSlot, /*S2 MTE2*/ 0,
                   /*S2 MTE1*/ 2);
    args.sync->Set(EventKind::S2Mte2ToMte1Ready, pipelineSlot,
                   /*S2 MTE2*/ 0, /*S2 MTE1*/ 2);
    // GM scratch 的最后消费者就是当前 MTE2；free 必须紧跟本 head 的搬运，
    // 不能到消费函数中才发布，否则会错误包含下一个 head 的 MTE2。
    args.memory->ReleaseRightGmAfterS2Mte2(rightGmSlot);
    args.sync->Set(EventKind::RightGmFree, rightGmSlot, /*S2 MTE2*/ 0,
                   /*下一 S1 MTE3*/ 1);
}

inline void Stage2EnsureRightL1ReadyArch22(const CubeStage2Args& args, const HeadBinding& head)
{
    // Stage2 右操作数就绪公式：L1Right_c,h[NZ] = convert_ND_to_NZ(V_new_g/V_new[GM])。
    // ready/free 已由 producer 紧跟当前 head 的 MTE2 发布；这里只保留消费侧模块边界。
    (void)args;
    (void)head;
}

inline void Stage2ComputeDForHeadArch22(const CubeStage2Args& args, const HeadBinding& head,
                                        int pipelineSlot)
{
    // Stage2 计算公式：g-only 为 D_c,h = k_raw_c,kh^T @ V_new_g,c,h；gk-only 为 D_c,h = kg_c,kh^T @ V_new_c,h。
    const kg_binding& binding = args.plan->kg[head.kgSlot];
    Stage2EnsureKgReadyArch22(args, binding);

    // 一个 kg slot 的 ready token 只由第一个映射 head 消费；后续 head 直接复用 valid entry。
    if (head.roundHead == binding.firstConsumerRoundHead) {
        args.sync->Wait(EventKind::kg_ready, binding.slot, /*S2 MTE1*/ 2);
    }
    if (args.plan->stage0Required) {
        // AIC 四槽在本 round 内互不复用；这里只接收当前 chunk Stage0 对同一物理槽的 free。
        args.sync->Wait(EventKind::S0CubeToMte1Free, pipelineSlot, /*S2 MTE1*/ 2);
    }
    // 子模块 Stage2EnsureRightL1Ready：确认预取的 GM ND -> L1 NZ 已完成。
    Stage2EnsureRightL1ReadyArch22(args, head);
    args.sync->Wait(EventKind::S2Mte2ToMte1Ready, pipelineSlot, /*S2 MTE1*/ 2);
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

    Mte1LoadArch22(l1_kg(args.memory->l1, binding.slot), L0AArch22(head));
    Mte1LoadArch22(L1Right(args.memory->l1, head.hSlot), L0BArch22(head));
    args.sync->Set(EventKind::S2Mte1ToCubeReady, pipelineSlot,
                   /*S2 MTE1*/ 2, /*S2 Cube*/ 0);
    args.sync->Wait(EventKind::S2Mte1ToCubeReady, pipelineSlot, /*S2 Cube*/ 0);
    if (args.plan->stage0Required) {
        args.sync->Wait(EventKind::S0FixpipeToCubeFree, pipelineSlot, /*S2 Cube*/ 0);
    }

    // g-only 物理 kg slot 保存 k_raw，右操作数是 V_new_g；gk-only 保存 prepared kg，右操作数是 V_new。
    const auto dAcc = MmadBf16AccFp32Arch22(L0AArch22(head), L0BArch22(head),
                                               args.plan->chunk.validTokens);
    if (args.plan->hasNextChunk) {
        args.sync->Set(EventKind::S2CubeToMte1Free, pipelineSlot,
                       /*S2 Cube*/ 0, /*下一 chunk S0 MTE1*/ 1);
    }
    args.sync->Set(EventKind::S2CubeToFixpipeReady, pipelineSlot,
                   /*S2 Cube*/ 0, /*S2 Fixpipe*/ 3);
    args.sync->Wait(EventKind::S2CubeToFixpipeReady, pipelineSlot, /*S2 Fixpipe*/ 3);
    // A2/A3 没有 L0C -> UB，D 也必须走独立 GM scratch 再由 MTE2 回到 UB。
    wait_arch22_d_gm_slot_free(Arch22DScratch(*args.workspace, head.roundHead));
    FixpipeDNoQuantToGmArch22(dAcc, Arch22DScratch(*args.workspace, head.roundHead),
                              args.plan->chunk);
    args.sync->Set(EventKind::S2FixpipeToMte2Ready, pipelineSlot,
                   /*S2 Fixpipe*/ 3, /*S2 MTE2*/ 0);
    if (args.plan->hasNextChunk) {
        args.sync->Set(EventKind::S2FixpipeToCubeFree, pipelineSlot,
                       /*S2 Fixpipe*/ 3, /*下一 chunk S0 Cube*/ 2);
    }
    args.sync->Wait(EventKind::S2FixpipeToMte2Ready, pipelineSlot, /*S2 MTE2*/ 0);
    Arch22Mte2DFromGmToUb(args, head);
    args.memory->MarkDReady(head);
    args.sync->Set(EventKind::DReady, head.roundHead, /*A2 S2 MTE2 GM->UB*/ 0,
                   /*S3 VF*/ 3);
    args.sync->Release(EventKind::DgmFree, head.roundHead, /*A2 S2 MTE2 GM->UB*/ 0);

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
    PrepareArch22CubeWorkspace(*args.workspace, *args.tiling);

    Stage2LoadKgForRoundArch22(args);
    for (int coreHeadId = 0; coreHeadId < plan.activeHvCount; ++coreHeadId) {
        const CoreHeadBinding current = BindAicCoreHead(plan, coreHeadId);
        Stage2PrefetchRightToL1NzArch22(args, plan.heads[current.roundHead],
                                        current.pipelineSlot);
        if (coreHeadId > 0) {
            const CoreHeadBinding previous = BindAicCoreHead(plan, coreHeadId - 1);
            Stage2ComputeDForHeadArch22(args, plan.heads[previous.roundHead],
                                        previous.pipelineSlot);
        }
        ++result.activeTaskCount;
    }
    if (plan.activeHvCount > 0) {
        const int lastCoreHeadId = plan.activeHvCount - 1;
        const CoreHeadBinding last = BindAicCoreHead(plan, lastCoreHeadId);
        Stage2ComputeDForHeadArch22(args, plan.heads[last.roundHead], last.pipelineSlot);
    }
    result.kgLoadCount = plan.requiredKhCount;
    result.produced = true;
    return result;
}

} // 命名空间 fwd_h_pseudocode

#endif // FLA_FWD_H_PSEUDOCODE_ARCH22_CHUNK_GATED_DELTA_RULE_FWD_H_CUBE_H_
