// 仅伪代码。Stage0：加载 H/W 并计算 P = W @ H。

#pragma once

#include "fwd_h_pseudocode_memory.h"
#include "fwd_h_pseudocode_round_planner.h"
#include "fwd_h_pseudocode_sync.h"

namespace fwd_h_pseudocode {

struct Stage0Args {
    const ApiInputs* in = nullptr;
    const ApiOutputs* out = nullptr;
    const TilingPlan* tiling = nullptr;
    const RoundPlan* plan = nullptr;
    FixedMemory* memory = nullptr;
    SyncLedger* sync = nullptr;
};

struct Stage0Result {
    bool producedP = false;
};

inline Stage0Result RunStage0(const Stage0Args& args)
{
    Stage0Result result{};
    const RoundPlan& plan = *args.plan;
    if (!plan.stage0Required) {
        // 第一个 chunk 没有 initial state：不执行 W @ 0，也不生成 P。
        return result;
    }

    // Stage0 可以异步发起当前 chunk 的 kg 预取，但不消费 kg；
    // kg_ready 必须一直保持有效，直到 Stage2 的最后一次 MTE1 读取完成。
    std::array<bool, kMaxKeySlots> kgPrefetchLaunched{};
    for (int i = 0; i < plan.requiredKhCount && plan.stage2Required; ++i) {
        const auto& kg = plan.kg[i];
        if (args.memory->kg[kg.slot].generation > 0) {
            args.sync->Wait(EventKind::kg_overwrite_safe, kg.slot, /*S0 的 kg 生产者*/ 0);
        }
        args.memory->acquire_kg(kg.slot);
        launch_mte2_kg_async(
            /*源=*/KeyPayload(*args.in, plan.chunk, kg.kh, kg.payload),
            /*目的=*/l1_kg(args.memory->l1, kg.slot), plan.chunk.validTokens);
        kgPrefetchLaunched[kg.slot] = true;
    }

    for (int i = 0; i < plan.activeHvCount; ++i) {
        const HeadBinding& head = plan.heads[i];
        const bool bf16Initial = plan.chunk.first && args.tiling->useInitialState &&
                                 args.tiling->stateType == StateType::Bf16;
        if (bf16Initial) {
            args.sync->Wait(EventKind::HFree, head.hSlot, /*S0 的初态生产者*/ 0);
            args.memory->AcquireHForS0(head.hSlot);
            // BF16 initial_state 由本 S0 直接加载到规范的 L1 [K,V]。
            Mte2StateToCanonicalH(*args.in, head.hv, args.tiling->stateLayout,
                                  L1H(args.memory->l1, head.hSlot));
            args.memory->MarkHReady(head.hSlot);
        } else {
            // FP32 initial_state 来自 S-1；后续 chunk 来自 S3。
            args.sync->Wait(EventKind::HReady, head.hSlot, /*S0 消费者*/ 0);
            args.memory->BeginHReadFromS3(head.hSlot);
        }

        args.sync->Wait(EventKind::WFree, head.wSlot, /*S0 消费者*/ 0);
        args.memory->AcquireWForS0(head.wSlot);
        Mte2CopyW(*args.in, plan.chunk, head.hv, L1W(args.memory->l1, head.wSlot));
        args.memory->MarkWReady(head.wSlot);

        // MTE1 只读取本 Stage 的 W/H 操作数，不读取本 Stage 生成的 P/D。
        args.sync->Wait(EventKind::WReady, head.wSlot, /*MTE1 消费者*/ 1);
        args.sync->Wait(EventKind::HReady, head.hSlot, /*MTE1 消费者*/ 1);
        Mte1Load(L1W(args.memory->l1, head.wSlot), L0A(head));
        Mte1Load(L1H(args.memory->l1, head.hSlot), L0B(head));
        const auto pAcc = MmadBf16AccFp32(L0A(head), L0B(head), plan.chunk.validTokens);
        FixpipeP(pAcc, UbP(*args.memory, head));
        args.sync->Set(EventKind::PReady, head.roundHead, /*S0 生产者*/ 0, /*S1 消费者*/ 1);
        args.memory->ReleaseWAfterS0Mte1(head.wSlot);
        args.memory->ReleaseHAfterS0Mte1(head.hSlot);
        args.sync->Release(EventKind::WFree, head.wSlot, /*S0 消费者*/ 0);
        args.sync->Release(EventKind::HFree, head.hSlot, /*S0 消费者*/ 0);
    }
    for (int i = 0; i < plan.requiredKhCount && plan.stage2Required; ++i) {
        const auto& kg = plan.kg[i];
        if (kgPrefetchLaunched[kg.slot]) {
            wait_mte2_kg_done(kg.slot);
            args.memory->mark_kg_ready(kg.slot);
            args.sync->Set(EventKind::kg_ready, kg.slot, /*MTE2 生产者*/ 0, /*S2 消费者*/ 2);
        }
    }
    result.producedP = true;
    return result;
}

} // 命名空间 fwd_h_pseudocode
