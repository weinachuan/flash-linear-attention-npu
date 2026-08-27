// PSEUDOCODE ONLY. Stage0: load H/W and compute P = W @ H.

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
        // No initial state on the first chunk: do not issue W @ 0 and do not create P.
        return result;
    }

    // Stage0 is allowed to launch the current chunk's kg prefetch asynchronously. It does
    // not consume kg; kg_ready remains live through Stage2's last MTE1 read.
    std::array<bool, kMaxKeySlots> kgPrefetchLaunched{};
    for (int i = 0; i < plan.requiredKhCount && plan.stage2Required; ++i) {
        const auto& kg = plan.kg[i];
        if (args.memory->kg[kg.slot].generation > 0) {
            args.sync->Wait(EventKind::kg_overwrite_safe, kg.slot, /*S0 kg producer*/ 0);
        }
        args.memory->acquire_kg(kg.slot);
        launch_mte2_kg_async(
            /*src=*/KeyPayload(*args.in, plan.chunk, kg.kh, kg.payload),
            /*dst=*/l1_kg(args.memory->l1, kg.slot), plan.chunk.validTokens);
        kgPrefetchLaunched[kg.slot] = true;
    }

    for (int i = 0; i < plan.activeHvCount; ++i) {
        const HeadBinding& head = plan.heads[i];
        const bool bf16Initial = plan.chunk.first && args.tiling->useInitialState &&
                                 args.tiling->stateType == StateType::Bf16;
        if (bf16Initial) {
            args.sync->Wait(EventKind::HFree, head.hSlot, /*S0 initial producer*/ 0);
            args.memory->AcquireHForS0(head.hSlot);
            // BF16 initial_state is loaded directly by this S0 into canonical L1 [K,V].
            Mte2StateToCanonicalH(*args.in, head.hv, args.tiling->stateLayout,
                                  L1H(args.memory->l1, head.hSlot));
            args.memory->MarkHReady(head.hSlot);
        } else {
            // FP32 initial_state comes from S-1; later chunks come from S3.
            args.sync->Wait(EventKind::HReady, head.hSlot, /*S0*/ 0);
            args.memory->BeginHReadFromS3(head.hSlot);
        }

        args.sync->Wait(EventKind::WFree, head.wSlot, /*S0*/ 0);
        args.memory->AcquireWForS0(head.wSlot);
        Mte2CopyW(*args.in, plan.chunk, head.hv, L1W(args.memory->l1, head.wSlot));
        args.memory->MarkWReady(head.wSlot);

        // MTE1 reads only this Stage's W/H operands. It never reads P/D produced here.
        args.sync->Wait(EventKind::WReady, head.wSlot, /*MTE1*/ 1);
        args.sync->Wait(EventKind::HReady, head.hSlot, /*MTE1*/ 1);
        Mte1Load(L1W(args.memory->l1, head.wSlot), L0A(head));
        Mte1Load(L1H(args.memory->l1, head.hSlot), L0B(head));
        const auto pAcc = MmadBf16AccFp32(L0A(head), L0B(head), plan.chunk.validTokens);
        FixpipeP(pAcc, UbP(*args.memory, head));
        args.sync->Set(EventKind::PReady, head.roundHead, /*S0*/ 0, /*S1*/ 1);
        args.memory->ReleaseWAfterS0Mte1(head.wSlot);
        args.memory->ReleaseHAfterS0Mte1(head.hSlot);
        args.sync->Release(EventKind::WFree, head.wSlot, /*S0*/ 0);
        args.sync->Release(EventKind::HFree, head.hSlot, /*S0*/ 0);
    }
    for (int i = 0; i < plan.requiredKhCount && plan.stage2Required; ++i) {
        const auto& kg = plan.kg[i];
        if (kgPrefetchLaunched[kg.slot]) {
            wait_mte2_kg_done(kg.slot);
            args.memory->mark_kg_ready(kg.slot);
            args.sync->Set(EventKind::kg_ready, kg.slot, /*MTE2*/ 0, /*S2*/ 2);
        }
    }
    result.producedP = true;
    return result;
}

} // namespace fwd_h_pseudocode
