// PSEUDOCODE ONLY. Stage3: update rolling state and layout-aware outputs.

#pragma once

#include "fwd_h_pseudocode_stage2.h"

namespace fwd_h_pseudocode {

struct Stage3Args {
    const ApiInputs* in = nullptr;
    const ApiOutputs* out = nullptr;
    const TilingPlan* tiling = nullptr;
    const RoundPlan* plan = nullptr;
    FixedMemory* memory = nullptr;
    SyncLedger* sync = nullptr;
};

struct Stage3Result {
    bool nextHReady = false;
    bool finalStateWritten = false;
};

inline Stage3Result RunStage3(const Stage3Args& args)
{
    Stage3Result result{};
    const RoundPlan& plan = *args.plan;
    if (!plan.stage3Required) {
        return result;
    }

    for (int i = 0; i < plan.activeHvCount; ++i) {
        const HeadBinding& head = plan.heads[i];
        args.sync->Wait(EventKind::DReady, head.roundHead, /*S3*/ 3);

        const auto r = LoadRollingStateCanonical(*args.in, *args.out, *args.tiling, plan, head);
        auto rNextFp32 = ToFp32(r);
        if (plan.gateMode == GateMode::ScalarG) {
            const float alpha = plan.chunk.first && !args.tiling->useInitialState
                                    ? 1.0f
                                    : LoadAlpha(UbAlpha(*args.memory, head));
            rNextFp32 = VfMul(rNextFp32, alpha);
        } else {
            const auto gkLast = LoadGkLast(*args.in, plan.chunk, head.hv);
            const auto rowGate = GateExp(gkLast, args.tiling->useExp2);
            rNextFp32 = VfRowMul(rNextFp32, rowGate);
        }
        rNextFp32 = VfAdd(rNextFp32, UbD(*args.memory, head));

        const auto rNext = CastState(rNextFp32, args.tiling->stateType);
        const auto hNext = CastBf16(rNext);
        if (!plan.chunk.last) {
            // One VF produces canonical [K,V], then both GM h and L1 resident use it.
            args.memory->ProduceHForS3(head.hSlot);
            Mte3WriteHLayoutAware(*args.out, *args.tiling, plan, head, hNext);
            Mte3WriteL1Resident(L1H(args.memory->l1, head.hSlot), hNext);
            args.memory->MarkHReady(head.hSlot);
            args.sync->Set(EventKind::HReady, head.hSlot, /*S3*/ 3, /*next S0*/ 0);
            result.nextHReady = true;
        }
        if (plan.chunk.last && args.tiling->outputFinalState) {
            Mte3WriteFinalStateLayoutAware(*args.out, *args.tiling, plan, head, rNext);
            result.finalStateWritten = true;
        }

        // FP32 rolling state uses the explicit GM fallback only when a later S3 or the
        // public final_state consumes it. No state-ready event is created for a dead value.
        StoreOrReleaseRollingState(*args.in, *args.out, *args.tiling, plan, head, rNext);
        args.sync->Release(EventKind::DReady, head.roundHead, /*S3*/ 3);
    }
    return result;
}

} // namespace fwd_h_pseudocode
