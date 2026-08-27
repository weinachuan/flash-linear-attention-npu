// PSEUDOCODE ONLY. S-1: FP32 initial_state -> BF16 H0 Cube shadow.

#pragma once

#include "fwd_h_pseudocode_memory.h"
#include "fwd_h_pseudocode_sync.h"

namespace fwd_h_pseudocode {

struct SMinusOneArgs {
    const ApiInputs* in = nullptr;
    const ApiOutputs* out = nullptr;
    const TilingPlan* tiling = nullptr;
    const SequenceSpan* sequence = nullptr;
    FixedMemory* memory = nullptr;
    SyncLedger* sync = nullptr;
    int activeHvBegin = 0;
    int activeHvCount = 0;
};

inline void RunSMinusOne(const SMinusOneArgs& args)
{
    if (!args.tiling->useInitialState || args.tiling->stateType != StateType::Fp32) {
        return;
    }
    for (int local = 0; local < args.activeHvCount; ++local) {
        const int hv = args.activeHvBegin + local;
        const int hSlot = local;
        args.memory->ProduceInitialH(hSlot);

        // The only GM state layout conversion is this layout-aware read. The VF output
        // is canonical BF16 [K,V] for S0 and is also written to logical h0.
        const auto state = Mte2ReadStateLayoutAware(
            args.in->initialState, args.sequence->sequence, hv, args.tiling->stateLayout);
        const auto h0 = CastBf16(VfCastFp32ToBf16(state));
        Mte3WriteH0LayoutAware(*args.out, *args.tiling, *args.sequence, hv, h0);
        Mte3WriteL1Resident(L1H(args.memory->l1, hSlot), h0);
        args.memory->MarkHReady(hSlot);
        PublishInitialHReady(*args.sync, hSlot, hv);
    }
    WaitInitialPhaseDrain(*args.sync, args.sequence->sequence);
}

} // namespace fwd_h_pseudocode
