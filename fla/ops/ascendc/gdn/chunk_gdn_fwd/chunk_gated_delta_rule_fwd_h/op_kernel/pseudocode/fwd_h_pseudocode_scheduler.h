// PSEUDOCODE ONLY. Sequence -> head_round -> chunk scheduler and cross-round lifetime.

#pragma once

#include "fwd_h_pseudocode_host.h"
#include "fwd_h_pseudocode_s_minus_one.h"
#include "fwd_h_pseudocode_stage3.h"

namespace fwd_h_pseudocode {

struct SchedulerContext {
    ApiInputs inputs{};
    ApiOutputs outputs{};
    TilingPlan tiling{};
    FixedMemory memory{};
    SyncLedger sync{};
};

inline void RunOneChunk(SchedulerContext& ctx, const RoundPlan& plan)
{
    const bool final_v_new_only = plan.chunk.last && !ctx.tiling.outputFinalState;
    const bool noInitialFirst = plan.chunk.first && !ctx.tiling.useInitialState;

    if (plan.stage0Required) {
        RunStage0({&ctx.inputs, &ctx.outputs, &ctx.tiling, &plan, &ctx.memory, &ctx.sync});
    }

    Stage1Variant variant = Stage1Variant::WithP;
    if (final_v_new_only) {
        variant = Stage1Variant::v_new_only;
    } else if (noInitialFirst) {
        variant = Stage1Variant::NoP;
    }
    RunStage1({&ctx.inputs, &ctx.outputs, &ctx.tiling, &plan, &ctx.memory, &ctx.sync, variant});

    if (!plan.stage2Required) {
        // No kg prefetch, S2 MMAD, or S3 state materialization when final_state is not requested.
        return;
    }
    RunStage2({&ctx.inputs, &ctx.tiling, &plan, &ctx.memory, &ctx.sync});
    RunStage3({&ctx.inputs, &ctx.outputs, &ctx.tiling, &plan, &ctx.memory, &ctx.sync});
}

inline void RunFwdH(SchedulerContext& ctx)
{
    // Host validation has already established all shape/dtype/layout invariants.
    for (int n = 0; n < ctx.tiling.sequenceCount; ++n) {
        const auto& seq = ctx.tiling.sequences[n];
        // The loop order is mandatory: sequence -> head_round -> chunk.
        for (int round = 0; round * kMaxRoundHeads < ctx.tiling.hv; ++round) {
            RoundPlan previousRound{};
            bool hasPreviousRound = round > 0;
            if (hasPreviousRound) {
                // Wait for every kg/H/W slot and all asynchronous transfers from the previous
                // round before this round is allowed to prefetch any H/W/kg.
                previousRound = BuildRoundPlan(ctx.tiling, n, round - 1, seq.chunkCount - 1);
                ctx.sync.WaitBeforeNextRound(previousRound);
            }

            const int activeBegin = round * kMaxRoundHeads;
            const int activeCount = static_cast<int>(ctx.tiling.hv) - activeBegin > kMaxRoundHeads
                                        ? kMaxRoundHeads
                                        : static_cast<int>(ctx.tiling.hv) - activeBegin;
            // S-1 is a current-round producer. It must be after the previous-round barrier,
            // and it must drain before this round's first S0 consumes H.
            RunSMinusOne({&ctx.inputs, &ctx.outputs, &ctx.tiling, &seq, &ctx.memory, &ctx.sync,
                          activeBegin, activeCount});

            for (int c = 0; c < seq.chunkCount; ++c) {
                // kg slot payload is keyed by (chunk, kh) and is current-round only. If a
                // slot was used by the previous chunk, Stage0/Stage2 waits its overwrite-safe
                // event before reusing the fixed 16 KiB address.
                const RoundPlan plan = BuildRoundPlan(ctx.tiling, n, round, c);
                RunOneChunk(ctx, plan);
            }
            ctx.sync.Set(EventKind::TerminalDrain, round, /*round producer*/ 0, /*scheduler*/ -1);
        }
    }
}

} // namespace fwd_h_pseudocode
