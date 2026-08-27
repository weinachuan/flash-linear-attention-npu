// 仅伪代码。sequence -> head_round -> chunk 调度，以及跨 round 生命周期管理。

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
        // 未请求 final_state 时，不预取 kg、不执行 S2 MMAD，也不在 S3 落盘 state。
        return;
    }
    RunStage2({&ctx.inputs, &ctx.tiling, &plan, &ctx.memory, &ctx.sync});
    RunStage3({&ctx.inputs, &ctx.outputs, &ctx.tiling, &plan, &ctx.memory, &ctx.sync});
}

inline void RunFwdH(SchedulerContext& ctx)
{
    // Host 校验已经建立全部 shape/dtype/layout 不变量。
    for (int n = 0; n < ctx.tiling.sequenceCount; ++n) {
        const auto& seq = ctx.tiling.sequences[n];
        // 循环顺序固定为：sequence -> head_round -> chunk。
        for (int round = 0; round * kMaxRoundHeads < ctx.tiling.hv; ++round) {
            RoundPlan previousRound{};
            bool hasPreviousRound = round > 0;
            if (hasPreviousRound) {
                // 必须等待前一个 round 的所有 kg/H/W slot 和异步搬运完成，
                // 本 round 才允许预取任何 H/W/kg。
                previousRound = BuildRoundPlan(ctx.tiling, n, round - 1, seq.chunkCount - 1);
                ctx.sync.WaitBeforeNextRound(previousRound);
            }

            const int activeBegin = round * kMaxRoundHeads;
            const int activeCount = static_cast<int>(ctx.tiling.hv) - activeBegin > kMaxRoundHeads
                                        ? kMaxRoundHeads
                                        : static_cast<int>(ctx.tiling.hv) - activeBegin;
            // S-1 是当前 round 的生产者，必须在上一 round 屏障之后执行，
            // 并在本 round 的第一个 S0 消费 H 之前完成排空。
            RunSMinusOne({&ctx.inputs, &ctx.outputs, &ctx.tiling, &seq, &ctx.memory, &ctx.sync,
                          activeBegin, activeCount});

            for (int c = 0; c < seq.chunkCount; ++c) {
                // kg slot 的 payload 由 (chunk, kh) 标识，只属于当前 round。
                // 如果 slot 被前一个 chunk 使用过，Stage0/Stage2 必须先等待覆盖安全事件，
                // 再复用固定的 16 KiB 地址。
                const RoundPlan plan = BuildRoundPlan(ctx.tiling, n, round, c);
                RunOneChunk(ctx, plan);
            }
            ctx.sync.Set(EventKind::TerminalDrain, round, /*round 生产者*/ 0, /*调度器*/ -1);
        }
    }
}

} // 命名空间 fwd_h_pseudocode
