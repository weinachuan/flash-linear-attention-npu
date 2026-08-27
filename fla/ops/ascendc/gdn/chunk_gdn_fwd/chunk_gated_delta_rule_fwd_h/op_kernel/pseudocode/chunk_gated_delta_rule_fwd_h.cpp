// 仅伪代码。FwdH kernel 入口和 sequence -> round -> chunk 总调度。
// 本文件对应真实 op_kernel 中的 chunk_gated_delta_rule_fwd_h.cpp。
// 伪代码目录不加入任何 CMake 构建目标。

#include "chunk_gated_delta_rule_fwd_h_policy.h"
#include "chunk_gated_delta_rule_fwd_h_utils.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/chunk_gated_delta_rule_fwd_h_cube.h"
#include "arch35/chunk_gated_delta_rule_fwd_h_vec.h"
#else
#include "arch22/chunk_gated_delta_rule_fwd_h_cube.h"
#include "arch22/chunk_gated_delta_rule_fwd_h_vec.h"
#endif

namespace fwd_h_pseudocode {

inline CubeStageResult RunStage0ByArch(const CubeStage0Args& args)
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    return RunStage0Arch35(args);
#else
    return RunStage0Arch22(args);
#endif
}

inline VecStageResult RunStage1ByArch(const VecStageArgs& args)
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    return RunStage1Arch35(args);
#else
    return RunStage1Arch22(args);
#endif
}

inline CubeStageResult RunStage2ByArch(const CubeStage2Args& args)
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    return RunStage2Arch35(args);
#else
    return RunStage2Arch22(args);
#endif
}

inline VecStageResult RunStage3ByArch(const VecStage3Args& args)
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    return RunStage3Arch35(args);
#else
    return RunStage3Arch22(args);
#endif
}

inline void RunSMinusOneByArch(const SMinusOneArgs& args)
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    RunSMinusOneArch35(args);
#else
    RunSMinusOneArch22(args);
#endif
}

inline void RunOneChunk(SchedulerContext& ctx, const RoundPlan& plan)
{
    const bool finalVNewOnly = plan.finalVNewOnly;
    const bool noInitialFirst = plan.chunk.first && !ctx.tiling.useInitialState;

    if (FwdHStagePolicy::NeedStage0(plan)) {
        // S0 内部已完成 kg 异步预取、H/W MTE2、MTE1、MMAD、P Fixpipe，并发布 PReady。
        RunStage0ByArch({&ctx.inputs, &ctx.outputs, &ctx.tiling, &plan, &ctx.memory, &ctx.sync});
    }

    Stage1Variant variant = Stage1Variant::WithP;
    if (finalVNewOnly) {
        variant = Stage1Variant::v_new_only;
    } else if (noInitialFirst) {
        variant = Stage1Variant::NoP;
    }
    // S1 等待 PReady（如有），一次 VF 生成 V_new/右操作数，并在 MTE3 完成后发布 RightReady。
    RunStage1ByArch({&ctx.inputs, &ctx.outputs, &ctx.tiling, &plan,
                     &ctx.memory, &ctx.sync, variant});

    if (!FwdHStagePolicy::NeedStage2(plan)) {
        // final_state 未请求时，最终 chunk 不加载 kg、不执行 S2/S3，也不写无消费者的 L1 右操作数。
        return;
    }
    // S2 等待 kg_ready/RightReady，完成每个 head 的 MMAD/D Fixpipe，并按最后消费者释放 kg/right。
    RunStage2ByArch({&ctx.inputs, &ctx.tiling, &plan, &ctx.memory, &ctx.sync});
    // S3 等待 DReady，更新 rolling state；非末 chunk 发布 HReady，末 chunk 按需写 final_state。
    RunStage3ByArch({&ctx.inputs, &ctx.outputs, &ctx.tiling, &plan, &ctx.memory, &ctx.sync});
}

inline void RunFwdH(SchedulerContext& ctx)
{
    // Host 校验已经建立全部 shape/dtype/layout 不变量。
    for (int n = 0; n < ctx.tiling.sequenceCount; ++n) {
        const auto& seq = ctx.tiling.sequences[n];
        // 循环顺序固定为：sequence -> head_round -> chunk。
        for (int round = 0; round * kMaxRoundHeads < ctx.tiling.hv; ++round) {
            // 先固定本 head_round 的 requiredKhCount 和每个 H_v -> kgSlot 映射。
            // 这些关系只依赖 HK/HV，与 chunk 无关；后续 chunk 只绑定 token payload。
            const RoundPlan headRoundPlan = BuildHeadRoundPlan(ctx.tiling, round);
            if (round > 0) {
                // 上一 round 的 kg、H、W 及异步搬运全部排空后，才允许本 round 预取。
                const RoundPlan previousHeadRound = BuildHeadRoundPlan(ctx.tiling, round - 1);
                const RoundPlan previousRound =
                    BuildChunkPlan(previousHeadRound, ctx.tiling, seq, seq.chunkCount - 1);
                ctx.sync.WaitBeforeNextRound(previousRound);
                ctx.memory.ReleaseStateAfterRoundBarrier(previousRound);
            }

            // S-1 是当前 round 的生产者，必须在屏障后执行，并在首个 S0 前排空。
            RunSMinusOneByArch({&ctx.inputs, &ctx.outputs, &ctx.tiling, &seq,
                                &ctx.memory, &ctx.sync, headRoundPlan.activeHvBegin,
                                headRoundPlan.activeHvCount, round > 0});

            for (int c = 0; c < seq.chunkCount; ++c) {
                // chunk 只绑定 token 范围和最终 chunk 分支，不重新计算 required_hk_round。
                const RoundPlan plan = BuildChunkPlan(headRoundPlan, ctx.tiling, seq, c);
                RunOneChunk(ctx, plan);
            }
            ctx.sync.Set(EventKind::TerminalDrain, round, /*round 生产者*/ 0, /*调度器*/ -1);
        }
    }
}

// 框架 tensor 到 ApiInputs 的适配是占位逻辑；实际实现沿用 fast-launch 的输入封装。
inline ApiInputs MakeApiInputs(PseudocodeTensor k, PseudocodeTensor w, PseudocodeTensor u,
                               OptionalPseudocodeTensor g, OptionalPseudocodeTensor gk,
                               OptionalPseudocodeTensor initialState, bool outputFinalState,
                               int64_t chunkSize, bool saveNewValue, OptionalIntArray cuSeqlens,
                               OptionalIntArray chunkIndices, bool useExp2, bool stateVFirst)
{
    (void)k;
    (void)w;
    (void)u;
    (void)g;
    (void)gk;
    (void)initialState;
    (void)outputFinalState;
    (void)chunkSize;
    (void)saveNewValue;
    (void)cuSeqlens;
    (void)chunkIndices;
    (void)useExp2;
    (void)stateVFirst;
    return {};
}

inline PseudocodeTensorTuple MakeParameterError(const char* error)
{
    (void)error;
    PseudocodeTensorTuple result{};
    result.parameterError = true;
    return result;
}

inline PseudocodeTensorTuple MakeOutputs(const ApiOutputs& outputs)
{
    PseudocodeTensorTuple result{};
    result.outputs = outputs;
    return result;
}

// 入口参数形状参考真实 fast-launch 入口；这里保留为可直接迁移的伪代码。
inline PseudocodeTensorTuple chunk_gated_delta_rule_fwd_h(
    PseudocodeTensor k,
    PseudocodeTensor w,
    PseudocodeTensor u,
    OptionalPseudocodeTensor g,
    OptionalPseudocodeTensor gk,
    OptionalPseudocodeTensor initial_state,
    bool output_final_state,
    int64_t chunk_size,
    bool save_new_value,
    OptionalIntArray cu_seqlens,
    OptionalIntArray chunk_indices,
    bool use_exp2,
    bool state_v_first)
{
    SchedulerContext ctx{};
    ctx.inputs = MakeApiInputs(k, w, u, g, gk, initial_state, output_final_state,
                               chunk_size, save_new_value, cu_seqlens, chunk_indices,
                               use_exp2, state_v_first);

    const auto host = ValidateAndBuildTiling(ctx.inputs);
    if (!host.ok) {
        return MakeParameterError(host.error);
    }
    ctx.tiling = host.tiling;
    ctx.outputs = host.outputs;
    RunFwdH(ctx);
    return MakeOutputs(ctx.outputs);
}

} // 命名空间 fwd_h_pseudocode
