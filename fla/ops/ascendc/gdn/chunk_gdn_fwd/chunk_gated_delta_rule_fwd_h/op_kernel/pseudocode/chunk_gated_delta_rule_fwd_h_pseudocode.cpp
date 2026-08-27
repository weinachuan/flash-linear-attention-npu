// 仅伪代码。新的 FwdH 设计的 fast-launch 形状入口。
// 本文件明确不加入任何 CMake 构建目标。

#include "fwd_h_pseudocode_scheduler.h"

namespace ascend_ops {
namespace ChunkGatedDeltaRuleFwdH {

// 函数签名沿用 PR370 的 fast-launch 入口形状，但门控输入遵循当前设计的
// g/gk 互斥约束，并由 kernel 原生接收 state_v_first 属性。
fwd_h_pseudocode::PseudocodeTensorTuple chunk_gated_delta_rule_fwd_h_pseudocode(
    fwd_h_pseudocode::PseudocodeTensor k,
    fwd_h_pseudocode::PseudocodeTensor w,
    fwd_h_pseudocode::PseudocodeTensor u,
    fwd_h_pseudocode::OptionalPseudocodeTensor g,
    fwd_h_pseudocode::OptionalPseudocodeTensor gk,
    fwd_h_pseudocode::OptionalPseudocodeTensor initial_state,
    bool output_final_state,
    int64_t chunk_size,
    bool save_new_value,
    fwd_h_pseudocode::OptionalIntArray cu_seqlens,
    fwd_h_pseudocode::OptionalIntArray chunk_indices,
    bool use_exp2,
    bool state_v_first)
{
    // 1. 将框架 tensor 适配为 ApiInputs，不改变 state 的物理布局。
    fwd_h_pseudocode::SchedulerContext ctx{};
    ctx.inputs = MakeApiInputs(k, w, u, g, gk, initial_state, output_final_state,
                               chunk_size, save_new_value, cu_seqlens, chunk_indices,
                               use_exp2, state_v_first);

    // 2. Host 侧完成参数校验和推导，生成 tiling 与输出描述符。
    const auto host = fwd_h_pseudocode::ValidateAndBuildTiling(ctx.inputs);
    if (!host.ok) {
        return MakeParameterError(host.error);
    }
    ctx.tiling = host.tiling;
    ctx.outputs = host.outputs;

    // 3. kernel launch 的结构与现有 fast 入口一致，但这里调用的是新的
    //    sequence/round/chunk 伪代码调度器。
    fwd_h_pseudocode::RunFwdH(ctx);
    return MakeOutputs(ctx.outputs);
}

} // 命名空间 ChunkGatedDeltaRuleFwdH
} // 命名空间 ascend_ops
