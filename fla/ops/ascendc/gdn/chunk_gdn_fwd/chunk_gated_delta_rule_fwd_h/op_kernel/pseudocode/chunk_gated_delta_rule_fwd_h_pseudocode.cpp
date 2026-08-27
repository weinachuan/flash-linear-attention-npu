// PSEUDOCODE ONLY. Fast-launch-shaped entry for the new FwdH design.
// This source is intentionally excluded from every CMake target.

#include "fwd_h_pseudocode_scheduler.h"

namespace ascend_ops {
namespace ChunkGatedDeltaRuleFwdH {

// The signature follows the PR370 fast-launch entry shape, with the current design's
// g/gk exclusive gate inputs and native state_v_first attribute.
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
    // 1. Adapt framework tensors into ApiInputs without changing state layout.
    fwd_h_pseudocode::SchedulerContext ctx{};
    ctx.inputs = MakeApiInputs(k, w, u, g, gk, initial_state, output_final_state,
                               chunk_size, save_new_value, cu_seqlens, chunk_indices,
                               use_exp2, state_v_first);

    // 2. Host-side validation/inference creates tiling and output descriptors.
    const auto host = fwd_h_pseudocode::ValidateAndBuildTiling(ctx.inputs);
    if (!host.ok) {
        return MakeParameterError(host.error);
    }
    ctx.tiling = host.tiling;
    ctx.outputs = host.outputs;

    // 3. Kernel launch is structurally equivalent to the existing fast entry, but the
    // implementation called here is the new sequence/round/chunk pseudocode scheduler.
    fwd_h_pseudocode::RunFwdH(ctx);
    return MakeOutputs(ctx.outputs);
}

} // namespace ChunkGatedDeltaRuleFwdH
} // namespace ascend_ops
