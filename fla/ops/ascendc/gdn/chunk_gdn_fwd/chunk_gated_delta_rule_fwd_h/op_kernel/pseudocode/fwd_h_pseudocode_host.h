// PSEUDOCODE ONLY. Host validation and tiling contract.

#pragma once

#include "fwd_h_pseudocode_types.h"

namespace fwd_h_pseudocode {

struct HostResult {
    bool ok = false;
    TilingPlan tiling{};
    ApiOutputs outputs{};
    const char* error = nullptr;
};

inline StateType InferStateType(const ApiInputs& in)
{
    if (in.initialState.present) {
        return in.initialState.dtype;
    }
    if (in.outputFinalState && in.finalState.present) {
        return in.finalState.dtype;
    }
    return StateType::Fp32;
}

inline bool ValidateStateShape(const TensorRef& state, bool stateVFirst)
{
    if (!state.present) {
        return true;
    }
    const int64_t last0 = stateVFirst ? kValueDim : kKeyDim;
    const int64_t last1 = stateVFirst ? kKeyDim : kValueDim;
    return state.shape.d2 == last0 && state.shape.d3 == last1;
}

inline HostResult ValidateAndBuildTiling(const ApiInputs& in)
{
    HostResult result{};

    // Presence checks happen before any dtype inference or GM dereference.
    const bool hasG = in.g.present;
    const bool hasGk = in.gk.present;
    if (hasG == hasGk) {
        result.error = "exactly one of g and gk must be present";
        return result;
    }
    if (!in.k.present || !in.w.present || !in.u.present) {
        result.error = "k, w and u are required";
        return result;
    }
    if (!in.saveNewValue || in.chunkSize != kBatchTokens) {
        result.error = "only save_new_value=true and chunk_size=64 are supported";
        return result;
    }
    if (in.outputFinalState != in.finalState.present) {
        result.error = "output_final_state must match final_state presence";
        return result;
    }
    if (!ValidateStateShape(in.initialState, in.stateVFirst) ||
        !ValidateStateShape(in.finalState, in.stateVFirst)) {
        result.error = "state shape does not match state_v_first";
        return result;
    }
    if (in.initialState.present && in.finalState.present &&
        in.initialState.dtype != in.finalState.dtype) {
        result.error = "initial_state and final_state must have the same dtype";
        return result;
    }
    if (in.k.dtype != StateType::Bf16 || in.w.dtype != StateType::Bf16 ||
        in.u.dtype != StateType::Bf16) {
        result.error = "k, w and u must be BF16";
        return result;
    }
    if (in.k.shape.d3 != kKeyDim || in.u.shape.d3 != kValueDim ||
        in.w.shape.d3 != kKeyDim) {
        result.error = "K and V must both be 128";
        return result;
    }

    const int64_t hk = in.k.shape.d1;
    const int64_t hv = in.u.shape.d1;
    if (in.w.shape.d1 != hv || in.g.present && in.g.shape.d1 != hv ||
        in.gk.present && in.gk.shape.d1 != hv) {
        result.error = "w, gate and u value-head dimensions must match";
        return result;
    }
    if (hasG && (hv <= 0 || hk <= 0 || hv % hk != 0)) {
        result.error = "g-only requires HV % HK == 0";
        return result;
    }
    if (hasGk && hk != hv) {
        result.error = "gk-only k is prepared kg and must have HV heads";
        return result;
    }

    const bool hasCu = in.cuSeqlens != nullptr;
    const bool hasIndices = in.chunkIndices != nullptr;
    if (hasCu != hasIndices) {
        result.error = "cu_seqlens and chunk_indices must be provided together";
        return result;
    }

    result.tiling.hk = hk;
    result.tiling.hv = hv;
    result.tiling.seqlen = in.k.shape.d2;
    result.tiling.batch = in.k.shape.d0;
    result.tiling.varlen = hasCu;
    result.tiling.stateType = InferStateType(in);
    result.tiling.gateMode = hasG ? GateMode::ScalarG : GateMode::KeyWiseGk;
    result.tiling.stateLayout = in.stateVFirst ? StateLayout::VK : StateLayout::KV;
    result.tiling.outputFinalState = in.outputFinalState;
    result.tiling.useInitialState = in.initialState.present;
    result.tiling.useExp2 = in.useExp2;

    // Build sequence spans. The real host implementation also validates chunk_indices
    // are exactly the compact [sequence, chunk] order implied by these spans.
    if (!hasCu) {
        result.tiling.sequenceCount = in.k.shape.d0;
        for (int n = 0; n < result.tiling.sequenceCount; ++n) {
            auto& seq = result.tiling.sequences[n];
            seq.sequence = n;
            seq.tokenBegin = 0;
            seq.tokenEnd = in.k.shape.d2;
            seq.length = in.k.shape.d2;
            seq.chunkCount = static_cast<int>((seq.length + kBatchTokens - 1) / kBatchTokens);
            seq.chunkPrefix = result.tiling.totalChunks;
            result.tiling.totalChunks += seq.chunkCount;
        }
    } else {
        if (in.k.shape.d0 != 1 || in.cuSeqlensLength < 2) {
            result.error = "varlen requires B=1 and at least [0,T]";
            return result;
        }
        result.tiling.sequenceCount = in.cuSeqlensLength - 1;
        for (int n = 0; n < result.tiling.sequenceCount; ++n) {
            const int64_t begin = in.cuSeqlens[n];
            const int64_t end = in.cuSeqlens[n + 1];
            if (end <= begin) {
                result.error = "varlen sequences must be non-empty";
                return result;
            }
            auto& seq = result.tiling.sequences[n];
            seq.sequence = n;
            seq.tokenBegin = begin;
            seq.tokenEnd = end;
            seq.length = end - begin;
            seq.chunkCount = static_cast<int>((seq.length + kBatchTokens - 1) / kBatchTokens);
            seq.chunkPrefix = result.tiling.totalChunks;
            result.tiling.totalChunks += seq.chunkCount;
        }
    }

    // Output allocation is shown symbolically; the real meta function supplies tensors.
    result.outputs.h.shape = {in.k.shape.d0, hv, result.tiling.totalChunks,
                              in.stateVFirst ? kValueDim : kKeyDim,
                              in.stateVFirst ? kKeyDim : kValueDim};
    result.outputs.v_new.shape = {in.k.shape.d0, hv, in.k.shape.d2, kValueDim, 0};
    if (in.outputFinalState) {
        result.outputs.finalState.shape = {result.tiling.sequenceCount, hv,
                                           in.stateVFirst ? kValueDim : kKeyDim,
                                           in.stateVFirst ? kKeyDim : kValueDim, 0};
    }
    result.ok = true;
    return result;
}

} // namespace fwd_h_pseudocode
