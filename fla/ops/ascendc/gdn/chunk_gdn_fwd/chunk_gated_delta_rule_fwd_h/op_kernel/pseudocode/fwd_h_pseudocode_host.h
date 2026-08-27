// 仅伪代码。Host 侧参数校验与 tiling 契约。

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

    // 先检查输入是否存在，再进行 dtype 推导或 GM 解引用。
    const bool hasG = in.g.present;
    const bool hasGk = in.gk.present;
    if (hasG == hasGk) {
        result.error = "g 和 gk 必须且只能提供一个";
        return result;
    }
    if (!in.k.present || !in.w.present || !in.u.present) {
        result.error = "k、w、u 为必选输入";
        return result;
    }
    if (!in.saveNewValue || in.chunkSize != kBatchTokens) {
        result.error = "仅支持 save_new_value=true 且 chunk_size=64";
        return result;
    }
    if (in.outputFinalState != in.finalState.present) {
        result.error = "output_final_state 必须与 final_state 是否存在一致";
        return result;
    }
    if (!ValidateStateShape(in.initialState, in.stateVFirst) ||
        !ValidateStateShape(in.finalState, in.stateVFirst)) {
        result.error = "state 形状与 state_v_first 不匹配";
        return result;
    }
    if (in.initialState.present && in.finalState.present &&
        in.initialState.dtype != in.finalState.dtype) {
        result.error = "initial_state 和 final_state 的 dtype 必须一致";
        return result;
    }
    if (in.k.dtype != StateType::Bf16 || in.w.dtype != StateType::Bf16 ||
        in.u.dtype != StateType::Bf16) {
        result.error = "k、w、u 必须为 BF16";
        return result;
    }
    if (in.k.shape.d3 != kKeyDim || in.u.shape.d3 != kValueDim ||
        in.w.shape.d3 != kKeyDim) {
        result.error = "K 和 V 必须均为 128";
        return result;
    }

    const int64_t hk = in.k.shape.d1;
    const int64_t hv = in.u.shape.d1;
    if (in.w.shape.d1 != hv || in.g.present && in.g.shape.d1 != hv ||
        in.gk.present && in.gk.shape.d1 != hv) {
        result.error = "w、gate、u 的 value-head 维度必须一致";
        return result;
    }
    if (hasG && (hv <= 0 || hk <= 0 || hv % hk != 0)) {
        result.error = "g-only 要求 HV % HK == 0";
        return result;
    }
    if (hasGk && hk != hv) {
        result.error = "gk-only 的 k 已是按 value head 展开的 kg，必须有 HV 个 head";
        return result;
    }

    const bool hasCu = in.cuSeqlens != nullptr;
    const bool hasIndices = in.chunkIndices != nullptr;
    if (hasCu != hasIndices) {
        result.error = "cu_seqlens 和 chunk_indices 必须同时提供";
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

    // 构造 sequence span。真实 Host 实现还需要校验 chunk_indices 是否严格按照
    // 这些 span 推导出的紧凑 [sequence, chunk] 顺序排列。
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
            result.error = "varlen 要求 B=1 且至少提供 [0,T]";
            return result;
        }
        result.tiling.sequenceCount = in.cuSeqlensLength - 1;
        for (int n = 0; n < result.tiling.sequenceCount; ++n) {
            const int64_t begin = in.cuSeqlens[n];
            const int64_t end = in.cuSeqlens[n + 1];
            if (end <= begin) {
                result.error = "varlen 序列不能为空";
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

    // 这里只符号化表示输出分配；真实 meta 函数负责提供 tensor。
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

} // 命名空间 fwd_h_pseudocode
