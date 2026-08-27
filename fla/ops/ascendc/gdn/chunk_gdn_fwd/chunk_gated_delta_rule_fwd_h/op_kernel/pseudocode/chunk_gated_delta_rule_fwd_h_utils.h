// 仅伪代码。FwdH 的公共校验、布局寻址、GVA 映射和 round 计划工具。
// 本文件对应真实 op_kernel 中的 chunk_gated_delta_rule_fwd_h_utils.h。

#pragma once

#include "chunk_gated_delta_rule_fwd_h_tiling_key.h"

namespace fwd_h_pseudocode {

inline int64_t state_gm_offset(int64_t base, int64_t k, int64_t v, bool stateVFirst)
{
    // 内部 L1/UB 始终使用规范 [K,V]；只有 GM state 的顺序会变化。
    return base + (stateVFirst ? v * kValueDim + k : k * kValueDim + v);
}

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
    if (in.w.shape.d1 != hv || (in.g.present && in.g.shape.d1 != hv) ||
        (in.gk.present && in.gk.shape.d1 != hv)) {
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

inline int HvToKh(int hv, int hk, int valueHeads, GateMode mode)
{
    if (mode == GateMode::KeyWiseGk) {
        return hv; // gk 输入已经按 value head 展开并准备为 kg。
    }
    const int groupSize = valueHeads / hk;
    return hv / groupSize;
}

inline ChunkSpan MakeChunk(const SequenceSpan& seq, int chunk)
{
    ChunkSpan span{};
    span.sequence = seq.sequence;
    span.chunk = chunk;
    span.globalChunk = seq.chunkPrefix + chunk;
    span.tokenBegin = seq.tokenBegin + static_cast<int64_t>(chunk) * kBatchTokens;
    span.validTokens = seq.length - static_cast<int64_t>(chunk) * kBatchTokens;
    if (span.validTokens > kBatchTokens) {
        span.validTokens = kBatchTokens;
    }
    span.first = chunk == 0;
    span.last = chunk + 1 == seq.chunkCount;
    return span;
}

// head_round 级计划只依赖 HV/HK 和门控模式，与 chunk 无关。
// 它必须在进入 chunk 循环前构造；其中的 kg slot 数量和 H_v -> H_k 映射在整个 round 内固定。
inline RoundPlan BuildHeadRoundPlan(const TilingPlan& tiling, int round)
{
    RoundPlan plan{};
    plan.round = round;
    plan.gateMode = tiling.gateMode;
    plan.stateType = tiling.stateType;
    plan.kg_payload_kind = tiling.gateMode == GateMode::ScalarG ? kg_payload::raw_k : kg_payload::prepared_kg;
    plan.activeHvBegin = round * kMaxRoundHeads;
    plan.activeHvCount = static_cast<int>(tiling.hv) - plan.activeHvBegin;
    if (plan.activeHvCount > kMaxRoundHeads) {
        plan.activeHvCount = kMaxRoundHeads;
    }
    // 第一遍：为每个 active value head 创建一个 binding。
    for (int local = 0; local < plan.activeHvCount; ++local) {
        const int hv = plan.activeHvBegin + local;
        auto& binding = plan.heads[local];
        binding.roundHead = local;
        binding.hv = hv;
        binding.kh = HvToKh(hv, static_cast<int>(tiling.hk), static_cast<int>(tiling.hv), plan.gateMode);
        binding.hSlot = local;
        binding.wSlot = local;
        binding.aiv = local % kAivCount;
        binding.localSlot = local / kAivCount;
        binding.active = true;
    }

    // 第二遍：按首次出现顺序收集不重复的 required key head。
    // 只有这里决定 kg slot 数量，不能按整段 sequence 的 HK/HV 预留。
    for (int local = 0; local < plan.activeHvCount; ++local) {
        const int kh = plan.heads[local].kh;
        int slot = -1;
        for (int i = 0; i < plan.requiredKhCount; ++i) {
            if (plan.requiredKh[i] == kh) {
                slot = i;
                break;
            }
        }
        if (slot < 0) {
            slot = plan.requiredKhCount++;
            plan.requiredKh[slot] = kh;
            plan.kg[slot].slot = slot;
            plan.kg[slot].kh = kh;
            plan.kg[slot].payload = plan.kg_payload_kind;
            plan.kg[slot].firstConsumerRoundHead = local;
            plan.kg[slot].lastConsumerRoundHead = local;
        } else {
            plan.kg[slot].lastConsumerRoundHead = local;
        }
        plan.heads[local].kgSlot = slot;
    }
    // 不变量：requiredKhCount == Nkg_round <= activeHvCount <= 4。
    return plan;
}

// 将当前 chunk 的 token 范围和 Stage 分支绑定到已经确定的 head_round 计划。
// 这里只改变 chunk/sequence 字段，不重新计算 H_v -> H_k，也不重新分配 kg slot。
inline RoundPlan BuildChunkPlan(const RoundPlan& headRound, const TilingPlan& tiling,
                                const SequenceSpan& seq, int chunk)
{
    RoundPlan plan = headRound;
    plan.sequence = seq.sequence;
    plan.chunk = MakeChunk(seq, chunk);
    plan.stage2Required = tiling.outputFinalState || !plan.chunk.last;
    plan.stage3Required = plan.stage2Required;
    plan.stage0Required = !(plan.chunk.first && !tiling.useInitialState);
    plan.finalVNewOnly = plan.chunk.last && !tiling.outputFinalState;
    plan.hasNextChunk = !plan.chunk.last;
    plan.hasNextHeadRound = (plan.round + 1) * kMaxRoundHeads < tiling.hv;
    plan.nextRoundStartsWithS0 = plan.hasNextHeadRound && tiling.useInitialState;
    plan.nextRoundStartsWithS1NoP = plan.hasNextHeadRound && !tiling.useInitialState;
    plan.roundBoundaryDrained = plan.round > 0 && plan.chunk.first;
    return plan;
}

} // 命名空间 fwd_h_pseudocode
