// 仅伪代码。精确的 GVA round 规划与 head 到 key 的绑定。

#pragma once

#include "fwd_h_pseudocode_types.h"

namespace fwd_h_pseudocode {

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

inline RoundPlan BuildRoundPlan(const TilingPlan& tiling, int sequence, int round, int chunk)
{
    RoundPlan plan{};
    const auto& seq = tiling.sequences[sequence];
    plan.sequence = sequence;
    plan.round = round;
    plan.chunk = MakeChunk(seq, chunk);
    plan.gateMode = tiling.gateMode;
    plan.kg_payload_kind = tiling.gateMode == GateMode::ScalarG ? kg_payload::raw_k : kg_payload::prepared_kg;
    plan.activeHvBegin = round * kMaxRoundHeads;
    plan.activeHvCount = static_cast<int>(tiling.hv) - plan.activeHvBegin;
    if (plan.activeHvCount > kMaxRoundHeads) {
        plan.activeHvCount = kMaxRoundHeads;
    }
    plan.stage2Required = tiling.outputFinalState || !plan.chunk.last;
    plan.stage3Required = plan.stage2Required;
    plan.stage0Required = !(plan.chunk.first && !tiling.useInitialState);

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

} // 命名空间 fwd_h_pseudocode
