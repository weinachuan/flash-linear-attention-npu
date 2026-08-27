// PSEUDOCODE ONLY. Stage1: one full-head VF for V_new and optional V_new_g.

#pragma once

#include "fwd_h_pseudocode_memory.h"
#include "fwd_h_pseudocode_stage0.h"

namespace fwd_h_pseudocode {

enum class Stage1Variant { WithP, NoP, v_new_only };

struct Stage1Args {
    const ApiInputs* in = nullptr;
    const ApiOutputs* out = nullptr;
    const TilingPlan* tiling = nullptr;
    const RoundPlan* plan = nullptr;
    FixedMemory* memory = nullptr;
    SyncLedger* sync = nullptr;
    Stage1Variant variant = Stage1Variant::WithP;
};

struct Stage1Result {
    bool rightOperandReady = false;
    bool alphaReady = false;
};

inline Stage1Result RunStage1(const Stage1Args& args)
{
    Stage1Result result{};
    const RoundPlan& plan = *args.plan;
    for (int i = 0; i < plan.activeHvCount; ++i) {
        const HeadBinding& head = plan.heads[i];
        const bool hasP = plan.stage0Required;
        if (hasP) {
            args.sync->Wait(EventKind::PReady, head.roundHead, /*S1*/ 1);
        }

        // One RegBase VF covers [M,V] for this head. Tail rows are masked and zero-filled.
        const auto p = hasP ? UbP(*args.memory, head) : ZeroP(head);
        const auto v_new_fp32 = VfSubFp32(LoadU(*args.in, plan.chunk, head.hv), p);
        const auto v_new = CastBf16(v_new_fp32);
        write_v_new(*args.out, plan.chunk, head.hv, v_new);

        const bool needsRight = plan.stage2Required && args.variant != Stage1Variant::v_new_only;
        if (needsRight) {
            args.memory->AcquireRightForS1(head.hSlot);
        }
        if (plan.gateMode == GateMode::ScalarG && needsRight) {
            const auto gate = LoadG(*args.in, plan.chunk, head.hv);
            const auto v_new_g = CastBf16(VfMul(v_new_fp32, GateDelta(gate, plan.chunk.validTokens, args.tiling->useExp2)));
            Mte3WriteL1Right(L1Right(args.memory->l1, head.hSlot), v_new_g);
            StoreAlpha(UbAlpha(*args.memory, head), LastGate(gate, plan.chunk.validTokens, args.tiling->useExp2));
            args.sync->Set(EventKind::RightReady, head.hSlot, /*S1 MTE3*/ 1, /*S2*/ 2);
            result.alphaReady = true;
        } else if (plan.gateMode == GateMode::KeyWiseGk && needsRight) {
            // gk-only does not read gk in S1; chunk-relative decay is already in kg.
            Mte3WriteL1Right(L1Right(args.memory->l1, head.hSlot), v_new);
            args.sync->Set(EventKind::RightReady, head.hSlot, /*S1 MTE3*/ 1, /*S2*/ 2);
        }

        if (needsRight) {
            result.rightOperandReady = true;
        }
        if (plan.chunk.first) {
            // First h output is written with native state_v_first addressing. FP32 initial
            // already wrote h0 in S-1, so S1 does not read it again.
            WriteH0IfNeeded(*args.in, *args.out, *args.tiling, plan, head, args.variant);
        }
        if (needsRight) {
            args.memory->MarkRightReady(head.hSlot);
        }
    }
    return result;
}

} // namespace fwd_h_pseudocode
