// PSEUDOCODE ONLY. Stage2: map each value head to its current-round kg slot and MMAD D.

#pragma once

#include "fwd_h_pseudocode_stage1.h"

namespace fwd_h_pseudocode {

struct Stage2Args {
    const ApiInputs* in = nullptr;
    const TilingPlan* tiling = nullptr;
    const RoundPlan* plan = nullptr;
    FixedMemory* memory = nullptr;
    SyncLedger* sync = nullptr;
};

struct Stage2Result {
    bool producedD = false;
};

inline Stage2Result RunStage2(const Stage2Args& args)
{
    Stage2Result result{};
    const RoundPlan& plan = *args.plan;
    if (!plan.stage2Required) {
        return result;
    }

    // A single kg slot is shared by all heads whose HeadBinding.kh is equal. The
    // second index is the right operand's head slot, so every H_v has an explicit pair.
    for (int i = 0; i < plan.activeHvCount; ++i) {
        const HeadBinding& head = plan.heads[i];
        const kg_binding& kg = plan.kg[head.kgSlot];
        if (args.memory->kg[kg.slot].state != SlotState::Ready) {
            // S0 is skipped for the first no-initial chunk. In that case S2 creates only
            // the missing current-round slot, after waiting a previous generation if any.
            if (args.memory->kg[kg.slot].generation > 0) {
                args.sync->Wait(EventKind::kg_overwrite_safe, kg.slot, /*S2 producer*/ 2);
            }
            args.memory->acquire_kg(kg.slot);
            Mte2CopyAndZeroTail(
                /*src=*/KeyPayload(*args.in, plan.chunk, kg.kh, kg.payload),
                /*dst=*/l1_kg(args.memory->l1, kg.slot), plan.chunk.validTokens);
            args.memory->mark_kg_ready(kg.slot);
            args.sync->Set(EventKind::kg_ready, kg.slot, /*MTE2*/ 0, /*S2*/ 2);
        }
        if (head.roundHead == kg.firstConsumerRoundHead) {
            args.sync->Wait(EventKind::kg_ready, kg.slot, /*S2*/ 2);
        }
        args.sync->Wait(EventKind::RightReady, head.hSlot, /*S2*/ 2);
        Mte1Load(l1_kg(args.memory->l1, kg.slot), L0A(head));
        Mte1Load(L1Right(args.memory->l1, head.hSlot), L0B(head));

        // g-only:  D = k_raw^T @ V_new_g (the kg slot stores raw k).
        // gk-only: D = kg^T @ V_new (the input k is already prepared kg).
        // Both paths use BF16 operands and an FP32 accumulator; D remains FP32.
        const auto d = MmadBf16AccFp32(L0A(head), L0B(head), plan.chunk.validTokens);
        FixpipeD(d, UbD(*args.memory, head));
        args.sync->Set(EventKind::DReady, head.roundHead, /*S2*/ 2, /*S3*/ 3);

        // The last value head mapped to this key slot releases it. It is invalid for
        // the next round even if the next round maps to the same kh.
        if (kg.lastConsumerRoundHead == head.roundHead) {
            args.memory->release_kg_after_last_s2_mte1(kg.slot);
            args.sync->Release(EventKind::kg_overwrite_safe, kg.slot, /*S2*/ 2);
        }
        args.memory->ReleaseRightAfterS2Mte1(head.hSlot);
        args.sync->Release(EventKind::RightFree, head.hSlot, /*S2*/ 2);
    }
    result.producedD = true;
    return result;
}

} // namespace fwd_h_pseudocode
