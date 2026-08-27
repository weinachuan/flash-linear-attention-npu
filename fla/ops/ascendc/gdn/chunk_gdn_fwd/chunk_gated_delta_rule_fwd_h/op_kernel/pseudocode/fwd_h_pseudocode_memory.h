// PSEUDOCODE ONLY. Fixed L1/UB addresses and owner transitions.

#pragma once

#include "fwd_h_pseudocode_types.h"

namespace fwd_h_pseudocode {

struct MemoryTicket {
    int slot = -1;
    SlotState state = SlotState::Free;
    uint64_t generation = 0;
};

class FixedMemory {
public:
    L1SlotTable l1{};
    UbSlotTable ub{};
    std::array<MemoryTicket, kMaxRoundHeads> h{};
    std::array<MemoryTicket, kMaxRoundHeads> w{};
    std::array<MemoryTicket, kMaxRoundHeads> right{};
    std::array<MemoryTicket, kMaxKeySlots> kg{};

    void AcquireHForS0(int hSlot)
    {
        // H resident is [128, 256), 32 KiB per value head. It is either initial GM,
        // S-1 output, or previous S3 output; no mixed owner is allowed.
        RequireFree(h[hSlot]);
        h[hSlot].state = SlotState::Loading;
    }

    void BeginHReadFromS3(int hSlot)
    {
        // S3 has already produced this resident and published HReady.
        Require(h[hSlot].state == SlotState::Ready);
    }

    void ProduceHForS3(int hSlot)
    {
        // The previous S0/S2 owner must have released both aliases before S3 writes Hnext.
        RequireFree(h[hSlot]);
        RequireFree(right[hSlot]);
        h[hSlot].state = SlotState::Loading;
    }

    void ProduceInitialH(int hSlot)
    {
        // S-1 is a producer equivalent to S3, but only for the current head round.
        ProduceHForS3(hSlot);
    }

    void MarkHReady(int hSlot) { Require(h[hSlot].state == SlotState::Loading); h[hSlot].state = SlotState::Ready; }
    void ReleaseHAfterS0Mte1(int hSlot) { Require(h[hSlot].state == SlotState::Ready); h[hSlot].state = SlotState::Free; }

    void AcquireWForS0(int wSlot) { RequireFree(w[wSlot]); w[wSlot].state = SlotState::Loading; }
    void MarkWReady(int wSlot) { Require(w[wSlot].state == SlotState::Loading); w[wSlot].state = SlotState::Ready; }
    void ReleaseWAfterS0Mte1(int wSlot) { Require(w[wSlot].state == SlotState::Ready); w[wSlot].state = SlotState::Free; }

    void AcquireRightForS1(int hSlot)
    {
        // S1 reuses the same physical H slot only after S0's MTE1 has released it.
        RequireFree(right[hSlot]);
        right[hSlot].state = SlotState::Loading;
    }
    void MarkRightReady(int hSlot) { Require(right[hSlot].state == SlotState::Loading); right[hSlot].state = SlotState::Ready; }
    void ReleaseRightAfterS2Mte1(int hSlot) { Require(right[hSlot].state == SlotState::Ready); right[hSlot].state = SlotState::Free; }

    void acquire_kg(int kgSlot)
    {
        RequireFree(kg[kgSlot]);
        kg[kgSlot].state = SlotState::Loading;
        ++kg[kgSlot].generation;
    }
    void mark_kg_ready(int kgSlot) { Require(kg[kgSlot].state == SlotState::Loading); kg[kgSlot].state = SlotState::Ready; }
    void release_kg_after_last_s2_mte1(int kgSlot) { Require(kg[kgSlot].state == SlotState::Ready); kg[kgSlot].state = SlotState::Free; }

private:
    static void Require(bool condition)
    {
        // Pseudocode assertion: real kernel uses the event protocol, not a user-input static_assert.
        (void)condition;
    }
    static void RequireFree(const MemoryTicket& ticket) { Require(ticket.state == SlotState::Free); }
};

} // namespace fwd_h_pseudocode
