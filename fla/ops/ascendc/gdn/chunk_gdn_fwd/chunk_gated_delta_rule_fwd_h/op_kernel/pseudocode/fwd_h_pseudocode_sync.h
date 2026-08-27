// PSEUDOCODE ONLY. Event/flag ledger for intra-core and cross-core handoff.

#pragma once

#include "fwd_h_pseudocode_types.h"

namespace fwd_h_pseudocode {

enum class EventKind {
    HReady,
    HFree,
    WReady,
    WFree,
    kg_ready,
    kg_overwrite_safe,
    PReady,
    RightReady,
    RightFree,
    DReady,
    StateReady,
    StateFree,
    TerminalDrain,
};

struct EventRecord {
    EventKind kind{};
    int slot = -1;
    int producer = -1;
    int consumer = -1;
    EventToken token{};
};

class SyncLedger {
public:
    void Set(EventKind kind, int slot, int producer, int consumer)
    {
        // Set only after the producer pipe has completed its write. Every Set has one Wait.
        if (recordCount_ < static_cast<int>(records_.size())) {
            records_[recordCount_++] = {kind, slot, producer, consumer, NewToken()};
        }
    }

    void Wait(EventKind kind, int slot, int consumer)
    {
        // Wait uses the matching generation and pipe pair; no token is reset by hand.
        (void)kind;
        (void)slot;
        (void)consumer;
    }

    void Release(EventKind kind, int slot, int consumer)
    {
        // A free/overwrite-safe event is emitted only after the last read by this consumer.
        (void)kind;
        (void)slot;
        (void)consumer;
    }

    void WaitBeforeNextRound(const RoundPlan& previousRound)
    {
        // This is the mandatory cross-round barrier. It is called before any next-round
        // H/W/kg prefetch, even when the same kh appears again.
        if (previousRound.stage2Required) {
            for (int i = 0; i < previousRound.requiredKhCount; ++i) {
                Wait(EventKind::kg_overwrite_safe, previousRound.kg[i].slot, /*next round producer*/ -1);
            }
        }
        for (int i = 0; i < previousRound.activeHvCount; ++i) {
            Wait(EventKind::WFree, previousRound.heads[i].wSlot, /*next round producer*/ -1);
            Wait(EventKind::HFree, previousRound.heads[i].hSlot, /*next round producer*/ -1);
        }
        Wait(EventKind::TerminalDrain, previousRound.round, /*scheduler*/ -1);
    }

private:
    EventToken NewToken()
    {
        return EventToken{nextId_++, nextGeneration_++, true};
    }
    std::array<EventRecord, 256> records_{};
    int recordCount_ = 0;
    int nextId_ = 0;
    uint64_t nextGeneration_ = 1;
};

} // namespace fwd_h_pseudocode
