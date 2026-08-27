// 仅伪代码。核内与核间交接使用的事件/flag 台账。

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
        // 只有生产者 pipe 完成写入后才能 Set；每个 Set 都必须有一个对应的 Wait。
        if (recordCount_ < static_cast<int>(records_.size())) {
            records_[recordCount_++] = {kind, slot, producer, consumer, NewToken()};
        }
    }

    void Wait(EventKind kind, int slot, int consumer)
    {
        // Wait 必须使用匹配的 generation 和 pipe 对，禁止手工重置 token。
        (void)kind;
        (void)slot;
        (void)consumer;
    }

    void Release(EventKind kind, int slot, int consumer)
    {
        // 只有当前消费者完成最后一次读取后，才能发出 free/overwrite-safe 事件。
        (void)kind;
        (void)slot;
        (void)consumer;
    }

    void WaitBeforeNextRound(const RoundPlan& previousRound)
    {
        // 这是强制的跨 round 屏障。即使下一 round 再次出现相同 kh，
        // 也必须在任何 H/W/kg 预取之前调用它。
        if (previousRound.stage2Required) {
            for (int i = 0; i < previousRound.requiredKhCount; ++i) {
                Wait(EventKind::kg_overwrite_safe, previousRound.kg[i].slot, /*下一 round 生产者*/ -1);
            }
        }
        for (int i = 0; i < previousRound.activeHvCount; ++i) {
            Wait(EventKind::WFree, previousRound.heads[i].wSlot, /*下一 round 生产者*/ -1);
            Wait(EventKind::HFree, previousRound.heads[i].hSlot, /*下一 round 生产者*/ -1);
        }
        Wait(EventKind::TerminalDrain, previousRound.round, /*调度器*/ -1);
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

} // 命名空间 fwd_h_pseudocode
