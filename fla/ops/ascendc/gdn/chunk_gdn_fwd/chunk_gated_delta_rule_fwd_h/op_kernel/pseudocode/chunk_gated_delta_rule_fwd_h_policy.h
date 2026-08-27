// 仅伪代码。FwdH 的阶段策略、L1/UB 槽位和事件协议。
// 本文件对应真实 op_kernel 中的 chunk_gated_delta_rule_fwd_h_policy.h。

#pragma once

#include "chunk_gated_delta_rule_fwd_h_tiling_key.h"

namespace fwd_h_pseudocode {

enum class Stage1Variant { WithP, NoP, v_new_only };

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

struct EventToken {
    int id = -1;
    uint64_t generation = 0;
    bool valid = false;
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
        if (previousRound.stage0Required) {
            for (int i = 0; i < previousRound.activeHvCount; ++i) {
                Wait(EventKind::WFree, previousRound.heads[i].wSlot, /*下一 round 生产者*/ -1);
                Wait(EventKind::HFree, previousRound.heads[i].hSlot, /*下一 round 生产者*/ -1);
            }
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

struct MemoryTicket {
    int slot = -1;
    SlotState state = SlotState::Free;
    uint64_t generation = 0;
};

struct L1SlotTable {
    // 地址单位为 KiB，在整个 dispatch 期间固定不变。
    int wBaseKiB = 0;       // [0, 64)：4 个 16 KiB W slot
    int hBaseKiB = 128;     // [128, 256)：4 个 32 KiB H slot
    int kgBaseKiB = 256;    // [256, 320)：4 个 16 KiB kg slot
    int rightBaseKiB = 128; // S0 读取释放后，S1 才能复用 H slot
};

struct UbSlotTable {
    int localDataBaseKiB[kLocalSlotsPerAiv] = {0, 64};
    int pBaseKiB[kLocalSlotsPerAiv] = {0, 64};
    int dBaseKiB[kLocalSlotsPerAiv] = {0, 64};
    int v_new_work_base_kib[kLocalSlotsPerAiv] = {128, 160};
    int stateScratchBaseKiB = 160;
    int gateBaseKiB = 224;
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
        // H resident 位于 [128, 256)，每个 value head 占 32 KiB。
        // 来源只能是 initial GM、S-1 输出或前一个 S3 输出，禁止混合所有者。
        RequireFree(h[hSlot]);
        h[hSlot].state = SlotState::Loading;
    }

    void BeginHReadFromS3(int hSlot)
    {
        // S3 已生成该 resident，并发布 HReady。
        Require(h[hSlot].state == SlotState::Ready);
    }

    void ProduceHForS3(int hSlot)
    {
        // S3 写入 Hnext 前，前一个 S0/S2 所有者必须释放两个别名。
        RequireFree(h[hSlot]);
        RequireFree(right[hSlot]);
        h[hSlot].state = SlotState::Loading;
    }

    void ProduceInitialH(int hSlot)
    {
        // S-1 的生产语义等价于 S3，但只服务当前 head round。
        ProduceHForS3(hSlot);
    }

    void MarkHReady(int hSlot) { Require(h[hSlot].state == SlotState::Loading); h[hSlot].state = SlotState::Ready; }
    void ReleaseHAfterS0Mte1(int hSlot) { Require(h[hSlot].state == SlotState::Ready); h[hSlot].state = SlotState::Free; }

    void AcquireWForS0(int wSlot) { RequireFree(w[wSlot]); w[wSlot].state = SlotState::Loading; }
    void MarkWReady(int wSlot) { Require(w[wSlot].state == SlotState::Loading); w[wSlot].state = SlotState::Ready; }
    void ReleaseWAfterS0Mte1(int wSlot) { Require(w[wSlot].state == SlotState::Ready); w[wSlot].state = SlotState::Free; }

    void AcquireRightForS1(int hSlot)
    {
        // 只有 S0 的 MTE1 释放物理 H slot 后，S1 才能复用该 slot。
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
        // 伪代码断言：真实 kernel 使用事件协议，而不是对用户输入做 static_assert。
        (void)condition;
    }
    static void RequireFree(const MemoryTicket& ticket) { Require(ticket.state == SlotState::Free); }
};

struct FwdHStagePolicy {
    static bool IsFinalVNewOnly(const RoundPlan& plan, bool outputFinalState)
    {
        return plan.chunk.last && !outputFinalState;
    }

    static bool NeedStage0(const RoundPlan& plan)
    {
        return plan.stage0Required;
    }

    static bool NeedStage2(const RoundPlan& plan)
    {
        return plan.stage2Required;
    }
};

struct SchedulerContext {
    ApiInputs inputs{};
    ApiOutputs outputs{};
    TilingPlan tiling{};
    FixedMemory memory{};
    SyncLedger sync{};
};

} // 命名空间 fwd_h_pseudocode
