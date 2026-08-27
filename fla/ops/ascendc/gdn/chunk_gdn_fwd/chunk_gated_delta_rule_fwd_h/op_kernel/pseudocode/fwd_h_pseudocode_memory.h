// 仅伪代码。固定的 L1/UB 地址与所有权状态转换。

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
        // H resident 位于 [128, 256)，每个 value head 占 32 KiB。
        // 它的来源只能是 initial GM、S-1 输出或前一个 S3 输出，禁止混合所有者。
        RequireFree(h[hSlot]);
        h[hSlot].state = SlotState::Loading;
    }

    void BeginHReadFromS3(int hSlot)
    {
        // S3 已经生成该 resident，并发布 HReady。
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

} // 命名空间 fwd_h_pseudocode
