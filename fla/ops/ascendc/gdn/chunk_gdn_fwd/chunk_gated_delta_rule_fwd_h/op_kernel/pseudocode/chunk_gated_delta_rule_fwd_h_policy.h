// 仅伪代码。FwdH 的阶段策略、L1/UB 槽位和事件协议。
// 本文件对应真实 op_kernel 中的 chunk_gated_delta_rule_fwd_h_policy.h。

#ifndef FLA_FWD_H_PSEUDOCODE_CHUNK_GATED_DELTA_RULE_FWD_H_POLICY_H_
#define FLA_FWD_H_PSEUDOCODE_CHUNK_GATED_DELTA_RULE_FWD_H_POLICY_H_

#include "chunk_gated_delta_rule_fwd_h_tiling_key.h"

namespace fwd_h_pseudocode {

enum class Stage1Variant { WithP, NoP, v_new_only };

enum class EventKind {
    InitialInputReady, // S-1 MTE2 -> S-1 VF
    InitialInputFree,  // S-1 VF -> 下一代 S-1 MTE2
    InitialPhaseDrain, // 当前 round 全部 S-1 MTE3 完成
    HGmReady,          // S-1/S3 MTE3(GM layout-aware) -> 下一 S0 MTE2
    HReady,            // S0 MTE2(GM -> L1 NZ) -> S0 MTE1
    HFree,             // S0 MTE1 -> 下一 H owner
    WReady,            // S0 MTE2 -> S0 MTE1
    WFree,             // S0 MTE1 -> 下一 W owner
    kg_ready,          // kg MTE2 -> S2 首个 MTE1
    kg_overwrite_safe, // 当前 round 最后一个 S2 MTE1 -> 下一 kg MTE2
    PReady,            // S0 结果写回并进入 UB -> S1 VF
    PgmFree,           // arch22 L0C -> GM -> MTE2 完成 -> 下一 arch22 P GM writer
    PFree,             // S1 VF 最后一次读取 P -> 下一 local data owner
    LocalDataFree,     // 首块 H0 MTE3 -> 同一 local data bank 的下一真实 producer
    RightGmReady,      // S1 MTE3(UB ND -> GM ND) -> S2 MTE2
    RightGmFree,       // S2 MTE2(GM ND -> L1 NZ) -> 下一 S1 GM producer
    RightL1Ready,      // S2 MTE2(GM ND -> L1 NZ) -> S2 MTE1
    RightFree,         // S2 MTE1 -> 下一 L1 right/H owner
    DReady,            // S2 结果写回并进入 UB -> S3 VF
    DgmFree,           // arch22 L0C -> GM -> MTE2 完成 -> 下一 arch22 D GM writer
    DFree,             // S3 VF 最后一次读取 D -> 下一 local data owner
    VNewWorkFree,      // S1 V_new 相关 MTE3 -> 下一 S1 MTE2
    StateToVFree,      // state MTE3 -> 下一 Vector consumer
    StateToMte2Free,   // state MTE3 -> 下一 state MTE2 producer
    UnionFree,         // 特定 v_new-only union -> 下一真实 AIC S0
    StateReady,        // S1/状态 MTE2 -> S3 VF
    StateFree,         // 状态最后消费者 -> 下一状态 owner
    TerminalDrain,     // round/sequence 末尾所有异步搬运
};

class SyncLedger;
class FixedMemory;

// Cube 阶段的参数契约与具体架构无关；硬件搬运、MMAD 和 Fixpipe 只能在 arch22/arch35
// 各自的 Cube 头文件中实现，禁止把架构特性放进这个公共结构。
struct CubeStage0Args {
    const ApiInputs* in = nullptr;
    const ApiOutputs* out = nullptr;
    const TilingPlan* tiling = nullptr;
    const RoundPlan* plan = nullptr;
    WorkspaceRefs* workspace = nullptr;
    FixedMemory* memory = nullptr;
    SyncLedger* sync = nullptr;
};

struct CubeStage2Args {
    const ApiInputs* in = nullptr;
    const TilingPlan* tiling = nullptr;
    const RoundPlan* plan = nullptr;
    WorkspaceRefs* workspace = nullptr;
    FixedMemory* memory = nullptr;
    SyncLedger* sync = nullptr;
};

struct CubeStageResult {
    bool produced = false;
    int activeTaskCount = 0;
    int kgLoadCount = 0;
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
        // 只有当前消费者完成最后一次读取后，才能发出 free/overwrite-safe 事件；
        // 这里复用 Set 记录一个可被下一 owner Wait 的反向 token。
        Set(kind, slot, consumer, /*下一 owner*/ -1);
    }

    void WaitBeforeNextRound(const RoundPlan& previousRound)
    {
        // 这是强制的跨 round 屏障。即使下一 round 再次出现相同 kh，
        // 也必须在任何 H/W 搬运或 Stage2 kg 搬运之前调用它。
        if (previousRound.stage2Required) {
            for (int i = 0; i < previousRound.requiredKhCount; ++i) {
                Wait(EventKind::kg_overwrite_safe, previousRound.kg[i].slot, /*下一 round 生产者*/ -1);
            }
        }
        if (previousRound.stage0Required) {
            for (int i = 0; i < previousRound.activeHvCount; ++i) {
                Wait(EventKind::WFree, previousRound.heads[i].wSlot, /*下一 round 生产者*/ -1);
                Wait(EventKind::HFree, previousRound.heads[i].hSlot, /*下一 round 生产者*/ -1);
                Wait(EventKind::PFree, previousRound.heads[i].roundHead, /*下一 round S0 Fixpipe*/ 0);
            }
        }
        if (previousRound.stage2Required) {
            // D 是上一轮 S2 的输出；即使本轮没有 S0，也要等 S3 读完 D 后才能复用 local data。
            for (int i = 0; i < previousRound.activeHvCount; ++i) {
                Wait(EventKind::DFree, previousRound.heads[i].roundHead, /*下一 round local data producer*/ -1);
            }
            // 右操作数的 GM scratch 也按 head_round slot 复用；Stage2 MTE2 完成前
            // 下一 round 不得重新写同一 ND scratch。
            for (int i = 0; i < previousRound.activeHvCount; ++i) {
                Wait(EventKind::RightGmFree, previousRound.heads[i].roundHead,
                     /*下一 round S1 GM producer*/ -1);
            }
        }
        if (previousRound.hasNextHeadRound) {
            // S1 的 V_new/right GM MTE3 可能仍读取 UB work bank；下一 round 的 S-1 或 S1
            // 不能靠 head loop 的自然顺序覆写它。
            for (int i = 0; i < previousRound.activeHvCount; ++i) {
                const int bank = (i % kAivCount) * kLocalSlotsPerAiv + (i / kAivCount);
                Wait(EventKind::VNewWorkFree, bank,
                     /*下一 round producer*/ -1);
            }
            if (previousRound.stage3Required) {
                for (int i = 0; i < previousRound.activeHvCount; ++i) {
                    if (previousRound.stateType == StateType::Bf16) {
                        const int bank = (i % kAivCount) * kLocalSlotsPerAiv + (i / kAivCount);
                        if (previousRound.nextRoundStartsWithS0) {
                            Wait(EventKind::StateToMte2Free, bank, /*下一 round S1 MTE2*/ 0);
                        } else if (previousRound.nextRoundStartsWithS1NoP) {
                            Wait(EventKind::StateToVFree, bank, /*下一 round S1 VF*/ 1);
                        }
                    } else if (i == 0) {
                        // FP32 state 使用单个 shared scratch，下一 round 的第一个 S3 负责重新 MTE2。
                        Wait(EventKind::StateToMte2Free, 0, /*下一 round state MTE2*/ 0);
                    }
                }
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
    int rightBaseKiB = 128; // S2 GM->L1 NZ 后占用；与 H slot 物理重叠但生命周期不重叠
};

struct UbSlotTable {
    int localDataBaseKiB[kLocalSlotsPerAiv] = {0, 64};
    int pBaseKiB[kLocalSlotsPerAiv] = {0, 64};
    int dBaseKiB[kLocalSlotsPerAiv] = {0, 64};
    int vNewWorkBaseBf16KiB[kLocalSlotsPerAiv] = {192, 208};
    int vNewWorkBaseFp32KiB[kLocalSlotsPerAiv] = {128, 144};
    int stateScratchBaseKiB = 160;
    int gateBaseKiB = 224;
};

class FixedMemory {
public:
    L1SlotTable l1{};
    UbSlotTable ub{};
    std::array<MemoryTicket, kMaxRoundHeads> h{};
    std::array<MemoryTicket, kMaxRoundHeads> w{};
    std::array<MemoryTicket, kMaxRoundHeads> rightGm{};
    std::array<MemoryTicket, kMaxRoundHeads> right{};
    std::array<MemoryTicket, kMaxKeySlots> kg{};
    std::array<LocalDataTicket, kAivCount * kLocalSlotsPerAiv> localData{};
    std::array<MemoryTicket, kAivCount * kLocalSlotsPerAiv> vNewWork{};
    std::array<StateTicket, kAivCount * kLocalSlotsPerAiv> bf16State{};
    StateTicket fp32StateScratch{0, StateOwner::Free, 0};
    std::array<MemoryTicket, kAivCount * kLocalSlotsPerAiv> initialInput{};
    std::array<MemoryTicket, kAivCount * kLocalSlotsPerAiv> initialHOutput{};

    static int LocalBank(const HeadBinding& head)
    {
        return head.aiv * kLocalSlotsPerAiv + head.localSlot;
    }

    void AcquireLocalDataForP(const HeadBinding& head)
    {
        auto& ticket = localData[LocalBank(head)];
        Require(ticket.owner == LocalDataOwner::Free);
        ticket.owner = LocalDataOwner::P;
        ++ticket.generation;
    }

    void MarkPReady(const HeadBinding& head)
    {
        Require(localData[LocalBank(head)].owner == LocalDataOwner::P);
    }

    void ReleasePAfterS1(const HeadBinding& head)
    {
        auto& ticket = localData[LocalBank(head)];
        Require(ticket.owner == LocalDataOwner::P);
        ticket.previousOwner = ticket.owner;
        ticket.owner = LocalDataOwner::Free;
    }

    void BeginHWriteAfterDRow(const HeadBinding& head)
    {
        auto& ticket = localData[LocalBank(head)];
        Require(ticket.owner == LocalDataOwner::D);
        ticket.owner = LocalDataOwner::HWrite;
    }

    void AcquireLocalDataForH0(const HeadBinding& head)
    {
        auto& ticket = localData[LocalBank(head)];
        Require(ticket.owner == LocalDataOwner::Free);
        ticket.owner = LocalDataOwner::H0;
        ++ticket.generation;
    }

    void ReleaseH0AfterMte3(const HeadBinding& head)
    {
        auto& ticket = localData[LocalBank(head)];
        Require(ticket.owner == LocalDataOwner::H0);
        ticket.previousOwner = ticket.owner;
        ticket.owner = LocalDataOwner::Free;
    }

    void AcquireLocalDataForD(const HeadBinding& head)
    {
        auto& ticket = localData[LocalBank(head)];
        Require(ticket.owner == LocalDataOwner::Free);
        ticket.owner = LocalDataOwner::D;
        ++ticket.generation;
    }

    void MarkDReady(const HeadBinding& head)
    {
        Require(localData[LocalBank(head)].owner == LocalDataOwner::D);
    }

    void ReleaseDAfterS3(const HeadBinding& head)
    {
        auto& ticket = localData[LocalBank(head)];
        Require(ticket.owner == LocalDataOwner::D || ticket.owner == LocalDataOwner::HWrite);
        ticket.previousOwner = ticket.owner;
        ticket.owner = LocalDataOwner::Free;
    }

    void AcquireVNewWorkForS1(const HeadBinding& head)
    {
        auto& ticket = vNewWork[LocalBank(head)];
        RequireFree(ticket);
        ticket.state = SlotState::Loading;
        ++ticket.generation;
    }

    void MarkVNewWorkReady(const HeadBinding& head)
    {
        Require(vNewWork[LocalBank(head)].state == SlotState::Loading);
        vNewWork[LocalBank(head)].state = SlotState::Ready;
    }

    void ReleaseVNewWorkAfterMte3(const HeadBinding& head)
    {
        Require(vNewWork[LocalBank(head)].state == SlotState::Ready);
        vNewWork[LocalBank(head)].state = SlotState::Free;
    }

    void InitializeBf16StateInS1(const HeadBinding& head)
    {
        auto& ticket = bf16State[LocalBank(head)];
        Require(ticket.owner == StateOwner::Free);
        ticket.owner = StateOwner::RResident;
        ++ticket.generation;
    }

    void AcquireInitialInput(const HeadBinding& head)
    {
        auto& ticket = initialInput[LocalBank(head)];
        RequireFree(ticket);
        ticket.state = SlotState::Loading;
        ++ticket.generation;
    }

    void MarkInitialInputReady(const HeadBinding& head)
    {
        Require(initialInput[LocalBank(head)].state == SlotState::Loading);
        initialInput[LocalBank(head)].state = SlotState::Ready;
    }

    void ReleaseInitialInput(const HeadBinding& head)
    {
        Require(initialInput[LocalBank(head)].state == SlotState::Ready);
        initialInput[LocalBank(head)].state = SlotState::Free;
    }

    void AcquireInitialHOutput(const HeadBinding& head)
    {
        auto& ticket = initialHOutput[LocalBank(head)];
        RequireFree(ticket);
        ticket.state = SlotState::Loading;
        ++ticket.generation;
    }

    void MarkInitialHOutputReady(const HeadBinding& head)
    {
        Require(initialHOutput[LocalBank(head)].state == SlotState::Loading);
        initialHOutput[LocalBank(head)].state = SlotState::Ready;
    }

    void ReleaseInitialHOutput(const HeadBinding& head)
    {
        Require(initialHOutput[LocalBank(head)].state == SlotState::Ready);
        initialHOutput[LocalBank(head)].state = SlotState::Free;
    }

    void AcquireBf16StateForS3(const HeadBinding& head)
    {
        auto& ticket = bf16State[LocalBank(head)];
        Require(ticket.owner == StateOwner::Free || ticket.owner == StateOwner::RResident);
        ticket.owner = StateOwner::RResident;
        ++ticket.generation;
    }

    void MarkBf16StateMte3InFlight(const HeadBinding& head)
    {
        auto& ticket = bf16State[LocalBank(head)];
        Require(ticket.owner == StateOwner::RResident);
        ticket.owner = StateOwner::RNextMte3;
    }

    void MarkBf16StateConsumedByNextVf(const HeadBinding& head)
    {
        auto& ticket = bf16State[LocalBank(head)];
        Require(ticket.owner == StateOwner::RNextMte3 || ticket.owner == StateOwner::RResident);
        ticket.owner = StateOwner::RResident;
    }

    void ReleaseBf16StateAtTerminal(const HeadBinding& head)
    {
        auto& ticket = bf16State[LocalBank(head)];
        Require(ticket.owner == StateOwner::RResident || ticket.owner == StateOwner::RNextMte3);
        ticket.owner = StateOwner::Free;
    }

    void AcquireFp32StateScratchForS3()
    {
        Require(fp32StateScratch.owner == StateOwner::Free);
        fp32StateScratch.owner = StateOwner::RollingMte2;
        ++fp32StateScratch.generation;
    }

    void MarkFp32StateReady()
    {
        Require(fp32StateScratch.owner == StateOwner::RollingMte2);
        fp32StateScratch.owner = StateOwner::RResident;
    }

    void ReleaseFp32StateScratch()
    {
        Require(fp32StateScratch.owner == StateOwner::RResident ||
                fp32StateScratch.owner == StateOwner::RNextMte3);
        fp32StateScratch.owner = StateOwner::Free;
    }

    void ReleaseStateAfterRoundBarrier(const RoundPlan& previousRound)
    {
        // WaitBeforeNextRound 已经排空异步 state MTE3；把 owner 显式归还，
        // 使下一 round 的首块 S1 初始化或 S3 MTE2 不再依赖旧 round 的 owner。
        if (previousRound.stateType == StateType::Bf16) {
            for (int i = 0; i < previousRound.activeHvCount; ++i) {
                bf16State[LocalBank(previousRound.heads[i])].owner = StateOwner::Free;
            }
        } else {
            fp32StateScratch.owner = StateOwner::Free;
        }
    }

    void MarkFp32StateMte3InFlight()
    {
        Require(fp32StateScratch.owner == StateOwner::RResident);
        fp32StateScratch.owner = StateOwner::RNextMte3;
    }

    void AcquireHForS0(int hSlot)
    {
        // H resident 位于 [128, 256)，每个 value head 占 32 KiB。
        // 来源只能是 initial GM、S-1 输出或前一个 S3 输出，禁止混合所有者。
        RequireFree(h[hSlot]);
        h[hSlot].state = SlotState::Loading;
        ++h[hSlot].generation;
    }

    void BeginHReadFromS3(int hSlot)
    {
        // Stage0 已把 S3 的 GM ND 转成 L1 NZ，并发布 HReady。
        Require(h[hSlot].state == SlotState::Ready);
    }

    void ProduceHForS3(int hSlot)
    {
        // S3 写入 Hnext 前，前一个 S0/S2 所有者必须释放两个别名。
        RequireFree(h[hSlot]);
        RequireFree(right[hSlot]);
        h[hSlot].state = SlotState::Loading;
        ++h[hSlot].generation;
    }

    void ProduceInitialH(int hSlot)
    {
        // S-1 的生产语义等价于 S3，但只服务当前 head round。
        ProduceHForS3(hSlot);
    }

    void MarkHReady(int hSlot) { Require(h[hSlot].state == SlotState::Loading); h[hSlot].state = SlotState::Ready; }
    void ReleaseHAfterS0Mte1(int hSlot) { Require(h[hSlot].state == SlotState::Ready); h[hSlot].state = SlotState::Free; }

    void AcquireWForS0(int wSlot)
    {
        RequireFree(w[wSlot]);
        w[wSlot].state = SlotState::Loading;
        ++w[wSlot].generation;
    }
    void MarkWReady(int wSlot) { Require(w[wSlot].state == SlotState::Loading); w[wSlot].state = SlotState::Ready; }
    void ReleaseWAfterS0Mte1(int wSlot) { Require(w[wSlot].state == SlotState::Ready); w[wSlot].state = SlotState::Free; }

    void AcquireRightGmForS1(int roundHead)
    {
        // S1 只拥有 GM ND scratch，不拥有与 H 重叠的 L1 right slot。
        RequireFree(rightGm[roundHead]);
        rightGm[roundHead].state = SlotState::Loading;
        ++rightGm[roundHead].generation;
    }
    void MarkRightGmReady(int roundHead)
    {
        Require(rightGm[roundHead].state == SlotState::Loading);
        rightGm[roundHead].state = SlotState::Ready;
    }
    void ReleaseRightGmAfterS2Mte2(int roundHead)
    {
        Require(rightGm[roundHead].state == SlotState::Ready);
        rightGm[roundHead].state = SlotState::Free;
    }

    void AcquireRightForS2Mte2(int hSlot)
    {
        // L1 right 只有在 Stage2 的 GM->L1 NZ 开始时才 acquire；S1 不直接写 L1。
        RequireFree(right[hSlot]);
        right[hSlot].state = SlotState::Loading;
        ++right[hSlot].generation;
    }
    void MarkRightL1Ready(int hSlot) { Require(right[hSlot].state == SlotState::Loading); right[hSlot].state = SlotState::Ready; }
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
        return plan.finalVNewOnly || (plan.chunk.last && !outputFinalState);
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
    WorkspaceRefs workspace{};
    TilingPlan tiling{};
    FixedMemory memory{};
    SyncLedger sync{};
};

} // 命名空间 fwd_h_pseudocode

#endif // FLA_FWD_H_PSEUDOCODE_CHUNK_GATED_DELTA_RULE_FWD_H_POLICY_H_
