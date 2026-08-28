// 仅伪代码。FwdH 的阶段策略、L1/UB 槽位和事件协议。
// 本文件对应真实 op_kernel 中的 chunk_gated_delta_rule_fwd_h_policy.h。

#ifndef FLA_FWD_H_PSEUDOCODE_CHUNK_GATED_DELTA_RULE_FWD_H_POLICY_H_
#define FLA_FWD_H_PSEUDOCODE_CHUNK_GATED_DELTA_RULE_FWD_H_POLICY_H_

#include "chunk_gated_delta_rule_fwd_h_tiling_key.h"

namespace fwd_h_pseudocode {

enum class Stage1Variant { WithP, NoP, v_new_only };

enum class EventKind {
    // 下列 Stage 内事件按执行核类型分槽：AIC 使用 coreHeadId=0..3，AIV 使用本地 coreHeadId=0..1。
    // Set 必须紧跟本 slot 的最后一条 producer 指令，不能挪到另一 slot 的指令之后。
    SMinusVToMte3Ready,
    S0Mte2ToMte1Ready,
    S0Mte1ToCubeReady,
    S0CubeToFixpipeReady,
    S0FixpipeToMte2Ready,
    S0CubeToMte1Free,
    S0FixpipeToCubeFree,
    S1Mte2ToVReady,
    S1VToMte3Ready,
    S1Mte3ToMte2Free,
    S2Mte2ToMte1Ready,
    S2Mte1ToCubeReady,
    S2CubeToFixpipeReady,
    S2FixpipeToMte2Ready,
    S2CubeToMte1Free,
    S2FixpipeToCubeFree,
    S3Mte2ToVReady,
    S3VToMte3Ready,
    S3VToMte2Free,
    StateInitToH0Ready,       // arch22 S1 MTE2/V -> H0 MTE3
    StateInitToStage1VReady,  // arch35 S1 MTE2 -> 单次 RegBase VF
    StateMte2ToStage3VReady,  // FP32 rolling state MTE2 -> S3 VF
    InitialInputReady, // S-1 MTE2 -> S-1 VF
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
    StateToVFree,      // state MTE3 -> 下一 Vector consumer
    StateToMte2Free,   // state MTE3 -> 下一 state MTE2 producer
    UnionFree,         // 特定 v_new-only union -> 下一真实 AIC S0
    CubeRoundDrain,    // 当前 AIC round 的 MTE2/MTE1/Cube/Fixpipe 全部排空
    VectorRoundDrain,  // 当前 AIV round 的 MTE2/V/MTE3 全部排空
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
        // Stage 内事件对应 SetFlag<producer_consumer>(eventId[kind][slot])；AIC/AIV
        // 所有权交接事件对应具名 CrossCoreSetFlag 路由。两者都只覆盖 producer pipe
        // 在 Set 前的指令，因此当前槽的 Set 必须位于下一槽搬运之前。Set 本身不 Wait。
        if (recordCount_ < static_cast<int>(records_.size())) {
            records_[recordCount_++] = {kind, slot, producer, consumer, NewToken(kind, slot)};
        }
    }

    void Wait(EventKind kind, int slot, int consumer)
    {
        // 按 kind 的路由对应 WaitFlag 或 CrossCoreWaitFlag。只等待本 slot、本代际；
        // 当前槽 consumer 不等待其他槽 ready，且不同 AIV 不消费彼此的本核 EventID。
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

    void WaitCubeBeforeNextRound(const RoundPlan& previousRound)
    {
        // AIC 的强制跨 round 屏障。循环上界全部取上一 round 的实际有效数量，
        // 1/2/3/4-head round 只等待真实生成的 token，不补齐未激活的 AIC slot。
        if (previousRound.stage2Required) {
            for (int i = 0; i < previousRound.requiredKhCount; ++i) {
                Wait(EventKind::kg_overwrite_safe, previousRound.kg[i].slot,
                     /*下一 round kg MTE2*/ 0);
            }
        }
        if (previousRound.stage0Required) {
            for (int i = 0; i < previousRound.activeHvCount; ++i) {
                Wait(EventKind::WFree, previousRound.heads[i].wSlot, /*下一 round W MTE2*/ 0);
                Wait(EventKind::HFree, previousRound.heads[i].hSlot, /*下一 round H MTE2*/ 0);
                Wait(EventKind::PFree, previousRound.heads[i].roundHead, /*下一 round S0 Fixpipe*/ 0);
            }
        }
        if (previousRound.stage2Required) {
            for (int i = 0; i < previousRound.activeHvCount; ++i) {
                Wait(EventKind::DFree, previousRound.heads[i].roundHead, /*下一 round local data producer*/ -1);
            }
        }
        Wait(EventKind::CubeRoundDrain, previousRound.round, /*下一 round AIC 调度器*/ -1);
    }

    void WaitVectorBeforeNextRound(const RoundPlan& previousRound, int aivId)
    {
        // 每个 AIV 只等待自己上一 round 实际处理的 roundHead。另一 AIV 没有隐式顺序，
        // 也不能消费本核没有生产的本地槽 EventID。
        if (previousRound.stage2Required) {
            for (int i = 0; i < previousRound.activeHvCount; ++i) {
                const HeadBinding& head = previousRound.heads[i];
                if (head.aiv == aivId) {
                    Wait(EventKind::RightGmFree, head.roundHead,
                         /*下一 round S1 GM producer*/ -1);
                }
            }
        }
        if (previousRound.hasNextHeadRound) {
            if (previousRound.stage3Required) {
                for (int i = 0; i < previousRound.activeHvCount; ++i) {
                    const HeadBinding& head = previousRound.heads[i];
                    if (head.aiv != aivId) {
                        continue;
                    }
                    if (previousRound.stateType == StateType::Bf16) {
                        if (previousRound.nextRoundStartsWithS0) {
                            Wait(EventKind::StateToMte2Free, head.localSlot,
                                 /*下一 round S1 MTE2*/ 0);
                        } else if (previousRound.nextRoundStartsWithS1NoP) {
                            Wait(EventKind::StateToVFree, head.localSlot,
                                 /*下一 round S1 VF*/ 1);
                        }
                    } else if (head.localSlot == 0) {
                        // FP32 scratch 在每个 AIV 内共享，只由本 AIV 的首个 coreHead 等一次。
                        Wait(EventKind::StateToMte2Free, 0, /*下一 round state MTE2*/ 0);
                    }
                }
            }
        }
        Wait(EventKind::VectorRoundDrain, previousRound.round, /*下一 round AIV 调度器*/ -1);
    }

    void PublishCubeRoundDrain(const RoundPlan& terminalPlan)
    {
        // 伪代码接口：按 terminalPlan.activeHvCount/requiredKhCount 排空最后一代实际存在的
        // MTE2、MTE1、Cube、Fixpipe 事件后，再由 AIC 发布一个可供下一 work-round 消费的 token。
        // 落地时必须选目标 CANN 明确支持的 HardEvent 组合，不能用 PIPE_V barrier 代替。
        DrainOutstandingCubePipes(terminalPlan);
        Set(EventKind::CubeRoundDrain, terminalPlan.round, /*AIC drain*/ 0,
            /*下一 AIC work-round*/ -1);
    }

    void PublishVectorRoundDrain(const RoundPlan& terminalPlan, int aivId)
    {
        // 只枚举本 AIV 实际分到的 localSlot，排空 MTE2/V/MTE3；不存在的 pong 不参与。
        DrainOutstandingVectorPipes(terminalPlan, aivId);
        Set(EventKind::VectorRoundDrain, terminalPlan.round, /*AIV drain*/ 1,
            /*下一 AIV work-round*/ -1);
    }

    void DrainCubeAtKernelExit(const RoundPlan& terminalPlan)
    {
        // 最后一个 work-round 没有 token 消费者；直接排空后返回 kernel。
        DrainOutstandingCubePipes(terminalPlan);
    }

    void DrainVectorAtKernelExit(const RoundPlan& terminalPlan, int aivId)
    {
        DrainOutstandingVectorPipes(terminalPlan, aivId);
    }

private:
    void DrainOutstandingCubePipes(const RoundPlan& terminalPlan)
    {
        // 伪代码：真实实现逐个关闭仍在途的 slot/pipe 代际。
        (void)terminalPlan;
    }

    void DrainOutstandingVectorPipes(const RoundPlan& terminalPlan, int aivId)
    {
        // 伪代码：真实实现只处理 head.aiv == aivId 的有效本地槽。
        (void)terminalPlan;
        (void)aivId;
    }

    EventToken NewToken(EventKind kind, int slot)
    {
        // 本核物理键是 (HardEvent pipe 对, pipelineSlot)，跨核物理键还包含同步方向和参与核。
        // kind 在伪代码中保留完整路由身份；id 只是该路由内的 slot 序号。真实实现通过
        // AllocEventID/FetchEventID 或集中 flag 表分配，并在上一代闭环后复用有限资源。
        (void)kind;
        return EventToken{slot, nextGeneration_++, true};
    }

    std::array<EventRecord, 256> records_{};
    int recordCount_ = 0;
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

    void ReleaseStateAfterRoundBarrier(const RoundPlan& previousRound, int aivId)
    {
        // WaitVectorBeforeNextRound 已经排空本 AIV 的异步 state MTE3；只归还本核
        // 实际拥有的 bank，不能替另一 AIV 修改 owner。
        if (previousRound.stateType == StateType::Bf16) {
            for (int i = 0; i < previousRound.activeHvCount; ++i) {
                const HeadBinding& head = previousRound.heads[i];
                if (head.aiv == aivId) {
                    bf16State[LocalBank(head)].owner = StateOwner::Free;
                }
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
    int aivId = 0; // 当前 AIV 子核：0 处理 roundHead 0/2，1 处理 roundHead 1/3
};

} // 命名空间 fwd_h_pseudocode

#endif // FLA_FWD_H_PSEUDOCODE_CHUNK_GATED_DELTA_RULE_FWD_H_POLICY_H_
