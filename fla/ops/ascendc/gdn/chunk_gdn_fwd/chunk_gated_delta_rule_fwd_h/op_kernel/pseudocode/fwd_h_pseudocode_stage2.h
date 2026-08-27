// 仅伪代码。Stage2：将每个 value head 映射到当前 round 的 kg slot，并执行 MMAD 得到 D。

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

    // HeadBinding.kh 相同的 head 共享一个 kg slot。
    // 第二个索引是右操作数的 head slot，因此每个 H_v 都有明确的一对操作数。
    for (int i = 0; i < plan.activeHvCount; ++i) {
        const HeadBinding& head = plan.heads[i];
        const kg_binding& kg = plan.kg[head.kgSlot];
        if (args.memory->kg[kg.slot].state != SlotState::Ready) {
            // 第一个无初态 chunk 会跳过 S0。此时 S2 只创建缺失的当前 round slot，
            // 如存在上一代数据，必须先等待上一代完成。
            if (args.memory->kg[kg.slot].generation > 0) {
                args.sync->Wait(EventKind::kg_overwrite_safe, kg.slot, /*S2 生产者*/ 2);
            }
            args.memory->acquire_kg(kg.slot);
            Mte2CopyAndZeroTail(
                /*源=*/KeyPayload(*args.in, plan.chunk, kg.kh, kg.payload),
                /*目的=*/l1_kg(args.memory->l1, kg.slot), plan.chunk.validTokens);
            args.memory->mark_kg_ready(kg.slot);
            args.sync->Set(EventKind::kg_ready, kg.slot, /*MTE2 生产者*/ 0, /*S2 消费者*/ 2);
        }
        if (head.roundHead == kg.firstConsumerRoundHead) {
            args.sync->Wait(EventKind::kg_ready, kg.slot, /*S2 消费者*/ 2);
        }
        args.sync->Wait(EventKind::RightReady, head.hSlot, /*S2 消费者*/ 2);
        Mte1Load(l1_kg(args.memory->l1, kg.slot), L0A(head));
        Mte1Load(L1Right(args.memory->l1, head.hSlot), L0B(head));

        // g-only：D = k_raw^T @ V_new_g（kg slot 保存 raw k）。
        // gk-only：D = kg^T @ V_new（输入 k 已经是准备好的 kg）。
        // 两条路径都使用 BF16 操作数和 FP32 累加器，D 保持 FP32。
        const auto d = MmadBf16AccFp32(L0A(head), L0B(head), plan.chunk.validTokens);
        FixpipeD(d, UbD(*args.memory, head));
        args.sync->Set(EventKind::DReady, head.roundHead, /*S2 生产者*/ 2, /*S3 消费者*/ 3);

        // 映射到该 key slot 的最后一个 value head 负责释放它。
        // 即使下一 round 映射到相同 kh，该 slot 也不能跨 round 继续使用。
        if (kg.lastConsumerRoundHead == head.roundHead) {
            args.memory->release_kg_after_last_s2_mte1(kg.slot);
            args.sync->Release(EventKind::kg_overwrite_safe, kg.slot, /*S2 消费者*/ 2);
        }
        args.memory->ReleaseRightAfterS2Mte1(head.hSlot);
        args.sync->Release(EventKind::RightFree, head.hSlot, /*S2 消费者*/ 2);
    }
    result.producedD = true;
    return result;
}

} // 命名空间 fwd_h_pseudocode
