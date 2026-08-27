// 仅伪代码。S-1：FP32 initial_state -> BF16 H0 Cube shadow。

#pragma once

#include "fwd_h_pseudocode_memory.h"
#include "fwd_h_pseudocode_sync.h"

namespace fwd_h_pseudocode {

struct SMinusOneArgs {
    const ApiInputs* in = nullptr;
    const ApiOutputs* out = nullptr;
    const TilingPlan* tiling = nullptr;
    const SequenceSpan* sequence = nullptr;
    FixedMemory* memory = nullptr;
    SyncLedger* sync = nullptr;
    int activeHvBegin = 0;
    int activeHvCount = 0;
};

inline void RunSMinusOne(const SMinusOneArgs& args)
{
    if (!args.tiling->useInitialState || args.tiling->stateType != StateType::Fp32) {
        return;
    }
    for (int local = 0; local < args.activeHvCount; ++local) {
        const int hv = args.activeHvBegin + local;
        const int hSlot = local;
        args.memory->ProduceInitialH(hSlot);

        // 唯一的 GM state 布局转换发生在这次 layout-aware 读取。
        // VF 输出是供 S0 使用的规范 BF16 [K,V]，同时写入逻辑 h0。
        const auto state = Mte2ReadStateLayoutAware(
            args.in->initialState, args.sequence->sequence, hv, args.tiling->stateLayout);
        const auto h0 = CastBf16(VfCastFp32ToBf16(state));
        Mte3WriteH0LayoutAware(*args.out, *args.tiling, *args.sequence, hv, h0);
        Mte3WriteL1Resident(L1H(args.memory->l1, hSlot), h0);
        args.memory->MarkHReady(hSlot);
        PublishInitialHReady(*args.sync, hSlot, hv);
    }
    WaitInitialPhaseDrain(*args.sync, args.sequence->sequence);
}

} // 命名空间 fwd_h_pseudocode
