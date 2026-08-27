// 仅伪代码。arch35（A5）Vec 阶段入口。
// A5 的具体 RegBase/VF API 在实现时核对；阶段契约与 arch22 保持一致。

#pragma once

#include "../arch22/chunk_gated_delta_rule_fwd_h_vec.h"

namespace fwd_h_pseudocode {

inline void RunSMinusOneArch35(const SMinusOneArgs& args)
{
    // A5 的 S-1 仍然只做一次 FP32 -> BF16 H0 转换，并原生处理 state_v_first。
    RunSMinusOneArch22(args);
}

inline VecStageResult RunStage1Arch35(const VecStageArgs& args)
{
    // A5 的 Stage1 使用完整 head 的 RegBase VF；最终 v_new-only 不写 L1 右操作数。
    return RunStage1Arch22(args);
}

inline VecStageResult RunStage3Arch35(const VecStage3Args& args)
{
    // A5 的 Stage3 只在存在消费者时生成下一 H 或 final_state。
    return RunStage3Arch22(args);
}

} // 命名空间 fwd_h_pseudocode
