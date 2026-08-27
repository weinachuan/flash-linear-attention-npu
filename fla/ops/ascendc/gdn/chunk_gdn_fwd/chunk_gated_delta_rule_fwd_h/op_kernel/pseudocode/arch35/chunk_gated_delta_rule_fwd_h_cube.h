// 仅伪代码。arch35（A5）Cube 阶段入口。
// A5 的具体 MTE/Cube/Fixpipe API 在实现时核对；阶段契约与 arch22 保持一致。

#pragma once

#include "../arch22/chunk_gated_delta_rule_fwd_h_cube.h"

namespace fwd_h_pseudocode {

inline CubeStageResult RunStage0Arch35(const CubeStage0Args& args)
{
    // Stage0 阶段公式：P_c,h = W_c,h @ H_c,h；A5 由 Cube 完成 BF16 x BF16 -> FP32 累加。
    // A5 的 Stage0 只处理 H/W -> P；kg 由 Stage2 入口统一搬运并在最后一次 MTE1 后释放。
    // 这里复用相同的数据契约；真实实现替换为 arch35 的 Cube 指令和事件 API。
    return RunStage0Arch22(args);
}

inline CubeStageResult RunStage2Arch35(const CubeStage2Args& args)
{
    // Stage2 阶段公式：g-only 为 D_c,h = k_raw_c,kh^T @ V_new_g,c,h，gk-only 为 D_c,h = kg_c,kh^T @ V_new_c,h。
    // A5 的 Stage2 先搬运当前 round 的 distinct kg，再逐 head 执行 kg @ right；kg slot 只在本 round 内共享。
    return RunStage2Arch22(args);
}

} // 命名空间 fwd_h_pseudocode
