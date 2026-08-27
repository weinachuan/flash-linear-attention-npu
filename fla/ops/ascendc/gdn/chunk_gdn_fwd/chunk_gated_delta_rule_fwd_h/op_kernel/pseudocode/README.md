# FwdH 伪代码蓝图

本目录是 A5 FwdH 重构的实现蓝图。目录形状参考已有的 `op_host`、`op_kernel/gemm`、
`op_kernel/epilogue` 和 fast-launch 入口，但代码被隔离在 `op_kernel/pseudocode/` 下，
明确不加入任何构建目标，也不复制 PR370 的调度或内存行为。

设计文档 `gdn-fwd-h-ascendc-design.md` 是唯一依据。伪代码严格使用设计中的术语：`kg`
保持小写，公开输出继续使用 `V_new`，`state_v_first` 是 kernel 原生属性，不依赖外部转置。

## 模块契约

| 模块 | 职责 | 输入 | 输出 | 生命周期 / 控制 |
| --- | --- | --- | --- | --- |
| `fwd_h_pseudocode_host.h` | 校验 API，推导 `StateT`、门控模式、varlen span 和输出形状 | API tensor、属性、可选数组 | `TilingPlan`、输出描述符、错误 | kernel launch 前执行；拒绝存在性/形状/dtype 不匹配 |
| `fwd_h_pseudocode_round_planner.h` | 构造一个 `(sequence, round, chunk)` 计划和当前 round 的精确 key 集合 | `TilingPlan`、sequence/chunk id | `HeadBinding[<=4]`、`requiredKh[]`、`kg_binding[]` | 每个 chunk 重新构造；不预留整段 sequence 的 `kg` |
| `fwd_h_pseudocode_memory.h` | 跟踪固定 L1/UB 地址和单一所有者状态转换 | slot id、Stage 所有者 | H/W/right/kg 的 ready/free 状态 | H/W/right 在最后一个消费者后释放；`kg` 在最后一次 S2 MTE1 后释放 |
| `fwd_h_pseudocode_sync.h` | 描述 ready/free/terminal 事件和跨 round 屏障 | 生产者、消费者、slot、generation | 匹配的 `Wait`/`Set`/`Release` 操作 | 不留下未消费 token；屏障返回前下一 round 不能预取 |
| `fwd_h_pseudocode_s_minus_one.h` | 每个当前 head round 只转换一次 FP32 初态 | FP32 `initial_state`、布局属性、active head 范围 | L1 中规范 BF16 H0 和 GM h0 | 上一 round 屏障后执行，S0 消费 H 前排空；BF16 或无初态时省略 |
| `fwd_h_pseudocode_stage0.h` | 可选 S0：加载 H/W，异步预取当前 `kg`，计算 `P = W @ H` | 当前 round 计划、GM k/w/state | L1 kg/H/W、UB `P` | `kg` 预取可与 S0 重叠，但直到 S2 才消费；第一个无初态 chunk 不执行 S0 |
| `fwd_h_pseudocode_stage1.h` | S1 全 head VF：计算 `V_new_fp32`、BF16 `V_new`，以及可选的 `V_new_g`、`alpha` | `u`、可选 `P`、g/gk 模式 | 公共 `v_new`、L1 右操作数、alpha、h0 | `V_new`/`V_new_g` 保留到 S2 MTE1；最终且不输出 final state 时走 `v_new-only` |
| `fwd_h_pseudocode_stage2.h` | 将每个 `H_v` 与映射的 `kg` 配对并计算 `D` | 当前 round 的 kg slot、S1 右操作数 | UB FP32 `D` | 只有本 round 内 `kh` 相同的 head 才共享 kg slot；round 结束前失效 |
| `fwd_h_pseudocode_stage3.h` | 更新 rolling state，并按布局写 h/final state | `D`、当前 rolling state、alpha 或 `gk_last` | 下一个 H resident、GM h、GM final_state | 内部 state 始终为 `[K,V]`；GM offset 使用 `state_v_first` |
| `fwd_h_pseudocode_scheduler.h` | 强制 `sequence -> head_round -> chunk`、S-1、屏障和最终分支 | context 和计划 | 完整输出 tensor | 每个后续 head round 前等待 kg 覆盖安全、W/H free 和 terminal drain |
| `chunk_gated_delta_rule_fwd_h_pseudocode.cpp` | fast-launch 形状适配器和 kernel 入口 | 框架 tensor/属性 | tuple `(h, v_new, final_state)` | 只负责入口结构；不做外部 L2 转置 |

## 精确的 round 绑定

g-only 模式：

```text
groupSize = HV / HK
kh(hv) = floor(hv / groupSize)
```

gk-only 模式中 `kh(hv) = hv`；输入 `k` 已经是准备好的 `kg`，不同 value head 不共享条目。
两种模式都使用以下 round 划分：

```text
active_hv_round(r) = [4*r, min(4*(r+1), HV))
required_hk_round(r) = unique({kh(hv) for hv in active_hv_round(r)})
Nkg_round = len(required_hk_round(r))
```

每个 `HeadBinding` 保存完整关系：

```text
head.hv      -> value-head 输入/state/右操作数
head.kh      -> 物理 key head
head.kgSlot  -> 当前 round 的 kg payload
head.hSlot   -> 当前 H/右操作数 L1 slot
head.wSlot   -> 当前 W L1 slot
head.aiv/localSlot -> AIV 分配
```

因此 g-only 的具体绑定为：

```text
HK=1, HV=3, round 0:
  roundHead  hv  kh  kgSlot  右操作数
      0       0   0     0    V_new_g[hv=0]
      1       1   0     0    V_new_g[hv=1]
      2       2   0     0    V_new_g[hv=2]

HK=2, HV=4, round 0:
  roundHead  hv  kh  kgSlot  右操作数
      0       0   0     0    V_new_g[hv=0]
      1       1   0     0    V_new_g[hv=1]
      2       2   1     1    V_new_g[hv=2]
      3       3   1     1    V_new_g[hv=3]
```

Stage2 对表中每一行执行一次 MMAD。共享 `kh` 的两行复用当前 round 的 `kgSlot`，
但仍使用各自不同的 H/value-head 右操作数。gk-only 将 `kh` 替换为 `hv`，因此每行
都会得到独立的 `kgSlot`。

Stage2 对准确的 `head.hv` 使用 `(kg[head.kgSlot], right[head.hSlot])`。因此 `HK:HV=1:3`
生成 3 个 head binding 和 1 个 kg slot，`1:2` 生成 4 个 binding 和 2 个 kg slot，
`1:6` 生成两个 round、每个 round 一个 kg slot；第二个 round 必须在第一个 round 排空后
重新加载 kg。

固定的 L1 key 区域是 `[256, 320)` KiB，共 4 个、每个 16 KiB 的物理 slot。旧方案保留
16 份会单独占用 `16 * 16 KiB = 256 KiB`，而设计还需要 W 的 64 KiB 和 H/resident 的
128 KiB，合计至少 448 KiB，无法放入 320 KiB 的 L1 规划。因此算法不会创建 16-slot 表，
而是只为当前 round 创建 `Nkg_round` 个 slot，在最后一个 S2 MTE1 消费者完成后释放，
下一 round 需要时重新加载相同的 `kh`。

## 时序控制骨架

```text
对 sequence n：
    对 head_round r：
        如果 r > 0：
            等待前一个 round 每个有效 kg slot 的 kg_overwrite_safe
            等待前一个 round 每个 active head 的 W_free 和 H_free
            等待前一个 round 的 terminalDrain
            # 所有等待返回前，禁止预取任何 H/W/kg
        如果 initial_state 为 FP32，则对本 round 的 active head 执行一次 S-1
        对 chunk c：
            plan = BuildRoundPlan(n, r, c)
            如果 plan.stage0Required：
                Stage0：异步预取恰好 Nkg_round 个 kg slot，并计算 P
            Stage1：计算 V_new（仅在 S2 需要时计算 V_new_g/alpha）
            如果 plan.stage2Required：
                Stage2：等待 kg/right，计算 MMAD D，在最后一次 MTE1 后释放 kg/right
                Stage3：更新 state，写 h/final_state，发布下一个 H ready
            否则：
                # final chunk 且 output_final_state=false：不加载 kg、不算 D、不更新 state、不写最终 GM
```

`state_v_first=false` 将逻辑 state `[k,v]` 映射为 `base + k*V + v`；`true` 映射为
`base + v*K + k`。所有 L1/UB resident 始终保持规范 `[K,V]`，入口和调度器都不包含
外部转置。
