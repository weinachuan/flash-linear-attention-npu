# FwdH 伪代码蓝图

本目录是 A5 FwdH 重构的实现蓝图。其内部 `op_kernel` 结构与实际工程保持一致，
但整个目录不加入任何 CMake 构建目标，也不修改现有 kernel。

```text
op_kernel/
├── arch22/
│   ├── chunk_gated_delta_rule_fwd_h_cube.h   # Stage0、Stage2
│   └── chunk_gated_delta_rule_fwd_h_vec.h    # S-1、Stage1、Stage3
├── arch35/
│   ├── chunk_gated_delta_rule_fwd_h_cube.h   # A5 的 Cube 入口
│   └── chunk_gated_delta_rule_fwd_h_vec.h    # A5 的 Vec 入口
├── chunk_gated_delta_rule_fwd_h_tiling_key.h # 常量、属性和计划数据结构
├── chunk_gated_delta_rule_fwd_h_policy.h    # 槽位、所有权和事件策略
├── chunk_gated_delta_rule_fwd_h_utils.h     # 校验、寻址、GVA 映射和 round 计划
└── chunk_gated_delta_rule_fwd_h.cpp         # fast-launch 形状入口和总调度
```

设计文档 `gdn-fwd-h-ascendc-design.md` 是唯一依据。伪代码严格使用设计中的术语：`kg`
保持小写，公开输出继续使用 `V_new`，`state_v_first` 是 kernel 原生属性，不依赖外部转置。

## 文件职责

| 文件 | 职责 | 输入 | 输出 | 生命周期 / 控制 |
| --- | --- | --- | --- | --- |
| `chunk_gated_delta_rule_fwd_h_tiling_key.h` | 定义固定维度、属性、输入输出、`HeadBinding`、`RoundPlan` | API tensor 和属性 | tiling/round 数据结构 | dispatch 前确定；每个 chunk 使用新的 `RoundPlan` |
| `chunk_gated_delta_rule_fwd_h_policy.h` | 定义 Stage 分支、L1/UB 地址、槽位所有权和事件台账 | slot、Stage、producer/consumer | ready/free/terminal 状态 | 槽位复用必须先等待对应 free 或覆盖安全事件 |
| `chunk_gated_delta_rule_fwd_h_utils.h` | 参数校验、state 布局寻址、GVA 映射、`required_hk_round` 规划 | `ApiInputs`、sequence/chunk id | `HostResult`、`RoundPlan` | 不读取未校验的 GM；不按整段 sequence 预留 kg |
| `arch22/chunk_gated_delta_rule_fwd_h_cube.h` | arch22 的 Stage0/Stage2 Cube 伪代码 | H/W/k、当前 round 计划 | P、D、kg ready/free | Stage0 可异步预取 kg；Stage2 最后一次 MTE1 后释放 kg |
| `arch22/chunk_gated_delta_rule_fwd_h_vec.h` | arch22 的 S-1/Stage1/Stage3 Vec 伪代码 | initial_state、u、P、D、门控 | H0、`V_new`、right、h、final_state | `V_new` 只保留到 S2 MTE1；无消费者的最终 chunk 不写 right |
| `arch35/chunk_gated_delta_rule_fwd_h_cube.h` | A5 Cube 入口和架构替换点 | 与 arch22 相同 | 与 arch22 相同 | 当前用 arch22 契约占位，落地时替换 A5 指令/API |
| `arch35/chunk_gated_delta_rule_fwd_h_vec.h` | A5 Vec 入口和 RegBase VF 替换点 | 与 arch22 相同 | 与 arch22 相同 | 当前用 arch22 契约占位，落地时替换 A5 指令/API |
| `chunk_gated_delta_rule_fwd_h.cpp` | Host 适配、架构选择和 sequence -> round -> chunk 调度 | 框架 tensor、属性 | `(h, v_new, final_state)` | 下一 round 必须等上一 round 的 kg/H/W/异步搬运全部排空 |

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

具体 g-only 绑定示例：

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

Stage2 对表中每一行执行一次 MMAD。共享 `kh` 的行复用当前 round 的 `kgSlot`，
但仍使用各自不同的 H/value-head 右操作数。gk-only 将 `kh` 替换为 `hv`，因此每行
都会得到独立的 `kgSlot`。

`HK:HV=1:3` 生成 3 个 head binding 和 1 个 kg slot，`1:2` 生成 4 个 binding 和 2 个
kg slot，`1:6` 生成两个 round、每个 round 一个 kg slot；第二个 round 必须在第一个 round
排空后重新加载 kg。

固定的 L1 key 区域是 `[256, 320)` KiB，共 4 个、每个 16 KiB 的物理 slot。旧方案保留
16 份会单独占用 `16 * 16 KiB = 256 KiB`，而设计还需要 W 的 64 KiB 和 H/resident 的
128 KiB，合计至少 448 KiB，无法放入 320 KiB 的 L1 规划。因此算法不会创建 16-slot 表，
而是只为当前 round 创建 `Nkg_round` 个 slot，在最后一个 S2 MTE1 消费者完成后释放，
下一 round 需要时重新加载相同的 `kh`。

## 阶段和时序

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
                Stage0：异步预取恰好 Nkg_round 个 kg slot，并计算 P = W @ H
            Stage1：计算 V_new（仅在 S2 需要时计算 V_new_g/alpha）
            如果 plan.stage2Required：
                Stage2：等待 kg/right，计算 MMAD D，在最后一次 MTE1 后释放 kg/right
                Stage3：更新 state，写 h/final_state，发布下一个 H ready
            否则：
                # final chunk 且 output_final_state=false：不加载 kg、不算 D、不更新 state
                # 也不把 v_new-only 误写入没有消费者的 L1 右操作数区域
```

`state_v_first=false` 将逻辑 state `[k,v]` 映射为 `base + k*V + v`；`true` 映射为
`base + v*K + k`。所有 L1/UB resident 始终保持规范 `[K,V]`，入口和调度器都不包含
外部转置。
