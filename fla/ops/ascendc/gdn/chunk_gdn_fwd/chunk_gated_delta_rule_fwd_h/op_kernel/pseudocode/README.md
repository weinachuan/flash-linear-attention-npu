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
| `chunk_gated_delta_rule_fwd_h_tiling_key.h` | 定义固定维度、属性、输入输出、`HeadBinding`、`RoundPlan` 及 owner 数据结构 | API tensor 和属性 | tiling/head-round/chunk 数据结构 | dispatch 前确定 head 映射；每个 chunk 只复制并补充 token/Stage 字段 |
| `chunk_gated_delta_rule_fwd_h_policy.h` | 定义 Stage 分支、L1/UB 地址、槽位所有权和事件台账 | slot、Stage、producer/consumer | ready/free/terminal 状态 | 槽位复用必须先等待对应 free 或覆盖安全事件 |
| `chunk_gated_delta_rule_fwd_h_utils.h` | 参数校验、state 布局寻址、GVA 映射、`required_hk_round` 规划 | `ApiInputs`、sequence/chunk id | `HostResult`、`RoundPlan` | 不读取未校验的 GM；不按整段 sequence 预留 kg |
| `arch22/chunk_gated_delta_rule_fwd_h_cube.h` | arch22 的 Stage0/Stage2 Cube 伪代码 | H/W/k、当前 round 计划 | P、D、kg ready/free | Stage0 可异步预取 kg；Stage2 最后一次 MTE1 后释放 kg |
| `arch22/chunk_gated_delta_rule_fwd_h_vec.h` | arch22 的 S-1/Stage1/Stage3 Vec 伪代码 | initial_state、u、P、D、门控 | H0、`V_new`、right、h、final_state | `V_new` 只保留到 S2 MTE1；无消费者的最终 chunk 不写 right |
| `arch35/chunk_gated_delta_rule_fwd_h_cube.h` | A5 Cube 入口和架构替换点 | 与 arch22 相同 | 与 arch22 相同 | 当前用 arch22 契约占位，落地时替换 A5 指令/API |
| `arch35/chunk_gated_delta_rule_fwd_h_vec.h` | A5 Vec 入口和 RegBase VF 替换点 | 与 arch22 相同 | 与 arch22 相同 | 当前用 arch22 契约占位，落地时替换 A5 指令/API |
| `chunk_gated_delta_rule_fwd_h.cpp` | Host 适配、架构选择和 sequence -> round -> chunk 调度 | 框架 tensor、属性 | `(h, v_new, final_state)` | 下一 round 必须等上一 round 的 kg/H/W/异步搬运全部排空 |

## Stage 内部模块

以下每个模块对应一个可替换的 Ascend C 实现单元。模块输入和输出均以一个
`(sequence, chunk, value_head)` 为原子粒度；一个 `head_round` 最多包含四个这样的 task。
`RoundPlan` 提供 `HeadBinding`、chunk 有效行数 `M`、gate mode、StateT 和固定 slot，模块不能
自行重新推导 head 映射或临时申请地址。

### S-1：FP32 initial 转换（仅 FP32 initial）

| 模块 | 输入 | 输出 | 主要驻留 | 必须闭合的事件 |
| --- | --- | --- | --- | --- |
| `SMinusOneLoadInitial` | `initial_state[n,hv,:,:]`、`state_v_first` | FP32 canonical `[K,V]` | 独立 initial input bank | `InitialInputReady`、上一代 `InitialInputFree` |
| `SMinusOneConvertAndWriteH0` | FP32 initial bank | BF16 `H_0` | 独立 H0 bank、L1 `H` slot、公开 `h0` | VF 末次读取 input 后 `InitialInputFree`；两路 MTE3 完成后 `HReady` |
| `RunSMinusOneArch22` | 当前 round 的 active head mask | 当前 round 全部 H ready | 不占用 main P/D local data bank | 所有 active head 的 `InitialPhaseDrain` 后才能启动 S0 |

S-1 的一次完整 task 顺序固定为：`MTE2(initial GM -> FP32 UB)` -> `MTE2_V ready` ->
`VF cast FP32 -> BF16` -> `MTE3(h0 GM)` 与 `MTE3(H L1)`。两路 MTE3 都完成后才释放 H0
bank；`HReady` 在 phase drain 后统一发布，不能让单个 head 先启动 S0。

### Stage0：Cube 计算 `P = W @ H`

| 模块 | 输入 | 输出 | L1/UB 驻留 | 必须闭合的事件 |
| --- | --- | --- | --- | --- |
| `Stage0PrefetchKgAsync` | 当前 `requiredKh[]` 的 `(chunk, kh)` | `kg`/`k_raw` payload | L1 `[256,320)` 的 `Nkg_round` 个 slot | 等上一代 `kg_overwrite_safe` 后 acquire；只发起 MTE2，不在 S0 等待 |
| `Stage0LoadH` | BF16 initial、S-1 H0 或前一 S3 H | canonical `H_c[K,V]` | L1 head slot `[128,256)` | BF16 initial 为本地 MTE2；其他来源等待 `HReady` |
| `Stage0LoadW` | `w[chunk,hv]` 有效 `M` 行 | `w_c[M,K]` | L1 W slot `[0,64)` | `WFree -> WReady`；尾部先清零再覆盖有效行 |
| `Stage0ComputeP` | L1 W/H | `Pacc` FP32，再转 `P` | L0A/L0B；UB local data bank 的 P 区 | `WReady/HReady -> MTE1 -> MMAD -> Fixpipe(PReady)`；S1 末次读取后 `PFree` |

S0 的真实顺序是：先为当前 round 的每个 distinct `kh` 发起 `kg` 异步 MTE2，然后每个
head 依次完成 H/W MTE2、MTE1、MMAD 和 Fixpipe。kg MTE2 可以与 H/W/P 重叠；`kg_ready`
只在 Stage2 首个消费者前等待和发布。`P` 按 StateT 选择 `F322BF16` 或 `NoQuant`，不经过
GM。首 chunk 无 initial 时整个 S0 跳过，不能用 `W @ 0` 伪造 P。

### Stage1：Vector 计算 `V_new` 和 Stage2 右操作数

| 模块 | 输入 | 输出 | UB/L1 驻留 | 必须闭合的事件 |
| --- | --- | --- | --- | --- |
| `Stage1PrepareInitialOrZeroState` | BF16 initial 或空初态 | `R_0/H_0` | BF16 state UB `[128,192)`；必要时 FP32 zero-H0 固定目标 | `StateReady`；H0 MTE3 只在确有下一 S0 时发布 `UnionFree` |
| `Stage1LoadUAndGate` | `u[0,M)`、g-only 的 `g[0,M)` | UB `u_c`、gate 临时区 | V_new work bank；gate `[224,225)` | 上代 `VNewWorkFree` 后 acquire；MTE2 -> VF ready |
| `Stage1ComputeAndWriteVNew` | `u_c`、可选 `P_c`、gate | FP32 `V_new_fp32`、BF16 `V_new`、可选 `V_new_g/alpha` | `V_new` work bank；g-only 独立 `V_new_g` 区；alpha `[225,226)` | `PReady -> VF`；P 末次读取后 `PFree` |
| `Stage1WriteRightOperand` | g-only `V_new_g` 或 gk-only `V_new` | L1 zN 右操作数 | 对应 H slot `[128,256)` | MTE3 完成后 `RightReady`；S2 末次 MTE1 后 `RightFree` |
| `Stage1ReleaseVNewWork` | V_new UB bank | 释放 work bank | 不写无消费者的 L1 | 所有以 bank 为源的 MTE3 完成后 `VNewWorkFree` |

每个 head 只调用一次完整 VF。VF 内先完成 `V_new_fp32 = u - P`，再按分支转 BF16 并写
公开 `v_new`。g-only 在同一次 VF 内生成 `V_new_g` 和 `alpha`，MTE3 直接将 ND 输出按
zN 描述符写入 L1；gk-only 不读取 gk，直接将 `V_new` 写入 L1。最终 chunk 且
`output_final_state=false` 时选择 `v_new-only`：只搬 `u` 和必要的 `P`，不搬 gate、不生成
`V_new_g/alpha`、不写 L1 右操作数。

### Stage2：Cube 计算 `D`

| 模块 | 输入 | 输出 | L1/UB 驻留 | 必须闭合的事件 |
| --- | --- | --- | --- | --- |
| `Stage2EnsureKgReady` | 当前 `kg_binding` | valid `kg[M,K]` | L1 kg slot `[256,320)` | S0 Loading 时等待 MTE2；未预取时按 distinct `(chunk,kh)` 补齐 |
| `Stage2WaitRightAndAcquireD` | L1 右操作数、P/D local data owner | D owner | UB local data bank `[0,128)` | `RightReady` 和前一 owner `PFree/DFree` |
| `Stage2ComputeDForHead` | g-only `k_raw + V_new_g` 或 gk-only `kg + V_new` | FP32 `D` | L0A/L0B；Fixpipe 到 UB D 区 `[0,128)` | 首个 mapped head 等 `kg_ready`；`MMAD -> NoQuant Fixpipe -> DReady` |
| `Stage2ReleaseInputs` | 本次 MTE1 读取完成 | 释放 right/kg | L1 right 与 kg slot | right 最后消费者 `RightFree`；kg 最后消费者 `kg_overwrite_safe` |

同一个 `kh` 的多个 value head 共享一份 L1 kg payload，但每个 head 仍以自己的右操作数
执行一次 MMAD。`kg_ready` 只由第一个 mapped head 等待一次，后续 head 直接复用 valid entry；
最后一个 mapped head 的 MTE1 完成后才释放该 kg slot。S2 的 D 始终 `NoQuant` 写 FP32 UB，
不写 GM，不读取本 Stage 产生的 D。

### Stage3：Vector 更新 rolling state

| 模块 | 输入 | 输出 | UB/L1 驻留 | 必须闭合的事件 |
| --- | --- | --- | --- | --- |
| `Stage3PrepareState` | BF16 UB resident 或 FP32 rolling GM | 当前 `R_c` | BF16 `[128,192)`；FP32 shared scratch `[160,224)` | 前代 state MTE3 等待 `StateToVFree/StateToMte2Free`；FP32 MTE2 后 `StateReady` |
| `Stage3LoadGate` | g-only alpha 或 gk 最后有效行 `gk[M-1,:]` | VF gate 输入 | alpha `[225,226)` 或 gk_last `[224,225)` | gk MTE2 -> VF ready；尾 chunk 使用 `M-1` |
| `Stage3ComputeRNext` | `D_c`、`R_c`、gate | `Rnext`、BF16 `Hnext` | D local data；state scratch/resident | `DReady/StateReady -> 一次完整 VF` |
| `Stage3WriteStateOutputs` | canonical `Rnext/Hnext` | 非末 chunk 的 `h`+L1 H，末 chunk 的 `final_state` | L1 H slot；按 `state_v_first` 写 GM | 两路 H MTE3 完成后 `HReady`；无后继不发无消费者 token |
| `Stage3ReleaseStateAndD` | D 最后读取、state 最后输出 | 释放 D/state/gate | owner 迁移为 Free 或下一代 owner | `DFree`；state 按下一实际消费者发布对应 free |

g-only 使用 `Rnext = R_c * alpha + D`，gk-only 使用最后有效行 gate 做逐行缩放；所有
Vector 算术在 FP32 寄存器中完成，再按 StateT 量化。BF16 state 原位更新 UB resident；FP32
state 只在 VF 期间占用 shared scratch，需跨 chunk 时通过 rolling GM 保存。FP32 分支在同一
次 VF 中逐行读取 D，某行最后一次读取完成后才把该行固定低 32 KiB 子区交给 BF16 H 写者，
不做 UB 搬移。

## 同步协议矩阵

### 单 Stage 内部

| 生产者 -> 消费者 | 数据 | 事件/动作 | 复用条件 |
| --- | --- | --- | --- |
| S0 MTE2 -> S0 MTE1 | W/H | `WReady`、`HReady` | MTE1 完成后分别释放 `WFree`、`HFree` |
| S0 MTE1 -> S0 Cube/Fixpipe | W/H 到 L0 | MTE1-to-Cube 依赖 | Fixpipe 写 P 前不读取 P |
| S0 Fixpipe -> S1 VF | P | `PReady` | S1 VF 最后一次读 P 后 `PFree` |
| 首块 H0 MTE3 -> 本 chunk S2 | FP32 无 initial 的 H0 交接 | `LocalDataFree` | S2 获取同一 local data bank 前等待 |
| S1 MTE2 -> S1 VF | U/g/gk_last | MTE2-to-VF 事件 | VF 完成前不得复用 gate/work bank |
| S1 MTE3 -> S2 MTE1 | V_new_g/V_new | `RightReady` | S2 最后一次 MTE1 后 `RightFree` |
| S2 MTE2 -> S2 首个 MTE1 | kg | `kg_ready`，每 slot 每代一次 | 最后一个 mapped head 后 `kg_overwrite_safe` |
| S2 Fixpipe -> S3 VF | D | `DReady` | S3 最后一次读 D 后 `DFree` |
| S3 MTE3 -> 后继消费者 | H/state | `HReady`、`StateToVFree` 或 `StateToMte2Free` | 仅发布有真实消费者的 token |

### 跨 chunk

```text
chunk c:
    S3(h_{c+1} -> L1) --HReady--> chunk c+1 S0(MTE1)
    S3 state MTE3 --StateToVFree--> chunk c+1 S3(VF)
    S1 V_new MTE3 --VNewWorkFree--> chunk c+1 S1(MTE2)
    S2 D 最后读取 --DFree--> chunk c+1 S0/S2 复用 local data bank
```

chunk `c+1` 不能因为 `RunFwdH` 的函数调用已经返回就假设这些异步访问结束；每个消费者
必须等待自己对应的 slot/代际事件。`HReady` 只表示 L1 H 的两路写回完成，不替代 UB state
或 V_new work bank 的 free 事件。

### 跨 head_round

```text
round r 最后一个 chunk 完成：
    若存在 S2：等待每个有效 kg slot 的 kg_overwrite_safe
    若存在 S0：等待每个 W/H slot 的 WFree/HFree，并等待 local data 的 PFree/DFree
    若存在后继 head round：等待 VNewWorkFree 和 state 的下一 owner free
    等待 round r 的 TerminalDrain
    将已经等待完成的 state owner 显式归还 FREE；round r+1 首块标记 roundBoundaryDrained，
    不再对同一个 free token 二次 Wait
    才允许 round r+1 执行 S-1、S0 的 H/W/kg 预取
```

即使 `round r+1` 使用相同的 `kh`，也不能复用 round `r` 的 kg 数据；只复用物理 slot，
并重新读取新的 `(chunk,kh)` payload。`UnionFree` 仅用于下一真实写者是 AIC S0 的特殊
`v_new-only` 路径；下一 round 无 initial 时由同一 AIV 的 V pipe 顺序接管，不生成跨核
无消费者 token。

## 固定 UB/L1 地址合同

地址在 dispatch 初始化时一次确定，Stage 之间只移交 owner，不做 compact、搬移或重新拼接。

| 空间 | 固定区间 | Stage0 owner | Stage1 owner | Stage2 owner | Stage3 owner |
| --- | --- | --- | --- | --- | --- |
| L1 | `[0,64)` | 4 份 `W_c[M,K]`，每份 16 KiB | 空闲 | 空闲 | 空闲 |
| L1 | `[128,256)` | 4 份 `H_c[K,V]`，每份 32 KiB | g-only 为 `V_new_g`，gk-only 为 `V_new` | 同一右操作数 | 非末 chunk 为 `H_{c+1}` |
| L1 | `[256,320)` | `Nkg_round` 份 `kg/k_raw[M,K]`，每份 16 KiB | 继续驻留 | 同一 kg slot | 空闲 |
| UB local data | 每个 AIV `[0,64)`、`[64,128)` | `P`，BF16 16 KiB 或 FP32 32 KiB | 保留 `P`；g-only 右操作数使用固定子区 | `D`，FP32 64 KiB | `D`；FP32 逐行末读后低 32 KiB 改写 H |
| UB V_new work | BF16 `[192,224)`；FP32 `[128,160)` | 空闲 | `U -> V_new` 原位 owner 移交 | 已由 MTE3 释放 | 不使用 |
| UB state | `[128,192)`（BF16）或 `[160,224)`（FP32） | BF16 state 可驻留 | 首 chunk 初始化 BF16 state | 不读写 | 更新 `Rnext`，FP32 只在 VF 期间占用 |
| UB gate | `[224,225)` | 空闲 | g/gk_last 临时输入 | 空闲 | gk-only 使用 `gk_last` |
| UB alpha | `[225,226)` | 空闲 | g-only `alpha` | 保留 | 读取后释放 |

`L1[128,256)` 在每个 Stage 只记录当前 owner：S0 是 H，S1/S2 是右操作数，S3 是 Hnext；
前一 owner 的最后一次 MTE1/MTE3 完成并发布 free 后，下一 owner 才能写入。`L1[256,320)`
的未使用 slot 始终为 `FREE`，不因数组上限而预先写入 16 份 key。

## 细粒度时序伪代码

```text
处理一个 (sequence, head_round, chunk)：
    plan = BuildChunkPlan(headRoundPlan, tiling, sequence, chunk)

    如果 plan.stage0Required：
        S0.PrefetchKgAsync(requiredKh[])       # MTE2 发起后不等待
        对每个 active head：
            S0.LoadH()                         # BF16 initial 本地 MTE2；其余等待 HReady
            S0.LoadW()                         # 清零 tail 后搬入 M 个有效行
            S0.MTE1(W,H) -> MMAD -> Fixpipe(P)
            发布 PReady；释放 W/H；保留 P 到 S1 末次读取

    对每个 active head：
        S1.LoadUAndGate()                      # 等上代 VNewWorkFree
        等待 PReady（无 P 的首 chunk 不等待）
        VF: V_new_fp32 = U - P；转 BF16 并写 v_new GM
        如果 g-only 且存在 S2：同一次 VF 生成 V_new_g/alpha，MTE3 写 L1，发布 RightReady
        如果 gk-only 且存在 S2：MTE3 写 V_new 到 L1，发布 RightReady
        P 末次读和所有相关 MTE3 完成后发布 PFree/VNewWorkFree

    如果 plan.stage2Required：
        对每个 active head：
            确保其 kg slot valid；S0 Loading 时等待 MTE2，未预取时补齐当前 (chunk,kh)
            等待 RightReady 和 local data free
            MTE1(kg,right) -> MMAD -> NoQuant Fixpipe(D)
            发布 DReady；最后一个 mapped head 释放 kg_overwrite_safe

        对每个 active head：
            等待 DReady；准备 BF16 resident 或 FP32 rolling state
            g-only 读取 alpha，gk-only 读取最后有效行 gk[M-1,:]
            一次 VF 完成 Rnext/gate/add；D 末次读后才交出 D 地址
            非末 chunk：MTE3 写 h 和 L1 H，完成后发布 HReady
            最终 chunk 且 output_final_state：MTE3 写 final_state
            按下一实际 owner 发布 state free；释放 D/gate
```

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

因此，每个 `head_round` 需要的 `H_k` 数量就是 `Nkg_round`。调度器在进入该
`head_round` 的 chunk 循环前一次性构造 `requiredKh[]`，并固定四个 `H_v slot` 各自对应的
`kgSlot`；同一 round 的所有 chunk 复用这张映射，只更新对应 chunk 的 kg payload。进入下一
个 round 前先等待上一 round 的 kg 消费完成，再按下一 round 的 `requiredKh[]` 重新预取。

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
        headRoundPlan = BuildHeadRoundPlan(tiling, r)
        # 这里确定 requiredKhCount 和每个 H_v -> kgSlot；它们在本 round 内不随 chunk 变化
        如果 r > 0：
            如果前一个 round 的最终 chunk 执行了 Stage2：
                等待前一个 round 每个有效 kg slot 的 kg_overwrite_safe
            如果前一个 round 的最终 chunk 执行了 Stage0：
                等待前一个 round 每个 active head 的 W_free 和 H_free
            等待前一个 round 的 terminalDrain
            # 所有等待返回前，禁止预取任何 H/W/kg
        如果 initial_state 为 FP32，则对本 round 的 active head 执行一次 S-1
        对 chunk c：
            plan = BuildChunkPlan(headRoundPlan, tiling, sequence[n], c)
            # 这里只绑定 token 范围和 Stage 分支，不重新决定 H_k 数量或槽位映射
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
