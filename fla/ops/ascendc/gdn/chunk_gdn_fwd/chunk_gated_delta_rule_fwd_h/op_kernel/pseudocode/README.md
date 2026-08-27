# FwdH pseudocode blueprint

This directory is an implementation blueprint for the A5 FwdH redesign. It mirrors the
existing `op_host`, `op_kernel/gemm`, `op_kernel/epilogue`, and fast-launch entry layout used by
the earlier PR, but it is deliberately isolated under `op_kernel/pseudocode/` and is excluded from
all build targets. It does not copy PR370's scheduling or memory behavior.

The source of truth is `gdn-fwd-h-ascendc-design.md`. The pseudocode uses the design's exact
terms: `kg` is lower-case, `V_new` is retained as the public name, and `state_v_first` is a native
kernel attribute rather than an external transpose.

## Module contracts

| Module | Responsibility | Inputs | Outputs | Lifetime / control |
| --- | --- | --- | --- | --- |
| `fwd_h_pseudocode_host.h` | Validate API and infer `StateT`, gate mode, varlen spans, and output shapes | API tensors, attrs, optional arrays | `TilingPlan`, output descriptors, error | Runs before any kernel launch; rejects invalid presence/shape/dtype combinations |
| `fwd_h_pseudocode_round_planner.h` | Build one `(sequence, round, chunk)` plan and exact current-round key set | `TilingPlan`, sequence/chunk id | `HeadBinding[<=4]`, `requiredKh[]`, `kg_binding[]` | Rebuilt for every chunk; no whole-sequence `kg` reservation |
| `fwd_h_pseudocode_memory.h` | Track fixed L1/UB addresses and single-owner transitions | slot id, stage owner | ready/free state for H/W/right/kg | H/W/right are released at their last consumer; `kg` is released after last S2 MTE1 |
| `fwd_h_pseudocode_sync.h` | Model ready/free/terminal events and the cross-round barrier | producer, consumer, slot, generation | matching `Wait`/`Set`/`Release` operations | No unconsumed token; next round cannot prefetch before barrier return |
| `fwd_h_pseudocode_s_minus_one.h` | Convert FP32 initial state once per current head round | FP32 `initial_state`, layout attr, active head range | BF16 canonical H0 in L1 and h0 GM | Runs after the previous-round barrier and drains before S0; omitted for BF16/no initial |
| `fwd_h_pseudocode_stage0.h` | Optional S0: load H/W, asynchronously prefetch current `kg`, compute `P = W @ H` | current round plan, GM k/w/state | L1 kg/H/W, UB `P` | `kg` prefetch may overlap S0 but is not consumed until S2; no S0 for first no-initial chunk |
| `fwd_h_pseudocode_stage1.h` | S1 full-head VF: `V_new_fp32`, BF16 `V_new`, optional `V_new_g`, `alpha` | `u`, optional `P`, g/gk mode | public `v_new`, L1 right operand, alpha, h0 | `V_new`/`V_new_g` live through S2 MTE1; final no-final-state path is `v_new-only` |
| `fwd_h_pseudocode_stage2.h` | Pair each `H_v` with its mapped `kg` and compute `D` | current-round kg slots, S1 right operand | UB FP32 `D` | A kg slot is shared only by equal `kh` within this round; it expires before next round |
| `fwd_h_pseudocode_stage3.h` | Update rolling state and write layout-aware h/final state | `D`, current rolling state, alpha or `gk_last` | next H resident, h GM, final_state GM | Internal state always `[K,V]`; GM offset uses `state_v_first` |
| `fwd_h_pseudocode_scheduler.h` | Enforce `sequence -> head_round -> chunk`, S-1, barriers, and final branch | context and plans | completed output tensors | Before each next head round waits kg overwrite-safe, W/H free, terminal drain |
| `chunk_gated_delta_rule_fwd_h_pseudocode.cpp` | Fast-launch-shaped adapter and kernel entry | framework tensors/attrs | tuple `(h, v_new, final_state)` | Structural adapter only; no external L2 transpose |

## Exact round binding

For g-only:

```text
groupSize = HV / HK
kh(hv) = floor(hv / groupSize)
```

For gk-only, `kh(hv) = hv`; the input `k` is already prepared `kg`, so different value heads
never share an entry. For either mode:

```text
active_hv_round(r) = [4*r, min(4*(r+1), HV))
required_hk_round(r) = unique({kh(hv) for hv in active_hv_round(r)})
Nkg_round = len(required_hk_round(r))
```

Each `HeadBinding` stores the complete relation:

```text
head.hv      -> value-head input/state/right operand
head.kh      -> physical key head
head.kgSlot  -> current round kg payload
head.hSlot   -> current H/right L1 slot
head.wSlot   -> current W L1 slot
head.aiv/localSlot -> AIV assignment
```

Concrete g-only bindings are therefore:

```text
HK=1, HV=3, round 0:
  roundHead  hv  kh  kgSlot  right operand
      0       0   0     0    V_new_g[hv=0]
      1       1   0     0    V_new_g[hv=1]
      2       2   0     0    V_new_g[hv=2]

HK=2, HV=4, round 0:
  roundHead  hv  kh  kgSlot  right operand
      0       0   0     0    V_new_g[hv=0]
      1       1   0     0    V_new_g[hv=1]
      2       2   1     1    V_new_g[hv=2]
      3       3   1     1    V_new_g[hv=3]
```

Stage2 performs one MMAD per row of this table. The two rows sharing a `kh` reuse the same
current-round `kgSlot`, but still use different H/value-head right operands. In gk-only, replace
`kh` with `hv`, so every row receives a distinct `kgSlot`.

Stage2 uses `(kg[head.kgSlot], right[head.hSlot])` for that exact `head.hv`. Thus `HK:HV=1:3`
creates three head bindings and one kg slot, `1:2` creates four bindings and two kg slots, and
`1:6` creates two rounds with one kg slot each; the second round reloads kg after the first round
has drained.

The fixed L1 key area is `[256, 320)` KiB: four physical slots at 16 KiB each. The old idea of
keeping 16 copies would consume `16 * 16 KiB = 256 KiB` by itself, while the design also needs
64 KiB for W and 128 KiB for H/resident data. That would require at least 448 KiB and cannot fit
the 320 KiB L1 plan. The algorithm therefore never creates a 16-slot table: it creates exactly
`Nkg_round` slots for the current round, releases them after the last S2 MTE1 consumer, and
reloads the same `kh` in the next round when necessary.

## Time-control skeleton

```text
for sequence n:
    for head_round r:
        if r > 0:
            wait previous kg_overwrite_safe for every valid kg slot
            wait previous W_free and H_free for every active head
            wait previous terminalDrain
            # no H/W/kg prefetch is permitted before all waits return
        S-1 once for this round's active heads if initial_state is FP32
        for chunk c:
            plan = BuildRoundPlan(n, r, c)
            if plan.stage0Required:
                Stage0: async prefetch exactly Nkg_round kg slots; compute P
            Stage1: compute V_new (and V_new_g/alpha only when S2 needs them)
            if plan.stage2Required:
                Stage2: wait kg/right, MMAD D, release kg/right at last MTE1
                Stage3: update state, write h/final_state, publish next H ready
            else:
                # final chunk + output_final_state=false: no kg, D, state update, or final GM write
```

`state_v_first=false` maps logical state `[k,v]` to `base + k*V + v`; `true` maps it to
`base + v*K + k`. All L1/UB residents remain canonical `[K,V]`, and no external transpose is
part of the entry or scheduler.
