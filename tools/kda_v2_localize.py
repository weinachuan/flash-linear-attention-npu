#!/usr/bin/env python3
"""KDA V2 前向偶发漂移：一次性跑完所有定位判据（离线取证 + 现场复现 + 对照）。

用法（在 A3 现场、装了出问题那版 fla_npu 的环境里）：

    python3 kda_v2_localize.py --dump dumps/bak-drift_v2_inv2426_vs_2435.pt

可选：--n-stress 300（现场复现的 trial 数）、--n-scan 100（每个对照的 trial 数）、
--skip-npu（只跑离线部分）。脚本只打印摘要，不会落盘大文件。
"""

from __future__ import annotations

import argparse
import time

import torch

KEYS = ["o", "final_state", "gk", "Aqk", "Akk", "w", "u", "qg", "kg", "v_new", "h"]
DEP_ORDER = ["gk", "Aqk", "Akk", "w", "u", "v_new", "h", "o", "final_state"]
CHUNK = 64


def log(*a):
    print(*a, flush=True)


def fp(t):
    tf = t.detach().float()
    return (float(tf.sum()), float(tf.abs().sum()), float((tf * tf).sum()),
            int(torch.count_nonzero(t)))


def bf(x):
    return x.detach().float()


def diff_summary(a, b, name):
    if a.shape != b.shape:
        return f"{name}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"
    d = (a != b)
    n = int(d.sum())
    if n == 0:
        return f"{name}: identical"
    idx = torch.nonzero(d)
    d0 = idx[:, 0]
    uniq = torch.unique(d0)
    per = torch.bincount(d0, minlength=a.shape[0])[uniq]
    mx = float((bf(a)[d] - bf(b)[d]).abs().max())
    return (f"{name}: {n}/{a.numel()} ({100.0 * n / a.numel():.4f}%) "
            f"dim0_hits={uniq.numel()} dim0=[{int(uniq.min())}..{int(uniq.max())}] "
            f"per_dim0=[{int(per.min())}..{int(per.max())}] max|diff|={mx:.4g}")


def region_stats(x):
    if x.numel() == 0:
        return "empty"
    v = bf(x)
    return (f"exact0={float((x == 0).float().mean()):.4f} "
            f"|x|<1e-6={float((v.abs() < 1e-6).float().mean()):.4f} "
            f"|x|>1={float((v.abs() > 1).float().mean()):.4f} "
            f"max={float(v.abs().max()):.4g}")


def chunk_map(cu, chunk=CHUNK):
    """(seq, chunk_in_seq) -> 全局 chunk 序号（与 Prepare/FwdH 的 chunk-major 编号一致）。"""
    table = []
    ordinal = 0
    for s in range(len(cu) - 1):
        n = (cu[s + 1] - cu[s] + chunk - 1) // chunk
        for c in range(n):
            table.append((s, c, ordinal, cu[s] + c * chunk))
            ordinal += 1
    return table


def locate_token(tok, cu, chunk=CHUNK):
    for s in range(len(cu) - 1):
        if cu[s] <= tok < cu[s + 1]:
            return s, (tok - cu[s]) // chunk, cu[s + 1] - cu[s]
    return None, None, None


def head_token_axes(key):
    """各公开输出的 (head 轴, token 轴)。o 是 (T,H,V)，其余 head-major 张量是 (H,T,...)。"""
    if key == "o":
        return 1, 0
    if key in ("gk", "w", "u", "qg", "kg", "v_new", "Aqk", "Akk"):
        return 0, 1
    return None


def copy_hunt(key, fwd_out, rec_out, inputs, skip=()):
    """坏区是不是「别人的数据」：同调用内各 tensor 的同位置内容 + 输入 + 同张量平移。

    如果坏值与某个张量在某个固定映射下逐位相同 ⇒ 像"脏写/错误搬运"；
    如果哪里都匹配不上、且以 0 为主 ⇒ 像"读到没人写过的地方"。
    """
    rec_t = rec_out[key]
    mask = (fwd_out[key] != rec_t)
    rec_flat = rec_t.reshape(-1)
    idx = torch.nonzero(mask.reshape(-1)).reshape(-1)
    region = rec_flat[idx]
    hits = []
    for name, t in list(rec_out.items()) + list(fwd_out.items()):
        if name == key:
            continue  # 自己比自己必然全等，跳过
        if any(t is s for s in skip):
            continue  # 坏张量自身/其参考值不参与匹配
        if t is None or t.numel() != rec_flat.numel():
            continue
        c = int((t.reshape(-1)[idx] == region).sum())
        if c:
            hits.append((c, f"same-position:{name}"))
    for name, t in inputs.items():
        if t is None or not torch.is_tensor(t) or t.numel() != rec_flat.numel():
            continue
        c = int((t.reshape(-1)[idx] == region).sum())
        if c:
            hits.append((c, f"same-position:in:{name}"))
    # 同张量平移：看坏区是不是"另一段自己的数据"
    step = rec_t.shape[-1]
    for shift_tokens in (-128, -64, 64, 128, 256):
        shifted = rec_flat.roll(shift_tokens * step)
        c = int((shifted[idx] == region).sum())
        if c:
            hits.append((c, f"self-shift:{shift_tokens}tok"))
    hits.sort(reverse=True)
    return hits[:5]


def akk_probe(u_good, u_bad, akk, v_in, beta_in):
    """用 Akk 反解：rec 的 u 是否「用正确操作数算得出来」。

    返回一行结论字符串；异常时返回原因。
    """
    try:
        idx = torch.nonzero(u_good != u_bad)
        head = int(torch.unique(idx[:, 0]).min())
        tok = int(torch.unique(idx[:, 1]).min())
        block = 64
        A = bf(akk[head, tok:tok + block]).double()
        ug = bf(u_good[head, tok:tok + block]).double()
        ub = bf(u_bad[head, tok:tok + block]).double()
        vv = bf(v_in[head, tok:tok + block]).double()
        bt = bf(beta_in[head, tok:tok + block]).double()[:, None]
        Vu, *_ = torch.linalg.lstsq(A, ug)
        pred = A @ Vu
        res_f = float((pred - ug).norm() / ug.norm())
        res_r = float((pred - ub).norm() / ub.norm())
        corr_f = float(torch.corrcoef(torch.stack([pred.reshape(-1), ug.reshape(-1)]))[0, 1])
        corr_r = float(torch.corrcoef(torch.stack([pred.reshape(-1), ub.reshape(-1)]))[0, 1])
        corr_v = float(torch.corrcoef(torch.stack([Vu.reshape(-1), (vv * bt).reshape(-1)]))[0, 1])
        tag = "操作数侧/交接侧" if res_r > 1e-3 else "写回/搬运侧"
        return (f"      Akk 反解: V_beta vs v*beta corr={corr_v:.4f} | "
                f"rel_res(好侧)={res_f:.3e} corr={corr_f:.4f} | "
                f"rel_res(坏侧)={res_r:.3e} corr={corr_r:.4f} => {tag}")
    except Exception as exc:  # noqa: BLE001
        return f"      Akk 反解跳过（{exc}）"


def localize_tensor(key, good_val, bad_val, cu):
    """打印某个 tensor 上"好/坏"差异的落点与幅度摘要。"""
    out = []
    dd = (good_val != bad_val)
    idx = torch.nonzero(dd)
    ax = head_token_axes(key)
    if ax is not None:
        ha, ta = ax
        heads = torch.unique(idx[:, ha]).tolist()
        toks = torch.unique(idx[:, ta])
        seq, cin, slen = locate_token(int(toks.min()), cu)
        ordn = [o for (sq, c, o, t0) in chunk_map(cu) if sq == seq and c == cin]
        out.append(f"      {key}: heads={heads} tokens=[{int(toks.min())}..{int(toks.max())}] "
                   f"-> seq={seq}(len={slen}) chunk_in_seq={cin} "
                   f"global_chunk={ordn[0] if ordn else None}")
    else:
        out.append(f"      {key}: dim0={torch.unique(idx[:, 0])[:6].tolist()} "
                   f"dim1={torch.unique(idx[:, 1]).tolist()}")
    out.append(f"      ndiff={int(dd.sum())} 坏侧 {region_stats(bad_val[dd])} | "
               f"好侧 {region_stats(good_val[dd])}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--n-stress", type=int, default=2000)
    ap.add_argument("--n-scan", type=int, default=200)
    ap.add_argument("--max-samples", type=int, default=5,
                    help="最多收集多少个漂移样本的落点信息（每个样本只记摘要，不落盘）")
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--skip-npu", action="store_true")
    ap.add_argument("--with-driver-controls", action="store_true",
                    help="额外跑 [3]/[4] 两组驱动方式对照（用例构造与 [2] 完全相同，只改并发/同步）")
    ap.add_argument("--with-tnd-control", action="store_true",
                    help="额外跑 [5] TND 拼写对照。注意：那不是本 issue 的用例构造，默认不跑")
    args = ap.parse_args()

    log("=" * 78)
    log("[0] 装载 dump")
    d = torch.load(args.dump, map_location="cpu", weights_only=False)
    s = d["scalars"]
    cu = list(s["cu_seqlens"])
    fwd_in = d.get("fwd_inputs") or d["inputs"]
    rec_in = d.get("recompute_inputs") or fwd_in
    fwd_out = d["fwd_outputs"]
    rec_out = d["recompute_outputs"]
    log(f"    kind={d.get('kind')} fwd_occ={d.get('fwd_occ')} recompute_occ={d.get('recompute_occ')}")
    log(f"    layout={s['layout']} chunk={s['chunk_size']} cu_seqlens={cu} "
        f"safe_gate={s['safe_gate']} use_gate_in_kernel={s['use_gate_in_kernel']} "
        f"use_exp2={s['use_exp2']}")
    same_in = all(torch.equal(fwd_in[k], rec_in[k]) for k in fwd_in if torch.is_tensor(fwd_in[k]))
    log(f"    inputs bitwise identical (fwd vs recompute): {same_in}")

    log("=" * 78)
    log("[1] 每个输出的差异概览")
    for k in KEYS:
        log("    " + diff_summary(fwd_out[k], rec_out[k], k))

    log("[1b] 依赖顺序上最早漂移的 tensor（注入点）")
    earliest = None
    for k in DEP_ORDER:
        if not torch.equal(fwd_out[k], rec_out[k]):
            earliest = k
            break
    log(f"    earliest drifting tensor = {earliest}")

    drifted = [k for k in KEYS if not torch.equal(fwd_out[k], rec_out[k])]
    log("[1c] 坏区性质（rec 侧）")
    for k in drifted:
        a, b = fwd_out[k], rec_out[k]
        m = (a != b)
        log(f"    {k:>11}: rec {region_stats(b[m])} | fwd {region_stats(a[m])}")

    log("[1d] 坏区是不是「别人的数据」（同调用各 tensor 同位置 + 输入 + 同张量平移）")
    for k in drifted:
        n_bad = int((fwd_out[k] != rec_out[k]).sum())
        hits = copy_hunt(k, fwd_out, rec_out, fwd_in)
        frac = (hits[0][0] / max(1, n_bad)) if hits else 0.0
        verdict = ("疑似抄了别的张量/错位搬运" if frac > 0.5
                   else "匹配不上 ⇒ 像读到没人写过的地方（不是抄来的数据）")
        log(f"    {k:>11}: 坏元素={n_bad} 最佳匹配={hits[0] if hits else None} "
            f"best_frac={frac:.4f}  ({verdict})")

    log("[1e] 落点到 work item / chunk 的映射")
    for k in drifted:
        a, b = fwd_out[k], rec_out[k]
        idx = torch.nonzero(a != b)
        ax = head_token_axes(k)
        if ax is not None:
            ha, ta = ax
            heads = torch.unique(idx[:, ha]).tolist()
            toks = torch.unique(idx[:, ta])
            seq, cin, slen = locate_token(int(toks.min()), cu)
            ordn = [o for (sq, c, o, t0) in chunk_map(cu) if sq == seq and c == cin]
            log(f"    {k:>11}: heads={heads} tokens=[{int(toks.min())}..{int(toks.max())}] "
                f"-> seq={seq} (len={slen}) chunk_in_seq={cin} "
                f"global_chunk={ordn[0] if ordn else None}")
        else:
            d0 = torch.unique(idx[:, 0])
            log(f"    {k:>11}: dim0={d0[:6].tolist()} (共 {int(d0.numel())} 个) "
                f"dim1={torch.unique(idx[:, 1]).tolist()}  "
                f"（h=每个 chunk 的 state：dim0=global_chunk；final_state=序列号）")

    log("[1f] Akk 反解：rec 的 u 是否与「正确操作数」的 GEMM 一致")
    if "u" in drifted:
        log(akk_probe(fwd_out["u"], rec_out["u"], fwd_out["Akk"],
                      fwd_in["v_in"], fwd_in["beta_in"]))
    else:
        log("    (本 dump 的 u 没有漂移，跳过)")

    if args.skip_npu:
        log("=" * 78)
        log("[done] --skip-npu：只做了离线取证")
        return

    torch.npu.set_device(args.device)
    from fla_npu.ops.ascendc import npu_chunk_kda_fwd

    layout = str(s["layout"]).upper()
    ins = {k: (v.to("npu") if torch.is_tensor(v) else v) for k, v in fwd_in.items()}
    log("=" * 78)
    log("[2] 现场复现：用例构造严格取自 issue 包（不转置、不合成、不改参数）")
    log("    输入 = dump 的 fwd_inputs（与 recompute_inputs 逐位相同），落到 NPU 后原样使用：")
    for k in ("q_in", "k_in", "v_in", "g_in", "beta_in", "A_log", "dt_bias"):
        v = ins[k]
        log(f"      {k:>10}: shape={tuple(v.shape)} dtype={v.dtype} "
            f"contig={v.is_contiguous() if hasattr(v, 'is_contiguous') else True}")
    log("    参数 = dump 的 scalars，调用形式与包内 kda_stress_multistream.py 完全一致：")
    log(f"      npu_chunk_kda_fwd(q,k,v,g,beta, {s['scale']}, {s['chunk_size']}, "
        f"layout='{layout}', initial_state=None, output_final_state=True, "
        f"cu_seqlens={cu}, chunk_indices=None, safe_gate={s['safe_gate']}, "
        f"lower_bound={s['lower_bound']}, use_gate_in_kernel={s['use_gate_in_kernel']}, "
        f"A_log=..., dt_bias=..., disable_recompute=True, use_exp2={s['use_exp2']})")

    def call(inputs):
        return npu_chunk_kda_fwd(
            inputs["q_in"], inputs["k_in"], inputs["v_in"], inputs["g_in"], inputs["beta_in"],
            s["scale"], s["chunk_size"], layout=layout, initial_state=None,
            output_final_state=True, cu_seqlens=cu, chunk_indices=None,
            safe_gate=s["safe_gate"], lower_bound=s["lower_bound"],
            use_gate_in_kernel=s["use_gate_in_kernel"], A_log=inputs["A_log"],
            dt_bias=inputs["dt_bias"], disable_recompute=True, use_exp2=s["use_exp2"])

    def run_trials(n_trial, n_streams, sync_per_call=False):
        """与包内多流脚本同结构：每 trial 在 n_streams 个 stream 上各调一次，再 sync。"""
        sts = [torch.npu.Stream() for _ in range(n_streams)]
        hist = [dict() for _ in range(n_streams)]
        first_bad = None
        t_bad = None
        samples = []
        trials_run = 0
        for i in range(n_trial):
            outs = []
            for st in sts:
                with torch.npu.stream(st):
                    outs.append(call(ins))
                if sync_per_call:
                    torch.npu.synchronize()
            torch.npu.synchronize()
            per = {k: [] for k in KEYS}
            for si, o in enumerate(outs):
                fps = {}
                for k, v in zip(KEYS, o):
                    if v is None:
                        continue
                    fps[k] = fp(v)
                    per[k].append(fps[k])
                hist[si][i] = fps
            bad = [k for k in KEYS if len(set(per[k])) > 1]
            if bad:
                # 与 warm 比，判出"坏的一侧"（哪个 stream 的输出与 warm 不一致）
                bad_side = {}
                for k in KEYS:
                    if k not in golden:
                        continue
                    g = fp(golden[k])
                    for si, o in enumerate(outs):
                        v = o[KEYS.index(k)]
                        if v is not None and fp(v) != g:
                            bad_side.setdefault(k, (si, v))
                            break
                target = next((k for k in DEP_ORDER if k in bad_side), None)
                loc = None
                if target is not None:
                    si, bad_val = bad_side[target]
                    idx = torch.nonzero(golden[target] != bad_val)
                    ax = head_token_axes(target)
                    if ax is not None:
                        ha, ta = ax
                        heads = torch.unique(idx[:, ha]).tolist()
                        toks = torch.unique(idx[:, ta])
                        seq, cin, slen = locate_token(int(toks.min()), cu)
                        ordn = [o for (sq, c, o, t0) in chunk_map(cu) if sq == seq and c == cin]
                        loc = (target, heads, int(toks.min()), int(toks.max()), seq, cin,
                               ordn[0] if ordn else None)
                    else:
                        loc = (target, None, int(torch.unique(idx[:, 0]).min()),
                               int(torch.unique(idx[:, 0]).max()), None, None, None)
                samples.append({"trial": i, "bad": sorted(bad_side), "target": target,
                                "loc": loc, "bad_side": bad_side})
                if first_bad is None:
                    first_bad, t_bad = outs, i
                if len(samples) >= args.max_samples:
                    break
            trials_run = i + 1
        return hist, first_bad, t_bad, samples, trials_run

    warm = call(ins)
    torch.npu.synchronize()
    golden = {k: v.clone() for k, v in zip(KEYS, warm) if v is not None}
    log(f"[2] 运行 {args.streams} stream × {args.n_stress} trial")
    log("    warm fp: " + ", ".join(f"{k}={fp(v)[0]:.4g}" for k, v in golden.items()))
    t0 = time.time()
    hist, first_bad, t_bad, samples, trials_run = run_trials(args.n_stress, args.streams)

    log("    [每个 stream 内部一致性]")
    inner_bad = False
    for si in range(args.streams):
        fps_all = [tuple(sorted(h.items())) for h in hist[si].values() if h]
        if len(set(fps_all)) > 1:
            n_uniq = len(set(fps_all))
            keys = [k for k in KEYS
                    if len({h[k] for h in hist[si].values() if k in h}) > 1]
            log(f"      ✗ stream {si}: {n_uniq} unique fps（{keys}）")
            inner_bad = True
    if not inner_bad:
        log(f"      ✓ 每个 stream 内部 {args.n_stress} trial 一致")

    log("    [跨 stream 一致性]")
    cross_bad = []
    for k in KEYS:
        trials = []
        for i in range(args.n_stress):
            vals = [hist[si][i].get(k) for si in range(args.streams) if i in hist[si]]
            vals = [v for v in vals if v is not None]
            if len(set(vals)) > 1:
                trials.append(i)
        if trials:
            cross_bad.append((k, trials))
            log(f"      ✗ {k}: 在 {len(trials)} 个 trial 上跨 stream 不一致 "
                f"(e.g., trial {trials[:5]})")
    if not cross_bad:
        log(f"      ✓ 跨 stream {args.n_stress} trial 一致")
    log(f"    漂移率: {len(samples)}/{trials_run} trial（实际跑过的 trial 数）, "
        f"用时 {time.time() - t0:.0f}s")

    if samples:
        log("    [样本落点表]（用于判断是不是每次都在同一个 work item）")
        log("      trial | 最早漂移 | head | tokens | seq | chunk_in_seq | global_chunk | 同调用下游")
        for s in samples:
            loc = s["loc"]
            if loc is None:
                log(f"      {s['trial']:>5} | {str(s['target']):>8} | (warm 自身偶发，无坏侧)")
                continue
            tgt, heads, tlo, thi, seq, cin, glob = loc
            down = [k for k in DEP_ORDER if k != tgt and k in s["bad"]]
            log(f"      {s['trial']:>5} | {tgt:>8} | {heads} | [{tlo}..{thi}] | {seq} | {cin} "
                f"| {glob} | {down}")
        glob_hist = {}
        head_hist = {}
        for s in samples:
            loc = s["loc"]
            if loc is None:
                continue
            glob_hist[loc[6]] = glob_hist.get(loc[6], 0) + 1
            key = tuple(loc[1]) if loc[1] else None
            head_hist[key] = head_hist.get(key, 0) + 1
        log(f"      global_chunk 直方图: {glob_hist}")
        log(f"      head 直方图: {head_hist}")

    if first_bad is not None:
        log(f"    [首个漂移 trial={t_bad} 的现场快照（与 warm 比，判「坏的一侧」）]")
        bad_info = {}
        for k in KEYS:
            if k not in golden:
                continue
            g = fp(golden[k])
            for si, o in enumerate(first_bad):
                v = o[KEYS.index(k)]
                if v is not None and fp(v) != g:
                    bad_info.setdefault(k, (si, v))
                    break
        log(f"      与 warm 不一致的 tensor: {sorted(bad_info)}")
        target = next((k for k in DEP_ORDER if k in bad_info), None)
        if target is None:
            log("      本次样本的每个输出都能在某个 stream 上等于 warm（疑似 warm 自身偶发），跳过取证")
        else:
            si, bad_val = bad_info[target]
            good_val = golden[target]
            log(f"      最早漂移点 = {target}（出现在 stream {si}）")
            for line in localize_tensor(target, good_val, bad_val, cu):
                log(line)
            fo = {k: v for k, v in golden.items()}
            ro = {target: bad_val}
            for j, o in enumerate(first_bad):
                for k2 in KEYS:
                    v = o[KEYS.index(k2)]
                    if v is not None:
                        ro[f"s{j}:{k2}"] = v
            hits = copy_hunt(target, fo, ro, ins)
            n_bad = int((good_val != bad_val).sum())
            frac = (hits[0][0] / max(1, n_bad)) if hits else 0.0
            log(f"      脏写检查: 最佳匹配={hits[0] if hits else None} best_frac={frac:.4f} "
                f"({ '疑似抄了别的张量/错位搬运' if frac > 0.5 else '匹配不上 ⇒ 像读到没人写过的地方' })")
            if target == "u":
                log(akk_probe(good_val, bad_val, golden["Akk"], ins["v_in"], ins["beta_in"]))
            downstream = [k for k in DEP_ORDER if k != target and k in bad_info]
            log(f"      同一次调用里下游也漂: {downstream}")
    else:
        log(f"    [首个漂移 trial] 跑满 {args.n_stress} trial 没抓到（加大 --n-stress 再试）")

    if args.with_driver_controls:
        log("=" * 78)
        log("[3] 驱动方式对照·并发度扫描（用例构造与 [2] 完全相同，只改并发数）")
        for ns in (1, 2, 4, 8):
            hist_c, bad_c, _, _, _ = run_trials(args.n_scan, ns)
            hit = sum(1 for i in range(args.n_scan)
                      if any(len({hist_c[si][i].get(k) for si in range(ns)
                                  if i in hist_c[si] and k in hist_c[si][i]}) > 1 for k in KEYS))
            log(f"      streams={ns}: 漂移 trial={hit}/{args.n_scan}")

        log("=" * 78)
        log("[4] 驱动方式对照·是否必须「两次调用同时在飞」（4 stream，每次调用后 sync）")
        hist_d, bad_d, _, _, _ = run_trials(args.n_scan, 4, sync_per_call=True)
        hit = sum(1 for i in range(args.n_scan)
                  if any(len({hist_d[si][i].get(k) for si in range(4)
                              if i in hist_d[si] and k in hist_d[si][i]}) > 1 for k in KEYS))
        log(f"      漂移 trial={hit}/{args.n_scan}（0 ⇒ 触发条件是『调用时间重叠』）")

    if args.with_tnd_control:
        log("=" * 78)
        log("[5] 注意：这不是 issue 的用例构造（改用 TND 拼写），仅作对照")
        ins_t = {k: (v.transpose(0, 1).contiguous() if torch.is_tensor(v) and v.dim() >= 2
                     and k in ("q_in", "k_in", "v_in", "g_in", "beta_in") else v)
                 for k, v in ins.items()}
        layout_t = "TND"
        try:
            sts = [torch.npu.Stream() for _ in range(4)]
            hit = 0
            for i in range(args.n_scan):
                outs = []
                for st in sts:
                    with torch.npu.stream(st):
                        outs.append(npu_chunk_kda_fwd(
                            ins_t["q_in"], ins_t["k_in"], ins_t["v_in"], ins_t["g_in"],
                            ins_t["beta_in"], s["scale"], s["chunk_size"], layout=layout_t,
                            initial_state=None, output_final_state=True, cu_seqlens=cu,
                            chunk_indices=None, safe_gate=s["safe_gate"],
                            lower_bound=s["lower_bound"],
                            use_gate_in_kernel=s["use_gate_in_kernel"],
                            A_log=ins_t["A_log"], dt_bias=ins_t["dt_bias"],
                            disable_recompute=True, use_exp2=s["use_exp2"]))
                torch.npu.synchronize()
                per = {k: [] for k in KEYS}
                for o in outs:
                    for k, v in zip(KEYS, o):
                        if v is not None:
                            per[k].append(fp(v))
                if any(len(set(per[k])) > 1 for k in KEYS):
                    hit += 1
            log(f"      TND 漂移 trial={hit}/{args.n_scan}")
        except Exception as exc:
            log(f"      TND 对照失败: {exc}")

    log("=" * 78)
    log("[done] 把上面的完整输出贴回来即可")


if __name__ == "__main__":
    main()
