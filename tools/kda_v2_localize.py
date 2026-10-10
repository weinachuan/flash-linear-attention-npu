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


def copy_hunt(key, fwd_out, rec_out, inputs):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--n-stress", type=int, default=300)
    ap.add_argument("--n-scan", type=int, default=100)
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--skip-npu", action="store_true")
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
    try:
        idx = torch.nonzero(fwd_out["u"] != rec_out["u"])
        head = int(torch.unique(idx[:, 0]).min())      # u 是 (H, T, V)
        tok = int(torch.unique(idx[:, 1]).min())
        block = 64
        A = bf(fwd_out["Akk"][head, tok:tok + block]).double()
        u_f = bf(fwd_out["u"][head, tok:tok + block]).double()
        u_r = bf(rec_out["u"][head, tok:tok + block]).double()
        v_in = bf(fwd_in["v_in"][head, tok:tok + block]).double()
        beta = bf(fwd_in["beta_in"][head, tok:tok + block]).double()[:, None]
        Vu, *_ = torch.linalg.lstsq(A, u_f)
        pred = A @ Vu
        res_f = float((pred - u_f).norm() / u_f.norm())
        res_r = float((pred - u_r).norm() / u_r.norm())
        corr_f = float(torch.corrcoef(torch.stack([pred.reshape(-1), u_f.reshape(-1)]))[0, 1])
        corr_r = float(torch.corrcoef(torch.stack([pred.reshape(-1), u_r.reshape(-1)]))[0, 1])
        corr_v = float(torch.corrcoef(torch.stack([Vu.reshape(-1), (v_in * beta).reshape(-1)]))[0, 1])
        log(f"    pred = Akk @ V_beta_true; V_beta 反解与 v*beta 相关={corr_v:.4f}")
        log(f"    rel_res(fwd u)={res_f:.3e} corr={corr_f:.4f} | rel_res(rec u)={res_r:.3e} corr={corr_r:.4f}")
        log("    => " + ("rec 的 u 不是用正确操作数算出来的（操作数侧/交接侧）"
                        if res_r > 1e-3 else "rec 的 u 与正确操作数一致（写回/搬运侧）"))
    except Exception as exc:
        log(f"    (跳过: {exc})")

    if args.skip_npu:
        log("=" * 78)
        log("[done] --skip-npu：只做了离线取证")
        return

    torch.npu.set_device(args.device)
    from fla_npu.ops.ascendc import npu_chunk_kda_fwd

    ins = {k: (v.to("npu") if torch.is_tensor(v) else v) for k, v in fwd_in.items()}
    layout = str(s["layout"]).upper()

    def call(lay, inputs):
        return npu_chunk_kda_fwd(
            inputs["q_in"], inputs["k_in"], inputs["v_in"], inputs["g_in"], inputs["beta_in"],
            s["scale"], s["chunk_size"], layout=lay, initial_state=None,
            output_final_state=True, cu_seqlens=cu, chunk_indices=None,
            safe_gate=s["safe_gate"], lower_bound=s["lower_bound"],
            use_gate_in_kernel=s["use_gate_in_kernel"], A_log=inputs["A_log"],
            dt_bias=inputs["dt_bias"], disable_recompute=True, use_exp2=s["use_exp2"])

    def tnd_inputs(inputs):
        return {k: (v.transpose(0, 1).contiguous() if torch.is_tensor(v) and v.dim() >= 2
                    and k in ("q_in", "k_in", "v_in", "g_in", "beta_in") else v)
                for k, v in inputs.items()}

    log("=" * 78)
    log(f"[2] 现场复现（{layout}，{args.streams} stream × {args.n_stress} trial）")
    warm = call(layout, ins)
    torch.npu.synchronize()
    golden = {k: v.clone() for k, v in zip(KEYS, warm) if v is not None}
    log("    warm fp: " + ", ".join(f"{k}={fp(v)[0]:.4g}" for k, v in golden.items()))

    streams = [torch.npu.Stream() for _ in range(args.streams)]
    uniq = {k: set() for k in KEYS}
    cross = {k: 0 for k in KEYS}
    first_sample = None
    first_trial = None
    t0 = time.time()
    for i in range(args.n_stress):
        outs = []
        for st in streams:
            with torch.npu.stream(st):
                outs.append(call(layout, ins))
        torch.npu.synchronize()
        per = {k: [] for k in KEYS}
        for o in outs:
            for k, v in zip(KEYS, o):
                if v is None:
                    continue
                per[k].append(fp(v))
                uniq[k].add(fp(v))
        bad = [k for k in KEYS if len(set(per[k])) > 1]
        if bad and first_sample is None:
            first_sample = outs
            first_trial = i
        for k in bad:
            cross[k] += 1
        if (i + 1) % 50 == 0:
            log(f"    [{i+1}/{args.n_stress}] cross_drift={ {k: cross[k] for k in KEYS if cross[k]} } "
                f"{time.time() - t0:.0f}s")
    hit_desc = ", ".join(f"{k}={cross[k]}" for k in KEYS if cross[k]) or "无"
    log(f"    结果: trial 总数={args.n_stress}, 跨 stream 漂移 trial: {hit_desc}")
    log("    每 tensor unique_fp: " + ", ".join(f"{k}={len(uniq[k])}" for k in KEYS))
    if first_trial is not None:
        log(f"    首个跨 stream 漂移 trial = {first_trial}")
        inconsistent = []
        for k in DEP_ORDER:
            vals = [o[KEYS.index(k)] for o in first_sample]
            if any(v is None for v in vals):
                continue
            if len({tuple(fp(v)) for v in vals}) > 1:
                inconsistent.append(k)
        log(f"    该 trial 上跨 stream 不一致的 tensor: {inconsistent}")
        for k in inconsistent:
            vals = [o[KEYS.index(k)] for o in first_sample]
            a = vals[0]
            others = [v for v in vals[1:] if not torch.equal(v, a)]
            if not others:
                continue
            dd = (a != others[0])
            idx = torch.nonzero(dd)
            log(f"    最早注入点(按依赖顺序)={k}: heads={torch.unique(idx[:, 1]).tolist()} "
                f"dim0=[{int(torch.unique(idx[:, 0]).min())}..{int(torch.unique(idx[:, 0]).max())}] "
                f"ndiff={int(dd.sum())} 坏侧 {region_stats(others[0][dd])}")
            break

    log("=" * 78)
    log(f"[3] 并发度扫描（{layout}，{args.n_scan} trial/档）")
    for ns in (1, 2, 4, 8):
        sts = [torch.npu.Stream() for _ in range(ns)]
        hit = 0
        for i in range(args.n_scan):
            outs = []
            for st in sts:
                with torch.npu.stream(st):
                    outs.append(call(layout, ins))
            torch.npu.synchronize()
            per = {k: [] for k in KEYS}
            for o in outs:
                for k, v in zip(KEYS, o):
                    if v is not None:
                        per[k].append(fp(v))
            if any(len(set(per[k])) > 1 for k in KEYS):
                hit += 1
        log(f"    streams={ns}: 漂移 trial={hit}/{args.n_scan}")

    log("=" * 78)
    log(f"[4] 是否必须「调用在时间上重叠」（{layout}，4 stream，每次调用后 sync）")
    sts = [torch.npu.Stream() for _ in range(4)]
    hit = 0
    for i in range(args.n_scan):
        outs = []
        for st in sts:
            with torch.npu.stream(st):
                outs.append(call(layout, ins))
            torch.npu.synchronize()
        per = {k: [] for k in KEYS}
        for o in outs:
            for k, v in zip(KEYS, o):
                if v is not None:
                    per[k].append(fp(v))
        if any(len(set(per[k])) > 1 for k in KEYS):
            hit += 1
    log(f"    漂移 trial={hit}/{args.n_scan}（0 ⇒ 必须两次调用同时在飞）")

    log("=" * 78)
    log(f"[5] 拼写对照（TND，4 stream × {args.n_scan} trial）")
    try:
        ins_t = tnd_inputs(ins)
        sts = [torch.npu.Stream() for _ in range(4)]
        hit = 0
        for i in range(args.n_scan):
            outs = []
            for st in sts:
                with torch.npu.stream(st):
                    outs.append(call("TND", ins_t))
            torch.npu.synchronize()
            per = {k: [] for k in KEYS}
            for o in outs:
                for k, v in zip(KEYS, o):
                    if v is not None:
                        per[k].append(fp(v))
            if any(len(set(per[k])) > 1 for k in KEYS):
                hit += 1
        log(f"    TND 漂移 trial={hit}/{args.n_scan}")
    except Exception as exc:
        log(f"    TND 对照失败: {exc}")

    log("=" * 78)
    log("[done] 把上面的完整输出贴回来即可")


if __name__ == "__main__":
    main()
