#!/usr/bin/env python3
"""判定一份 KDA fwd drift dump 的"签名"，用于跨版本/跨 incident 比对。

只读 torch.save 的 zip（不 import torch，numpy 足够），输出控制在几十行内，
便于直接贴回讨论。做三件事：

  [1] 输入是否逐位一致；哪些输出不一致、差异落在哪些 dim0 索引上；
  [2] 差异是否仍集中"一个 (value head, chunk) work item"（head 单一 + 连续 token 段
      长度 <= chunk_size；h / final_state 各自只看一个 chunk / 一个序列）；
  [3] 对 u 反解等效右操作数 V_beta' = Akk^{-1} u，与"正确操作数"对比：中位幅值比、
      |x|>1 的比例、与 v*beta 的相关系数 —— 判断是否仍是"几乎全 0 + 少量巨大值"。

用法:
    python3 kda_drift_signature.py <dump.pt> [--u-idx 6]

支持两种 dump 形态：
  * v2（推荐）：含 fwd_inputs / fwd_outputs / recompute_outputs（本脚本全量分析）；
  * v1：只含 inputs + 指纹（out_fp/prev_out_fp），此时只打印指纹对比。
"""

from __future__ import annotations

import argparse
import io
import pickle
import zipfile

import numpy as np

DTYPES = {
    "BFloat16Storage": np.dtype(np.uint16),
    "HalfStorage": np.dtype(np.float16),
    "FloatStorage": np.dtype(np.float32),
    "LongStorage": np.dtype(np.int64),
    "IntStorage": np.dtype(np.int32),
    "ByteStorage": np.dtype(np.uint8),
}
OUT_KEYS = ["o", "final_state", "gk", "Aqk", "Akk", "w", "u", "qg", "kg", "v_new", "h"]


# ---------------------------------------------------------------- dump loading
class Storage:
    def __init__(self, cls, key=None):
        self.cls = cls
        self.key = key


class TensorStub:
    def __init__(self, storage, offset, size, stride):
        self.storage = storage
        self.size = tuple(int(d) for d in size)
        self.stride = tuple(int(d) for d in stride)


class Any:
    def __init__(self, name, *args):
        self.name = name


def _rebuild(storage, offset, size, stride, *a, **k):
    return TensorStub(storage, offset, size, stride)


_STORAGE_CLASSES = {}


def _storage_class(qualified):
    cls = _STORAGE_CLASSES.get(qualified)
    if cls is not None:
        return cls
    simple = qualified.rsplit(".", 1)[-1]
    return _STORAGE_CLASSES.setdefault(
        qualified, type(simple, (Storage,), {"__init__": lambda self, *a, **k: None}))


class Unpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        kind = pid[0]
        if isinstance(kind, str):
            key, numel = pid[2], pid[4]
            name = getattr(pid[1], "__name__", None) or type(pid[1]).__name__
        else:
            key, numel = pid[1], pid[3]
            name = type(kind).__name__
        return Storage(f"torch.{name}", (key, numel))

    def find_class(self, module, name):
        full = f"{module}.{name}"
        if "Storage" in name:
            return _storage_class(full)
        if "rebuild" in name:
            return _rebuild
        return lambda *a, **k: Any(full)


def load_dump(path):
    archive = zipfile.ZipFile(path)
    pkl = [n for n in archive.namelist() if n.endswith("data.pkl")][0]
    obj = Unpickler(io.BytesIO(archive.read(pkl))).load()

    def arr(tensor):
        dtype_name = tensor.storage.cls.rsplit(".", 1)[-1]
        member = next(n for n in archive.namelist()
                      if n.endswith(f"data/{tensor.storage.key[0]}"))
        raw = np.frombuffer(archive.read(member), dtype=DTYPES[dtype_name])
        if dtype_name == "BFloat16Storage":
            raw = (raw.astype(np.uint32) << 16).view(np.float32)
        return raw.reshape(tensor.size)

    sections = {}
    for name, value in obj.items():
        if isinstance(value, dict) and any(isinstance(v, TensorStub) for v in value.values()):
            sections[name] = {k: arr(v) for k, v in value.items() if isinstance(v, TensorStub)}
        else:
            sections[name] = value
    return sections


# ------------------------------------------------------------------- reporting
def dim0_report(a, b):
    """返回 (ndiff, dim0 索引数组, 说明)。"""
    if a is None or b is None:
        return None, None, "missing"
    if a.shape != b.shape:
        return None, None, f"shape mismatch {a.shape} vs {b.shape}"
    diff = a != b
    ndiff = int(diff.sum())
    if ndiff == 0:
        return 0, np.array([], dtype=np.int64), "bitwise identical"
    idx = np.argwhere(diff)
    dim0 = np.unique(idx[:, 0])
    return ndiff, dim0, (f"dim0 n={dim0.size} first={dim0[:4].tolist()} last={dim0[-4:].tolist()}")


def contiguous(runs):
    runs = np.sort(np.asarray(runs))
    if runs.size == 0:
        return True
    return bool(np.array_equal(runs, np.arange(runs[0], runs[-1] + 1)))


def chunks_of(cu, chunk_size, tokens):
    if not cu:
        return [(s, min(s + chunk_size, tokens)) for s in range(0, tokens, chunk_size)]
    out = []
    for lo, hi in zip(cu, cu[1:]):
        s = lo
        while s < hi:
            out.append((s, min(s + chunk_size, hi)))
            s += chunk_size
    return out


def resolve_op(logits, hidden):
    """softmax(logits) @ hidden，按行（fp32 计算）。"""
    m = logits.max(axis=-1, keepdims=True)
    p = np.exp(logits - m)
    p /= p.sum(axis=-1, keepdims=True)
    return (p.astype(np.float32) @ hidden.astype(np.float32)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--u-key", default="u")
    ap.add_argument("--akk-key", default="Akk")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    d = load_dump(args.dump)
    tag = f"[{args.tag}] " if args.tag else ""
    print(f"{tag}dump={args.dump}")

    if "fwd_outputs" not in d or "recompute_outputs" not in d:
        print(f"{tag}不是 v2 dump（只有指纹）；keys={sorted(d)}")
        return

    scal = d.get("scalars", {})
    cu = scal.get("cu_seqlens")
    chunk = int(scal.get("chunk_size", 64))
    layout = scal.get("layout")
    print(f"{tag}layout={layout} chunk={chunk} cu_seqlens={cu}")

    # [1] 输入
    if "fwd_inputs" in d and "recompute_inputs" in d:
        bad = [k for k in d["fwd_inputs"]
               if k in d["recompute_inputs"]
               and not np.array_equal(d["fwd_inputs"][k], d["recompute_inputs"][k])]
        print(f"{tag}[1] inputs bitwise identical: "
              f"{'YES' if not bad else 'NO -> ' + str(bad)}")

    # [2] 输出差异与落点
    print(f"{tag}[2] outputs (fwd vs recompute):")
    diff_keys, detail = [], {}
    for k in OUT_KEYS:
        a, b = d["fwd_outputs"].get(k), d["recompute_outputs"].get(k)
        if a is None or b is None:
            continue
        ndiff, dim0, note = dim0_report(a, b)
        detail[k] = (a, b, ndiff, dim0)
        if ndiff:
            diff_keys.append(k)
        print(f"{tag}    {'OK ' if not ndiff else '!! '}{k:>11}: {note}")

    single_item, head_single = True, True
    # 根因 work item 判据：u 自身必须落在"单一 head + 连续 <=1 个 chunk"；
    # o/v_new 因为状态传播会覆盖到下一块，允许 <= 2 个 chunk。
    root_ok, propagated_ok = True, True
    for k in diff_keys:
        a, b, ndiff, dim0 = detail[k]
        if k in ("o", "u", "v_new", "gk", "w", "qg", "kg"):
            # head-major (H,T,C) 或 token-major (T,H,C)
            if a.ndim == 3 and a.shape[0] <= 8 and a.shape[1] > 64:      # (H,T,C)
                tok = np.unique(np.argwhere(a != b)[:, 1])
                heads = np.unique(np.argwhere(a != b)[:, 0])
            elif a.ndim == 3:
                tok = dim0
                heads = np.unique(np.argwhere(a != b)[:, 1])
            else:
                continue
            head_single &= heads.size == 1
            if k == "u":
                root_ok &= contiguous(tok) and tok.size <= chunk
            else:
                propagated_ok &= contiguous(tok) and tok.size <= 2 * chunk
            print(f"{tag}    {k}: heads={heads.tolist()} tokens n={tok.size} "
                  f"contiguous={contiguous(tok)} span=[{tok.min()},{tok.max()}] "
                  f"(<=chunk{'' if k == 'u' else 'x2'}: {tok.size <= (chunk if k == 'u' else 2 * chunk)})")
        elif k in ("h", "final_state"):
            print(f"{tag}    {k}: single index set = {dim0.tolist()[:6]} "
                  f"(n={dim0.size})")
            single_item &= dim0.size <= 2
    same_tensor_set = set(diff_keys) == {"o", "final_state", "u", "v_new", "h"}

    # [3] u 的等效右操作数
    verdict_operand = "n/a"
    if args.u_key in detail and detail[args.u_key][2]:
        u_f, u_r, ndiff, dim0 = detail[args.u_key]
        akk = d["fwd_outputs"].get(args.akk_key)
        v_in = (d.get("fwd_inputs") or {}).get("v_in")
        beta = (d.get("fwd_inputs") or {}).get("beta_in")
        if akk is not None and akk.ndim == 3 and u_f.ndim == 3 and u_f.shape[0] <= 8:
            head = int(dim0[0]) if dim0.size else 0
            tok = np.unique(np.argwhere(u_f != u_r)[:, 1])
            # 找包含该 token 段的 chunk 起点
            start = None
            for lo, hi in chunks_of(cu, chunk, u_f.shape[1]):
                if lo <= tok.min() < hi:
                    start, end = lo, hi
                    break
            rows = end - start
            A = akk[head, start:end, :rows].astype(np.float64)
            uf = u_f[head, start:end, :].astype(np.float64)
            ur = u_r[head, start:end, :].astype(np.float64)

            def solve(rhs):
                if A.shape[0] != A.shape[1]:
                    M = A.T @ A + 1e-12 * np.eye(A.shape[1])
                    return np.linalg.solve(M, A.T @ rhs)
                return np.linalg.solve(A, rhs)

            v_true = solve(uf)
            v_rec = solve(ur)
            res_true = np.linalg.norm(A @ v_true - uf) / max(np.linalg.norm(uf), 1e-30)
            res_rec = np.linalg.norm(A @ v_true - ur) / max(np.linalg.norm(ur), 1e-30)
            med_t = float(np.median(np.abs(v_true)))
            med_r = float(np.median(np.abs(v_rec)))
            huge = float((np.abs(v_rec) > 1.0).mean())
            corr = float(np.corrcoef(v_true.ravel(), v_rec.ravel())[0, 1]) \
                if v_true.size > 1 else float("nan")
            print(f"{tag}[3] chunk=[{start},{end}) rows={rows} head={head}")
            print(f"{tag}    solve residual(fwd)={res_true:.2e}  "
                  f"||A@V_true - u_rec||/||u_rec||={res_rec:.3e}")
            print(f"{tag}    V_true median|x|={med_t:.3e}   V_rec median|x|={med_r:.3e}   "
                  f"ratio={med_r / med_t if med_t else float('nan'):.3e}")
            print(f"{tag}    V_rec: |x|>1 占比={huge:.4f}  max|x|={np.abs(v_rec).max():.4g}  "
                  f"corr(V_true,V_rec)={corr:.4f}")
            if beta is not None and v_in is not None:
                vb = v_in[head, start:end, :].astype(np.float64) * beta[head, start:end, None]
                print(f"{tag}    corr(V_true, v*beta)="
                      f"{np.corrcoef(v_true.ravel(), vb.ravel())[0, 1]:.4f}")
            verdict_operand = ("empty-ish operand (median ratio << 1 with a few huge values)"
                               if (med_t and med_r / med_t < 0.1 and huge > 0.001)
                               else "operand looks normal -> different mechanism")

    print(f"{tag}VERDICT: tensor_set={'same' if same_tensor_set else sorted(diff_keys)} "
          f"single_work_item={bool(head_single and root_ok and propagated_ok and single_item)} "
          f"operand={verdict_operand}")


if __name__ == "__main__":
    main()
