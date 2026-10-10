#!/usr/bin/env bash
# 串行跑多个档位：一个臂接着一个臂"建包 + 跑测"，避免共享源码树 / 构建目录 / NPU 互相干扰。
#
# 用法：
#   SRC=/data/wnc/flash-linear-attention-npu \
#   TOOL=/data/wnc/issue_package/scripts/kda_v2_localize.py \
#   DUMP=/data/wnc/issue_package/dumps/bak-drift_v2_inv2426_vs_2435.pt \
#   bash kda_diag_serial.sh
#
# 可调环境变量：
#   ARMS="main:0:7b48499d8 pr865:2:b5b3bc6de pipe4:4:6983a43eb"
#                 串行顺序，格式 ARM:PIPE[:REV]；REV 用来把该算子目录切到指定提交（受控对比）
#   N=5000        每档 trial 数（想快点：N=2000，但区分能力会明显下降）
#   MAXS=3        每档最多收集几个漂移样本
#   DEV=0         固定用哪张卡（串行跑，不会被别人干扰）
#   REVERIFY=1    对"0 样本"的档位再单独复跑一次确认（默认 0）
#   FORCE=1       已有结果的档位也重跑（默认跳过已完成的）
#   REUSE=1       同档位包已存在则跳过重建（默认 1）
#
set -uo pipefail
SRC=${SRC:?请指定 #865 源码目录}
TOOL=${TOOL:?请指定 kda_v2_localize.py 路径}
DUMP=${DUMP:?请指定 dump 路径}
ROOT=${ROOT:-/data/wnc/kda_diag}
N=${N:-5000}
MAXS=${MAXS:-3}
DEV=${DEV:-0}
RELAY=${RELAY:-2}
SWAP=${SWAP:-0}
PAUSE=${PAUSE:-0}
REVERIFY=${REVERIFY:-0}
FORCE=${FORCE:-0}
ARMS=${ARMS:-"main:0:7b48499d8 pr865:2:b5b3bc6de pipe4:4:6983a43eb"}

HERE=$(cd "$(dirname "$0")" && pwd)
ARM_SH=$HERE/kda_diag_arm.sh
[ -f "$ARM_SH" ] || { echo "找不到 $ARM_SH" >&2; exit 2; }
mkdir -p "$ROOT"

echo "=== 串行档位计划: $ARMS"
echo "=== 每档: N=$N MAXS=$MAXS DEV=$DEV RELAY=$RELAY SWAP=$SWAP PAUSE=$PAUSE REVERIFY=$REVERIFY"
echo "=== 日志: $ROOT/<ARM>/console.log ；汇总: $ROOT/serial_summary.txt"
echo

run_one() {  # $1=ARM $2=PIPE $3=SAVE $4=REV
  ARM="$1" PIPE="$2" RELAY="$RELAY" SWAP="$SWAP" DEV="$DEV" \
  MAXS="$MAXS" N="$N" PAUSE="$PAUSE" SAVE="$3" REV="$4" \
  ALLOW_MISSING_MACRO=1 \
  SRC="$SRC" TOOL="$TOOL" DUMP="$DUMP" ROOT="$ROOT" \
  bash "$ARM_SH"
}

for spec in $ARMS; do
  # 规格: ARM:PIPE[:REV]
  ARM=${spec%%:*}
  rest=${spec#*:}
  PIPE=${rest%%:*}
  REV_ARM=""
  case "$rest" in *:*) REV_ARM=${rest#*:} ;; esac
  LOG=$ROOT/$ARM/console.log
  mkdir -p "$ROOT/$ARM"
  DONE=$ROOT/$ARM/log/run.log

  if [ "$FORCE" != 1 ] && [ -f "$DONE" ] && grep -q "漂移率: " "$DONE"; then
    echo "=== [$ARM] 已有结果，跳过（FORCE=1 可重跑）"
  else
    echo "=== [$ARM] PIPE=$PIPE REV=${REV_ARM:-<HEAD>} 开始 $(date +%H:%M:%S)"
    run_one "$ARM" "$PIPE" "$ROOT/$ARM/first.pt" "$REV_ARM" | tee "$LOG"
    echo "=== [$ARM] 结束 $(date +%H:%M:%S)"
  fi

  if [ "$REVERIFY" = 1 ] && [ -f "$DONE" ] && grep -q "漂移率: 0/" "$DONE"; then
    echo "=== [$ARM] 首轮 0 样本 → 单独复跑确认 $(date +%H:%M:%S)"
    run_one "$ARM" "$PIPE" "$ROOT/$ARM/first_verify.pt" "$REV_ARM" | tee -a "$LOG"
  fi
done

# ---------- 汇总 ----------
python3 - "$ROOT" "$ARMS" <<'PY' | tee "$ROOT/serial_summary.txt"
import pathlib, re, sys

root = pathlib.Path(sys.argv[1])
arms = sys.argv[2].split()
print("=" * 104)
print(f"{'ARM':<7}{'PIPE':>5}{'漂移率':>14}  {'最早漂移点 / 落点':<46}{'坏侧形态':<30}")
print("-" * 104)
for spec in arms:
    arm, _, pipe = spec.partition(":")
    log = root / arm / "log" / "run.log"
    rate = loc = shape = "-"
    runs = 0
    if log.exists():
        text = log.read_text(encoding="utf-8", errors="replace")
        hits = re.findall(r"漂移率: (\d+)/(\d+) trial", text)
        if hits:
            runs = len(hits)
            rate = " / ".join(f"{a}/{b}" for a, b in hits) + (f"（{runs} 轮）" if runs > 1 else "")
        m = re.search(r"最早漂移点 = (\w+)", text)
        tgt = m.group(1) if m else "-"
        m = re.search(
            r"heads=\[[^\]]*\] tokens=\[(\d+)\.\.(\d+)\] -> seq=(\d+)\(len=\d+\) "
            r"chunk_in_seq=\d+ global_chunk=(\d+)", text)
        if m:
            loc = f"{tgt} tokens[{m.group(1)}..{m.group(2)}] seq{m.group(3)} chunk{m.group(4)}"
        elif tgt != "-":
            loc = tgt
        m = re.search(r"坏侧 exact0=[\d.]+ \|x\|<1e-6=[\d.]+ \|x\|>1=([\d.eE+-]+) max=([\d.eE+-]+)", text)
        if m:
            shape = f"|x|>1={m.group(1)} max={m.group(2)}"
            # 附带样本数
            m2 = re.search(r"该 trial 上跨 stream 不一致的 tensor", text)
            if m2:
                shape += ""
    print(f"{arm:<7}{pipe:>5}{rate:>14}  {loc:<46}{shape:<30}")
print("=" * 104)
print(f"明细: {root}/<ARM>/console.log ；每档 run.log 在 <ARM>/log/run.log")
PY
