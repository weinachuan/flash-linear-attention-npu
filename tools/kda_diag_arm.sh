#!/usr/bin/env bash
# KDA V2 前向漂移：单档位构建 + 安装 + OPP 自检 + 现场跑测（一个档位一条命令）
#
# 用法示例（在 A3 现场）：
#   SRC=/data/wnc/flash-linear-attention-npu \
#   TOOL=/data/wnc/issue_package/scripts/kda_v2_localize.py \
#   DUMP=/data/wnc/issue_package/dumps/bak-drift_v2_inv2426_vs_2435.pt \
#   ARM=pipe4 PIPE=4 SWAP=0 RELAY=2 N=5000 \
#   bash kda_diag_arm.sh
#
# 常用的跑测参数（都是环境变量，默认见下）：
#   N=5000        每个档位跑多少 trial（4 stream → 4×N 次调用）
#   MAXS=5        收集多少个漂移样本后停止；=1 表示"第一次漂移就停"
#   PAUSE=1       抓到第一个漂移后暂停等回车（张量保活；stdin 是终端才生效）
#   SAVE=<path>   把第一个漂移样本的关键小切片存成 .pt（几百 KB，可外发）
#   EXTRA="..."   额外透传给 kda_v2_localize.py 的参数
#
# 运行前请先激活你的 conda 环境（例如 conda activate fzy_atk），脚本会沿用当前
# python3；CANN 环境脚本默认取 /usr/local/Ascend/ascend-toolkit/set_env.sh，
# 需要换路径时用 CANN_SET_ENV=/your/cann/set_env.sh。
#
# 三个档位（默认都在 policy.h 里，脚本会按环境变量改写后再编）：
#   RELAY = CHUNK_KDA_FWD_PREPARE_RELAY_SYNC      0/1/2   （#865 默认 2）
#   SWAP  = KDA_PREPARE_RELAY_ORDER_SWAP          0/1     （1 = 对调 V6 两份 relay 搬运顺序）
#   PIPE  = KDA_PREPARE_DIAG_PIPE_ALL             0/1/2/3/4/5/9（PIPE_ALL 粗粒度串行）
#
# 每个档位落到独立目录 <ROOT>/<ARM>，互不污染（wheels/py/log 各自一份）。
set -euo pipefail

SRC=${SRC:?请指定 #865 的源码检出目录，例如 /data/wnc/flash-linear-attention-npu}
ARM=${ARM:?请给这个档位起个名字，例如 base0 / pipe4 / pipe9 / swap1}
TOOL=${TOOL:-$(dirname "$0")/kda_v2_localize.py}
DUMP=${DUMP:?请指定 dump 路径}
ROOT=${ROOT:-/data/wnc/kda_diag}
N=${N:-5000}
RELAY=${RELAY:-2}
SWAP=${SWAP:-0}
PIPE=${PIPE:-0}
MAXS=${MAXS:-5}
PAUSE=${PAUSE:-0}
SAVE=${SAVE:-}
EXTRA=${EXTRA:-}

OP=fla/ops/ascendc/kda/chunk_kda_fwd_prepare
POL=$SRC/$OP/op_kernel/chunk_kda_fwd_prepare_policy.h
BASE=$ROOT/$ARM
mkdir -p "$BASE"/wheels "$BASE"/py "$BASE"/log

# ---------- 0) 源码状态 ----------
cd "$SRC"
git rev-parse --short HEAD
git log --oneline -1
git checkout -- "$OP" || true      # 只回滚这个算子的文件，不动你其它改动

# ---------- 1) 设置档位 ----------
set_macro() {  # $1=宏名 $2=值
  if grep -q "^#define $1 " "$POL"; then
    sed -i "s/^#define $1 .*/#define $1 $2/" "$POL"
  else
    echo "!! policy.h 里没有 $1，说明源码不是 #865+诊断分支" >&2
    exit 1
  fi
}
set_macro CHUNK_KDA_FWD_PREPARE_RELAY_SYNC "$RELAY"
set_macro KDA_PREPARE_RELAY_ORDER_SWAP     "$SWAP"
set_macro KDA_PREPARE_DIAG_PIPE_ALL        "$PIPE"
echo "=== 本次档位 ==="
grep -nE "^#define (CHUNK_KDA_FWD_PREPARE_RELAY_SYNC|KDA_PREPARE_RELAY_ORDER_SWAP|KDA_PREPARE_DIAG_PIPE_ALL) " "$POL"
echo "=== 本次 diff（相对于该分支提交） ==="
git --no-pager diff --stat -- "$OP"

# ---------- 2) 建 wheel ----------
CANN_SET_ENV=${CANN_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
source "$CANN_SET_ENV"
export FLA_NPU_SOC=${FLA_NPU_SOC:-ascend910_93}          # A3；A2 用 ascend910b
export FLA_NPU_OPS=chunk_kda_fwd,chunk_kda_fwd_prepare,chunk_kda_fwd_finalize,chunk_fwd_h
echo "=== build (SOC=$FLA_NPU_SOC) ==="
python3 -m pip wheel --no-build-isolation --no-deps . -w "$BASE/wheels" > "$BASE/log/build.log" 2>&1
echo "build rc=$?"
tail -3 "$BASE/log/build.log"
W=$(ls -t "$BASE"/wheels/*.whl | head -1)
echo "wheel: $W"

# ---------- 3) 安装到独立目录 ----------
rm -rf "$BASE/py"
python3 -m pip install --upgrade --force-reinstall --no-deps --target "$BASE/py" "$W" > "$BASE/log/install.log" 2>&1
echo "install rc=$?"

# ---------- 4) OPP 自检：档位真的进包了吗 ----------
P=$BASE/py/fla_npu
echo "=== 包内 policy 宏 ==="
grep -nE "^#define (CHUNK_KDA_FWD_PREPARE_RELAY_SYNC|KDA_PREPARE_RELAY_ORDER_SWAP|KDA_PREPARE_DIAG_PIPE_ALL) " \
  $P/opp/vendors/*/op_impl/ai_core/tbe/fla_npu_transformer_impl/ascendc/chunk_kda_fwd_prepare/chunk_kda_fwd_prepare_policy.h
echo "=== 内核对象（名字带源码哈希，跨档位应不同） ==="
ls -l $P/opp/vendors/*/op_impl/ai_core/tbe/kernel/*/chunk_kda_fwd_prepare/ | tee "$BASE/log/objs.txt" | head

# ---------- 5) 运行环境：PYTHONPATH + custom OPP ----------
cat > "$BASE/env.sh" <<ENV
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=$BASE/py:\${PYTHONPATH:-}
source $P/opp/vendors/fla_npu_transformer/bin/set_env.bash
ENV
source "$BASE/env.sh"
python3 - "$BASE" <<'PY'
import os, pathlib, sys
import fla_npu
base = pathlib.Path(sys.argv[1])
ok = pathlib.Path(fla_npu.__file__).resolve().is_relative_to(base)
print("fla_npu  :", fla_npu.__file__)
print("命中本次安装:", ok)
print("ASCEND_CUSTOM_OPP_PATH:", os.environ.get("ASCEND_CUSTOM_OPP_PATH"))
PY

# ---------- 6) 跑定位 ----------
ARGS=(--dump "$DUMP" --n-stress "$N" --max-samples "$MAXS")
[ "$PAUSE" = 1 ] && ARGS+=(--pause-on-first)
[ -n "$SAVE" ] && ARGS+=(--save-first "$SAVE")
if [ -n "$EXTRA" ]; then
  # shellcheck disable=SC2206
  ARGS+=($EXTRA)
fi
echo "=== run: python3 $TOOL ${ARGS[*]} ==="
echo "    （PAUSE=$PAUSE：只在抓到第一个漂移后暂停；进度条每 ~4% 一行）"
python3 "$TOOL" "${ARGS[@]}" 2>&1 | tee "$BASE/log/run.log"

echo
echo "=== 把下面两段贴回来即可 ==="
echo "1) $BASE/log/run.log 里 [1] / [2] 两节（含【样本落点表】与【首个漂移 trial】）"
echo "2) 上面「包内 policy 宏」与「内核对象」两行"
