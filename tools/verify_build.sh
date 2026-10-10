#!/usr/bin/env bash
# 一条命令：切分支 → 清残留 → 建包 → 隔离安装 → md5 自证 → 跑验证
#
# 用法（默认切主线；要验 #865 就换 REV）：
#   REV=7b48499d8 bash verify_build.sh                 # 主线
#   REV=b5b3bc6de bash verify_build.sh                 # #865
#   REV=7b48499d8 N=2000 bash verify_build.sh          # 主线，跑 2000 trial
#
# 环境变量：
#   SRC   源码目录（默认 /data/wnc/flash-linear-attention-npu）
#   TOOL  定位脚本（默认 /data/wnc/issue_package/scripts/kda_v2_localize.py）
#   DUMP  dump 路径（默认 /data/wnc/issue_package/dumps/bak-drift_v2_inv2426_vs_2435.pt）
#   SOC   目标平台（默认 ascend910_93 = A3；A2 用 ascend910b）
#   N     验证跑的 trial 数（默认 500）
#   CLEAN 1=建包前删掉 build/build_out/dist（默认 1，彻底排除残留）
set -euo pipefail

SRC=${SRC:-/data/wnc/flash-linear-attention-npu}
TOOL=${TOOL:-/data/wnc/issue_package/scripts/kda_v2_localize.py}
DUMP=${DUMP:-/data/wnc/issue_package/dumps/bak-drift_v2_inv2426_vs_2435.pt}
REV=${REV:-7b48499d8}
SOC=${SOC:-ascend910_93}
N=${N:-500}
CLEAN=${CLEAN:-1}
OUTROOT=${OUTROOT:-/data/wnc/verify_build}
OUT=$OUTROOT/$REV
mkdir -p "$OUT"

echo "==================== 1) 切分支 $REV ===================="
cd "$SRC"
git fetch --all -q 2>/dev/null || true
git checkout -f "$REV"
git --no-pager log --oneline -1
echo "HEAD = $(git rev-parse --short HEAD)  (期望 $REV)"
git status --short | head -5

if [ "$CLEAN" = 1 ]; then
  echo "==================== 2) 清构建残留 ===================="
  rm -rf "$SRC/build" "$SRC/build_out" "$SRC/dist"
  echo "已删除 \$SRC/{build,build_out,dist}（只删构建产物）"
fi

echo "==================== 3) 建 wheel (SOC=$SOC) ===================="
if [ -n "${CANN_SET_ENV:-}" ]; then source "$CANN_SET_ENV"; else source /usr/local/Ascend/ascend-toolkit/set_env.sh; fi
export FLA_NPU_SOC=$SOC
export FLA_NPU_OPS=chunk_kda_fwd,chunk_kda_fwd_prepare,chunk_kda_fwd_finalize,chunk_fwd_h
python3 -m pip wheel --no-build-isolation --no-deps --no-cache-dir . -w "$OUT/wheels" 2>&1 | tail -5
W=$(ls -t "$OUT"/wheels/*.whl | head -1)
echo "wheel: $W"

echo "==================== 4) 隔离安装 + 环境 ===================="
rm -rf "$OUT/py"
python3 -m pip install --upgrade --force-reinstall --no-deps --target "$OUT/py" "$W" 2>&1 | tail -3
P=$OUT/py/fla_npu
cat > "$OUT/env.sh" <<ENV
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=$OUT/py:\${PYTHONPATH:-}
source $P/opp/vendors/fla_npu_transformer/bin/set_env.bash
ENV
source "$OUT/env.sh"
python3 - "$OUT" <<'PY'
import pathlib, sys
import fla_npu
base = pathlib.Path(sys.argv[1])
print("fla_npu:", fla_npu.__file__)
print("命中本次安装:", pathlib.Path(fla_npu.__file__).resolve().is_relative_to(base))
PY

echo "==================== 5) md5 自证（与已知版本对照） ===================="
KD=$P/opp/vendors/fla_npu_transformer/op_impl/ai_core/tbe/fla_npu_transformer_impl/ascendc/chunk_kda_fwd_prepare
md5sum "$KD/arch22/chunk_kda_fwd_prepare_vec.h" \
       "$KD/arch22/chunk_kda_fwd_prepare_cube.h" \
       "$KD/chunk_kda_fwd_prepare_policy.h" | tee "$OUT/md5.txt"
echo "参照：主线 7b48499d8 = a189793e… / 228f9edd… / ccc55f19…"
echo "      #865 b5b3bc6de = 0fad4588… / 1133d1b3… / 072cfd66…"
echo "      （#865 的 vec/cube 里能看到 kArch22Rhs 与 DataCacheCleanAndInvalid；主线没有）"

echo "==================== 6) 验证跑（$N trial × 4 stream） ===================="
TOOL_HELP=$(python3 "$TOOL" -h 2>&1 || true)
EXTRA_ARGS=()
if printf '%s' "$TOOL_HELP" | grep -q -- "--save-first"; then
  EXTRA_ARGS=(--save-first "$OUT/first.pt")
fi
python3 "$TOOL" --dump "$DUMP" --n-stress "$N" --max-samples 1 "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$OUT/run.log" | tail -30 || true

echo "==================== 结果位置 ===================="
echo "轮子:  $W"
echo "md5 :  $OUT/md5.txt"
echo "跑测:  $OUT/run.log  （看 [1] 是否全部 identical、[2] 的漂移率与样本表）"
