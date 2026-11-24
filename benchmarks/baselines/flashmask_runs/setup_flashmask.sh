#!/bin/bash
# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
# Build FlashMask-4 in its own venv, pinned to the versions behind the MLSys rows.
# usage: setup_flashmask.sh h100|b300 [WORKDIR]   (creates WORKDIR/flashmask and WORKDIR/fm-venv)
#
# Versions read from the venvs that produced the results (H100 pods and the B300 node):
# Python 3.12.14, FlashMask 4.0.0+g6a150fc, torch 2.11.0+cu129, triton 3.6.0,
# nvidia-cutlass 4.2.0.0, nvidia-cutlass-dsl 4.5.2, apache-tvm-ffi 0.1.14.post1,
# numpy 2.5.4, setuptools 81.0.0, nvidia-nccl-cu12 2.28.9 (bundled with torch).
# B300 adds nvidia-nccl-cu13 2.32.3, also unpacked to WORKDIR/nccl13 for LD_LIBRARY_PATH.
set -euo pipefail
TARGET=${1:?usage: setup_flashmask.sh h100|b300 [WORKDIR]}
W=$(cd "${2:-.}" && pwd)
FLASHMASK_COMMIT=6a150fcf771bdbbaded07a427b53513ad1f5fb7b
unset LD_PRELOAD VIRTUAL_ENV

[ -d "$W/flashmask" ] || git clone https://github.com/PaddlePaddle/FlashMask.git "$W/flashmask"
git -C "$W/flashmask" checkout "$FLASHMASK_COMMIT"

uv venv "$W/fm-venv" --python 3.12.14
PY=$W/fm-venv/bin/python
PINS=(
  "torch==2.11.0+cu129" "triton==3.6.0" "numpy==2.5.4" "setuptools==81.0.0"
  "nvidia-cutlass==4.2.0.0" "nvidia-cutlass-dsl==4.5.2" "apache-tvm-ffi==0.1.14.post1"
  pip pytest
)
case $TARGET in
  h100) BUILD=core+cpb ;;
  b300) BUILD=all; PINS+=("nvidia-nccl-cu13==2.32.3") ;;
  *) echo "unknown target $TARGET (h100 or b300)" >&2; exit 2 ;;
esac
uv pip install --python "$PY" --index https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match "${PINS[@]}"

if [ "$TARGET" = b300 ]; then
  # The B300 build ran with NCCL_HOME at this NCCL; the runs load it via LD_LIBRARY_PATH.
  export NCCL_HOME=$("$PY" -c "import nvidia.nccl, os; print(os.path.dirname(nvidia.nccl.__file__))")
  "$PY" -m pip install --target "$W/nccl13" "nvidia-nccl-cu13==2.32.3"
fi

(cd "$W/flashmask" && FLASHMASK_FRAMEWORK=torch FLASHMASK_BUILD=$BUILD "$PY" -m pip install --no-build-isolation -e .)
"$PY" -c "import flashmask, torch; print('flashmask ok', torch.__version__)"
echo "export FLASHMASK_DIR=$W/flashmask"
if [ "$TARGET" = b300 ]; then
  echo "export LD_LIBRARY_PATH=$W/nccl13/nvidia/nccl/lib:\${LD_LIBRARY_PATH:-}"
fi
