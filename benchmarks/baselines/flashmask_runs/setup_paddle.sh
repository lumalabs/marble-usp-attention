#!/bin/bash
# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
# Paddle venv for the FlashMask-3 Hopper baseline (../flashmask3_paddle_bench.py).
# usage: setup_paddle.sh WHEEL [WORKDIR]
#   WHEEL: paddlepaddle_gpu-3.4.0-cp312-cp312-linux_x86_64.whl from Paddle's cu129 index
#          (https://www.paddlepaddle.org.cn/packages/stable/cu129/; sha256 of ours:
#          2b950cffd8cb88b384867b7acd07ea15e08f3dfb4612d8ed92232cc430f8d343)
#   WORKDIR (default ..): gets pd-venv/ and paddle/context_parallel_utils.py, which must be
#          copied there from PaddleFleet (Apache 2.0; we used its 2026-10 main).
set -euo pipefail
WHEEL=$1; W=$(cd "${2:-..}" && pwd)
unset LD_PRELOAD VIRTUAL_ENV
rm -rf "$W/pd-venv" && uv venv "$W/pd-venv" --python 3.12
uv pip install --python "$W/pd-venv/bin/python" "$WHEEL" numpy
"$W/pd-venv/bin/python" - <<'PY'
import inspect, paddle
from paddle.nn.functional.flash_attention import flashmask_attention
print("paddle", paddle.__version__, "cuda", paddle.version.cuda(), "gpus", paddle.device.cuda.device_count())
print("flashmask_attention params:", list(inspect.signature(flashmask_attention).parameters))
PY
[ -f "$W/paddle/context_parallel_utils.py" ] || echo "copy PaddleFleet's context_parallel_utils.py to $W/paddle/"
