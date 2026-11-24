#!/bin/bash
# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
# Run one list of FlashMask-3 (Paddle) argument lines on N nodes, one launch per line,
# from the usp_attention project root on every node at the same time.
# usage: run_paddle_list.sh NODE_RANK NNODES MASTER_ADDR LIST [OUT_DIR]
# env:   PD_VENV (default ../pd-venv), CP_UTILS (default ../paddle/context_parallel_utils.py),
#        RUN_TIMEOUT seconds per line (default 600; overlap runs can hang)
# Writes OUT_DIR/<list name>.jsonl (rank 0's RESULT rows), .log and .progress.
set -u
RANK=$1; NN=$2; MASTER=$3; LIST=$4; OUT_DIR=${5:-benchmarks/results/rerun}
NAME=$(basename "$LIST" .list)
PD_VENV=${PD_VENV:-../pd-venv}
export CP_UTILS=$(readlink -f "${CP_UTILS:-../paddle/context_parallel_utils.py}")
export FLAGS_flash_attn_version=3 NCCL_NET_PLUGIN=none
unset LD_PRELOAD VIRTUAL_ENV
mkdir -p "$OUT_DIR"
OUT=$OUT_DIR/$NAME.jsonl; LOG=$OUT_DIR/$NAME.log; PROG=$OUT_DIR/$NAME.progress
: > "$OUT"; : > "$LOG"; : > "$PROG"
i=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  i=$((i + 1))
  rm -rf "/tmp/pdlog_$NAME"
  # shellcheck disable=SC2086  # each line is an argument list
  timeout -k 30 "${RUN_TIMEOUT:-600}" "$PD_VENV/bin/python" -m paddle.distributed.launch \
    --nnodes "$NN" --rank "$RANK" --master "$MASTER:$((30100 + i % 200))" \
    --devices 0,1,2,3,4,5,6,7 --log_dir "/tmp/pdlog_$NAME" \
    benchmarks/baselines/flashmask3_paddle_bench.py $line >> "$LOG" 2>&1
  status=$?
  pkill -9 -f "[p]ython.*flashmask3_paddle_bench" 2>/dev/null
  cat "/tmp/pdlog_$NAME/workerlog.0" >> "$LOG" 2>/dev/null
  grep -h '^RESULT' "$LOG" | sed 's/^RESULT //' | awk '!seen[$0]++' > "$OUT"
  echo "$i exit=$status $line" >> "$PROG"
done < "$LIST"
echo ALLDONE >> "$PROG"
