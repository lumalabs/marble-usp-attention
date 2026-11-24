#!/bin/bash
# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
# Run one list of FlashMask launcher argument lines on N nodes, one torchrun per line.
# Every node runs this from the usp_attention project root with its own NODE_RANK;
# lines run in lockstep, so start all nodes together.
# usage: run_list.sh NODE_RANK NNODES MASTER_ADDR LIST [OUT_DIR]
# env:   FM_VENV (default ../fm-venv), FLASHMASK_DIR (default ../flashmask)
# Writes OUT_DIR/<list name>.jsonl (the RESULT rows), .log and .progress.
set -u
RANK=$1; NN=$2; MASTER=$3; LIST=$4; OUT_DIR=${5:-benchmarks/results/rerun}
NAME=$(basename "$LIST" .list)
FM_VENV=${FM_VENV:-../fm-venv}
export FLASHMASK_DIR=$(cd "${FLASHMASK_DIR:-../flashmask}" && pwd)
export NCCL_NET_PLUGIN=none  # on our pods the libfabric plugin fell back to TCP (~2 GB/s) across nodes
unset LD_PRELOAD VIRTUAL_ENV
mkdir -p "$OUT_DIR"
OUT=$OUT_DIR/$NAME.jsonl; LOG=$OUT_DIR/$NAME.log; PROG=$OUT_DIR/$NAME.progress
: > "$OUT"; : > "$LOG"; : > "$PROG"
i=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  i=$((i + 1))
  # shellcheck disable=SC2086  # each line is an argument list
  "$FM_VENV/bin/torchrun" --nnodes "$NN" --node-rank "$RANK" --nproc-per-node 8 \
    --master-addr "$MASTER" --master-port $((29600 + i % 200)) \
    benchmarks/baselines/flashmask_cp_bench.py $line >> "$LOG" 2>&1
  status=$?
  grep '^RESULT' "$LOG" | sed 's/^RESULT //' > "$OUT"
  echo "$i exit=$status $line" >> "$PROG"
done < "$LIST"
echo ALLDONE >> "$PROG"
