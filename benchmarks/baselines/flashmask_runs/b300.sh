#!/bin/bash
# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
# FlashMask rows of the B300 table (one node, 8 GPUs): FlashMask CP (balanced
# all-gather), FM + Ulysses, and FlashMask's Blackwell overlap path. Run from the
# usp_attention project root after `setup_flashmask.sh b300 WORKDIR`.
# usage: b300.sh WORKDIR [OUT_DIR]
# Rows land in OUT_DIR/b300_flashmask.jsonl. The committed rows are split over
# benchmarks/results/mlsys/b300_eval.jsonl (part A), b300_eval2.jsonl (B) and
# b300_eval3.jsonl (C), interleaved there with our own rows.
set -u
W=$(cd "${1:?usage: b300.sh WORKDIR [OUT_DIR]}" && pwd); OUT_DIR=${2:-benchmarks/results/rerun}
export FLASHMASK_DIR=$W/flashmask
unset LD_PRELOAD VIRTUAL_ENV
mkdir -p "$OUT_DIR"; OUT=$OUT_DIR/b300_flashmask.jsonl; LOG=$OUT_DIR/b300_flashmask.log; : > "$OUT"; : > "$LOG"
L=benchmarks/layouts/packs
fm() {
  LD_LIBRARY_PATH=$W/nccl13/nvidia/nccl/lib:${LD_LIBRARY_PATH:-} "$W/fm-venv/bin/torchrun" --nproc-per-node 8 \
    benchmarks/baselines/flashmask_cp_bench.py "$@" > /tmp/fm_b300.log 2>&1
  grep '^RESULT' /tmp/fm_b300.log | sed 's/^RESULT //' >> "$OUT"; cat /tmp/fm_b300.log >> "$LOG"
}
Q38="--heads 24 --kv-heads 4 --head-dim 256"; Q3="--heads 32 --kv-heads 4 --head-dim 128"
MO="--heads 32 --kv-heads 8 --head-dim 128"; GE="--heads 32 --kv-heads 16 --head-dim 128"
BA="--heads 28 --kv-heads 4 --head-dim 128"; MHA="--heads 32 --kv-heads 32 --head-dim 128"

# A. FlashMask CP and overlap on 1M-token batches (b300_eval.jsonl).
while read -r lab f heads; do
  for mode in overlap balanced; do
    fm --mode "$mode" --layout "$L/$f.json#0" --seq-len 1048576 ${!heads} --iters 5 --warmup 2 --label "fm_$lab"
  done
done <<LIST
qwen38 qwen38_video_1024k Q38
qwen3vl30 qwen38_video_1024k Q3
molmo2 molmo2_video_1024k MO
gemma3 gemma3_video_1024k GE
qwen3vl30 qwen38_mixed_1024k Q3
molmo2 molmo2_mixed_1024k MO
mha synth_mixed_1024k MHA
LIST

# B. Molmo2 mixed again, and Gemma 3 sliding-window layers (b300_eval2.jsonl).
for mode in balanced overlap; do
  fm --mode "$mode" --layout "$L/molmo2_mixed_1024k.json#0" --seq-len 1048576 $MO --iters 5 --warmup 2 --label fm_molmo2
done
for f in gemma3_video_1024k gemma3_interleaved_1024k; do
  for mode in balanced overlap; do
    fm --mode "$mode" --layout "$L/$f.json#0" --seq-len 1048576 $GE --sliding-window 1024 --iters 5 --warmup 2 --label fm_gemma3_swa
  done
done

# C. BAGEL, and FM + Ulysses at Ulysses 4 and 8 (b300_eval3.jsonl).
for f in bagel_video_1024k bagel_mix_1024k; do
  fm --mode balanced --layout "$L/$f.json#0" --seq-len 1048576 $BA --iters 4 --warmup 2 --label fm_bagel
  fm --mode balanced --ulysses 4 --layout "$L/$f.json#0" --seq-len 1048576 $BA --iters 4 --warmup 2 --label fm_bagel
  fm --mode overlap --layout "$L/$f.json#0" --seq-len 1048576 $BA --iters 4 --warmup 2 --label fm_bagel
done
for spec in "qwen38_video_1024k|$Q3|q3vl30|--ulysses 8 --replicate-kv 2" "molmo2_video_1024k|$MO|molmo2|--ulysses 8" \
            "gemma3_video_1024k|$GE|gemma3|--ulysses 8" "qwen38_video_1024k|$Q38|qwen38|--ulysses 8 --replicate-kv 2"; do
  IFS="|" read -r f H lab U <<< "$spec"
  fm --mode balanced --ulysses 4 --layout "$L/$f.json#0" --seq-len 1048576 $H --iters 4 --warmup 2 --label "fm_$lab"
  fm --mode balanced $U --layout "$L/$f.json#0" --seq-len 1048576 $H --iters 4 --warmup 2 --label "fm_$lab"
done
