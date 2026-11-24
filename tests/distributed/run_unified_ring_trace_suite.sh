#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
set -euo pipefail

# Run the unified ring attention trace captures
# (profile_unified_ring_traces.py in this directory).
#
# Common overrides:
#   NPROC_PER_NODE=8 SEQ_LEN=8192 OUT_DIR=profiler_traces ./run_unified_ring_trace_suite.sh
#   RUN_SYMM_DIRECT=0 RUN_NCCL4PY=0 ./run_unified_ring_trace_suite.sh
#   RECORD_SHAPES=1 WITH_STACK=1 ./run_unified_ring_trace_suite.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRACE_SCRIPT="tests/distributed/profile_unified_ring_traces.py"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
SEQ_LEN="${SEQ_LEN:-32768}"
HEADS="${HEADS:-8}"
HEAD_DIM="${HEAD_DIM:-128}"
DTYPE="${DTYPE:-bfloat16}"
WARMUP="${WARMUP:-2}"
ACTIVE="${ACTIVE:-1}"
OUT_DIR="${OUT_DIR:-profiler_traces}"
BACKWARD="${BACKWARD:-0}"
ACTIVATE_VENV="${ACTIVATE_VENV:-1}"

# Optional one-sided transports are included by default because they are part of
# the presentation trace set. Set either variable to 0 to skip unavailable
# runtimes.
RUN_SYMM_DIRECT="${RUN_SYMM_DIRECT:-1}"
RUN_NCCL4PY="${RUN_NCCL4PY:-1}"

RECORD_SHAPES="${RECORD_SHAPES:-0}"
WITH_STACK="${WITH_STACK:-0}"

cd "${PROJECT_DIR}"
if [[ "${ACTIVATE_VENV}" == "1" && -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

COMMON_ARGS=(
  "--out-dir" "${OUT_DIR}"
  "--seq-len" "${SEQ_LEN}"
  "--heads" "${HEADS}"
  "--head-dim" "${HEAD_DIM}"
  "--dtype" "${DTYPE}"
  "--warmup" "${WARMUP}"
  "--active" "${ACTIVE}"
)

if [[ "${BACKWARD}" == "1" ]]; then
  COMMON_ARGS+=("--backward")
else
  COMMON_ARGS+=("--no-backward")
fi

if [[ "${RECORD_SHAPES}" == "1" ]]; then
  COMMON_ARGS+=("--record-shapes")
fi

if [[ "${WITH_STACK}" == "1" ]]; then
  COMMON_ARGS+=("--with-stack")
fi

run_trace() {
  local case_name="$1"
  local transport="$2"
  local trace_name="$3"

  echo
  echo "=== ${trace_name} (${case_name}, ${transport}) ==="
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${TRACE_SCRIPT}" \
    --case "${case_name}" \
    --transport "${transport}" \
    --trace-name "${trace_name}" \
    "${COMMON_ARGS[@]}"
}

run_trace "causal_imbalance" "nccl_p2p" "unified_causal_imbalance"
run_trace "mixed_sparse_ptrr_off" "nccl_p2p" "unified_mixed_sparse_ptrr_off"
run_trace "mixed_sparse_ptrr_on" "nccl_p2p" "unified_mixed_sparse_ptrr_on"
run_trace "transport_overlap" "nccl_p2p" "unified_overlap_nccl_p2p"

if [[ "${RUN_SYMM_DIRECT}" == "1" ]]; then
  run_trace "transport_overlap" "symm_direct" "unified_overlap_symm_direct"
else
  echo
  echo "Skipping symm_direct trace because RUN_SYMM_DIRECT=${RUN_SYMM_DIRECT}"
fi

if [[ "${RUN_NCCL4PY}" == "1" ]]; then
  run_trace "transport_overlap" "nccl4py" "unified_overlap_nccl4py"
else
  echo
  echo "Skipping nccl4py trace because RUN_NCCL4PY=${RUN_NCCL4PY}"
fi

echo
echo "Trace suite complete. Outputs are in: ${PROJECT_DIR}/${OUT_DIR}"
