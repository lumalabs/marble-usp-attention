# Preliminary AMD packed sparse forward

Historical baseline. See [cached partial-tile mask results](AMD_SPARSE_BITMAP.md)
for the subsequent kernel optimization and updated timings.

This appendix experiment ports the general packed sparse path to the existing
AMD CDNA3 online kernel. It does **not** measure training steps or backward,
compare against NVIDIA hardware, or use two-call dense-varlen decompositions.

## Implementation and validation

- Native HIP BCSR traversal on 384-query × 64-KV tiles; empty tiles are skipped,
  full tiles bypass the packed predicate, and partial tiles use separate Q/KV
  segment, noise-group, document and original-position tracks.
- Online fp32 numerator/max/sum state survives KV rotations. Empty steps preserve
  it, and fully masked output rows are zero. Output is normalized and cast once.
- The existing conservative classifier accepts rectangular tiles. PTRR uses
  nonempty native tile counts, then applies the same block permutation to Q/K/V
  and metadata. Plans are reused across attention calls.
- Explicit forward-only Ulysses × ring wrapper, using existing all-to-all and
  process groups with overlapped RCCL P2P rotation. No backward integration.
- Separate dense/sparse compiled specializations keep the new sparse traversal
  and masking work out of existing dense calls.

The unit/regression command below checks dense and sparse MHA/GQA/MQA, BSHD/BHSD,
all rounding modes, ragged tiles, multiple batches sharing a mask, permuted
positions, multiple noise groups, follow-up text, empty steps and fully masked
rows, plus the shared block classifier. Every benchmark configuration additionally
checks **all rows at 6,144 tokens** against independent fp32 attention, then checks
17 evenly spaced local query rows per rank at the **actual timed length**, against
all keys. Reported errors are maximum-over-ranks relative L2 errors; they are not
the mean-relative error metric used in the paper. No assertion of bitwise or identical-error
parity with one GPU is made.

## Setup and measurement

Eight gfx942 GPUs, 256 GiB each, SKU `M3250101` (MI325X). PyTorch reports the generic
name `AMD Radeon Graphics`; the product evidence is preserved in the environment
JSON. PyTorch `2.11.0+rocm7.1`, HIP `7.1.52802`, RCCL `2.27.7`; hipcc and selected
Python package versions are in `results/amd_sparse_forward_environment.json`.
The amdgpu module reports version `6.16.13`. FA4, CuTe and NVIDIA NCCL are
not used by this kernel/transport.

B=1, bf16, head dimension 128, BAGEL's 28 query / 4 KV heads. All configurations
use the same native sparse backend. Ulysses-4 + DP runs two concurrent replicas;
ring-8 and USP 2×4 run one sequence. The DP replicas use identical seeded inputs
and mask shapes. This is a same-workload throughput comparison, not a training
experiment with different sampled examples.

Three warmups; three trials of ten iterations. Each iteration times the forward
with GPU events, then takes the maximum across all eight ranks. The table uses
the median of all 30 times; brackets report the min–max of the three trial
medians. All-to-alls, P2P, allocations, V transposes and online state traffic are
inside timing. Planning and initial permutation/sharding are outside, recorded
separately as rank-0 `plan_and_shard_s`. Output remains sharded and permuted.

Padding is to a common multiple of 3,072 tokens and belongs to an isolated dummy
document. Useful-token throughput excludes padding. 256k means 262,144 useful
and 264,192 physical tokens; 1M means 1,048,576 useful and 1,050,624 physical tokens.

Raw JSONL rows record all timings, git SHA/dirty flag, source and compiled-extension
hashes, metadata hash, device properties, runtime versions and communication
settings. Source commits and raw rows are the provenance; the NVIDIA project
lockfile is not an AMD environment lock.

## Results

The committed BAGEL pack has 95 documents and 1,070 spans (410 clean/full,
402 generation/noise and 258 causal). Its attended-pair density is
0.01005; 354 generation spans have following tokens in the same
sample. It exercises the general sparse predicate beyond the terminal-generation
restriction of the simple two-dense-call path.

Forward latency in ms; brackets are the range of trial medians.

| Workload | Ulysses-4 + DP | Contiguous ring-8 | PTRR ring-8 | USP 2×4 | U+DP / USP latency | USP / U+DP throughput |
|---|---:|---:|---:|---:|---:|---:|
| mixed 256k | 169.8 [169.6, 170.1] | 175.0 [175.0, 175.2] | 118.6 [118.6, 118.7] | 114.6 [114.5, 114.7] | 1.48× | 0.74× |
| mixed 1024k | 1816.4 [1815.8, 1816.8] | 1855.4 [1854.6, 1856.9] | 1001.4 [1000.8, 1001.9] | 998.8 [998.6, 999.0] | 1.82× | 0.91× |
| BAGEL pack 1024k | 215.4 [215.4, 215.5] | 196.7 [196.6, 196.8] | 127.9 [127.9, 128.0] | 135.0 [135.0, 135.0] | 1.60× | 0.80× |

For the synthetic masks, USP reduces per-sequence latency by **1.48× at 256k**
and **1.82× at 1M** versus Ulysses-4 + DP. DP still delivers approximately **35%**
and **10%** higher aggregate useful-token throughput, respectively. At 1M,
PTRR ring and USP differ by only 0.26%; that is too small to claim a meaningful
advantage for the Ulysses axis here.

On the committed BAGEL pack, **PTRR ring is fastest**, 5.2% lower latency than
USP and 1.68× faster than Ulysses+DP per sequence. USP is 1.60× faster than
Ulysses+DP per sequence, but its throughput is 0.80× DP's. This is a portability
and load-balancing result, not evidence that USP always wins on AMD.

Across the 12 configurations, sampled long-shape relative L2 error ranges from
0.002262 to 0.002346. Small-shape checks are approximately 0.00221–0.00226.
All used the same compiled extension. The measured source is commit
`899bd17a6a`; dirty flags reflect in-progress report/result files and a test-only
skip-condition change, not changes to the measured implementation.

Validation: **104 passed, 24 skipped** in the combined kernel/classifier suite
(the skipped cases are opt-in microbenchmarks). Repository pre-commit checks
passed. The GPU measurements and reference checks ran on all eight AMD devices.
There are no H100/B300 regression results for this change.

Raw rows:

- [Synthetic 256k](results/amd_sparse_forward_256k.jsonl)
- [Synthetic 1M](results/amd_sparse_forward_1m.jsonl)
- [Committed BAGEL pack 1M](results/amd_sparse_forward_bagel_pack_1m.jsonl)
- [Environment](results/amd_sparse_forward_environment.json)


## Reproduction

Use a ROCm PyTorch environment with hipcc and a gfx942 device. On this host the
project `.venv` reuses the already-installed ROCm packages through a local `.pth`
file; that machine-specific file is not committed. Do not run the NVIDIA
`uv sync --frozen` recipe to install AMD dependencies.

From `projects/usp_attention`, with the ROCm project environment at `.venv`:

```bash
(cd src/usp_attention/online_flash_amd && MAX_JOBS=4 ../../../.venv/bin/python setup.py build_ext --inplace)
PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/online_flash/test_sparse_amd.py tests/online_flash/test_online_flash_amd.py tests/ring/test_block_sparse.py -m 'not benchmark' -q

PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 benchmarks/bench_amd_sparse_forward.py --seq-len 262144 --output benchmarks/results/amd_sparse_forward_256k.jsonl
PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 benchmarks/bench_amd_sparse_forward.py --seq-len 1048576 --output benchmarks/results/amd_sparse_forward_1m.jsonl
PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 benchmarks/bench_amd_sparse_forward.py --seq-len 1048576 --layout file:benchmarks/layouts/packs/bagel_mix_1024k.json.gz#0 --output benchmarks/results/amd_sparse_forward_bagel_pack_1m.jsonl
```

Regenerate the results table from the raw rows:

```bash
.venv/bin/python benchmarks/report_amd_sparse_forward.py benchmarks/results/amd_sparse_forward_256k.jsonl benchmarks/results/amd_sparse_forward_1m.jsonl benchmarks/results/amd_sparse_forward_bagel_pack_1m.jsonl
```

Output files append one row per configuration; use fresh paths for independent
runs. A failed correctness check prevents the corresponding timing row.
`--check-only` runs the small distributed correctness matrix without timings.
File layouts use a synthetic mixed mask for that small check, then their actual
metadata for the long-shape sampled check. The committed packs are documented in
`benchmarks/layouts/README.md`; the BAGEL pack is a model-shaped generated pack,
not a recorded training batch.

## Limits and next work

This is a preliminary portability result on one eight-GPU node, not a tuned
AMD attention-library comparison. The native kernel retains its bf16 probability
packing and rescale policy; fp32 state avoids per-step output rounding but does
not imply identical output errors across different summation orders. No LSE,
backward, mixed-mask batching, AMD dense-varlen alternative, or cross-node
transport is provided here.

The planner stores a block grid and fixed-width BCSR rows, so planning/storage
remain quadratic in block count. This differs from quadratic token-mask storage
but still matters at longer lengths. State allocation and V transpose are per
call; permutation is not fused into Ulysses. The 384-token PTRR granularity and
per-rank total-work proxy do not guarantee balanced ring-step maxima. Further
optimization should retain the reference gates and report unsuccessful cases as
well as improvements.
