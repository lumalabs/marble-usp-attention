# AMD sparse forward: fused Q/KV exchange

This continues the [cached-mask measurements](AMD_SPARSE_BITMAP.md) from commit
`0484b8dde3005343ea503f3c276df9a13db1181f`. It measures forward attention on the same
eight MI325X GPUs, packed layouts, BF16 inputs, 28 query / 4 KV heads and D=128.

## Retained changes

- Sparse HIP workgroups visit the longest BCSR rows first, with heads adjacent.
  Scheduling changes physical launch order; online state keeps logical query/head
  indices. The softmax rescaling policy and FP32 online state are unchanged.
- PTRR balances all eight CP shards before Ulysses merges query windows. The ring
  path combines its three Q/K/V Ulysses transfers into one transfer.
- The optional KV-gather path routes Q to its Ulysses peers and KV to all ranks
  owning that KV head in one variable-size CP all-to-all. A Triton pack/unpack
  pair preserves batch, sequence and GQA head order. Attention remains native HIP.
- When Ulysses leaves one KV head per rank, the seven query heads can be folded
  into the query axis. Exact query metadata repeats in that order. The kernel
  uses BHSD storage; output is restored to the original sharding. Other Ulysses
  shapes use BHSD with explicit transposes.

The reported USP configuration uses a 2×4 mesh with **KV gather**, rather than two
online ring rotations. It requires full-sequence KV for its local KV head.
The ring baselines retain their KV rotations. Every configuration uses the same
optimized sparse HIP backend, including the scheduling change. Ulysses+DP uses
the same folded-query and fused-exchange options as USP.

## Method

The protocol matches the preceding reports: three warmups, three trials of ten
iterations, GPU events around the complete forward, and maximum time over all
eight ranks for each iteration. All exchanges, transposes, allocations, packing,
attention and output redistribution are timed. Planning, initial sharding,
reference checks and first-use compilation are outside timing. Plans are reused.

All configurations use the existing `rtna` probability-packing specialization;
this choice is recorded per row. The preceding report used the default `rtne`.
Both retain the same reference tolerances. No softmax-rescale approximation is
used in the retained implementation.

Each configuration first checks every row at 6,144 tokens against independent
FP32 attention, then 17 local query rows per rank against all keys at the timed
length. The metadata, seeds and isolated padding document match the earlier
reports. Useful-token throughput excludes padding. Ulysses-4+DP runs two replicas.
The committed BAGEL pack is model-shaped synthetic data, not a recorded batch.

## Results

Forward latency in ms; brackets are the range of trial medians.

| Workload | Ulysses-4 + DP | Contiguous ring-8 | PTRR ring-8 | USP 2×4 KV gather | U+DP / USP latency | USP / U+DP throughput |
|---|---:|---:|---:|---:|---:|---:|
| mixed 256k | 79.2 [79.1, 79.2] | 102.8 [102.7, 103.0] | 41.6 [41.6, 41.6] | 41.8 [41.8, 41.8] | 1.89× | 0.95× |
| mixed 1024k | 1148.4 [1148.4, 1148.5] | 1321.5 [1321.1, 1322.6] | 597.6 [597.4, 597.9] | 574.8 [574.8, 574.8] | 2.00× | 1.00× |
| BAGEL pack 1024k | 100.6 [100.5, 100.6] | 108.9 [108.8, 108.9] | 55.6 [55.6, 55.8] | 52.8 [52.8, 52.8] | 1.91× | 0.95× |

Compared with the preceding cached-mask implementation:

| Workload | Previous USP ms | New USP ms | USP speedup | Updated PTRR ring ms | USP vs updated ring |
|---|---:|---:|---:|---:|---|
| mixed 256k | 55.5 | 41.8 | 1.33× | 41.6 | 0.4% slower |
| mixed 1M | 666.4 | 574.8 | 1.16× | 597.6 | 3.8% lower latency |
| BAGEL pack 1M | 65.9 | 52.8 | 1.25× | 55.6 | 5.2% lower latency |

USP wins both 1M cases against the updated PTRR ring. At 256k the ring retains
a small 0.4% latency lead. The target of a strict win on every workload is not
achieved. Against the previous-commit PTRR ring, the new USP path is faster on
all three cases. The full table also shows the remaining Ulysses+DP throughput
advantage; its mixed-1M throughput differs from USP by only about 0.1%.

Maximum per-rank plan storage in MiB (unique storages):

| Workload | Ulysses+DP | Contiguous ring | PTRR ring | USP gather | USP bitmaps |
|---|---:|---:|---:|---:|---:|
| mixed 256k | 307.4 | 18.0 | 16.6 | 158.8 | 24.8 |
| mixed 1M | 3917.1 | 119.4 | 120.5 | 1988.3 | 109.1 |
| BAGEL pack 1M | 4825.9 | 130.0 | 124.5 | 2428.8 | 549.6 |

All 12 configurations passed both reference gates. Sampled long-shape relative
L2 error ranges from 0.0022614 to 0.0023455.

All rows were measured on the working tree based on
`0484b8dde3005343ea503f3c276df9a13db1181f`, with `dirty=true`.
Identical source and binary fingerprints were verified across all 12 rows:

- AMD source: `99c43dbbd7e9ddb81e87d0210af9baa17d959f995891201d16f3d02ccd560dd4`
- Compiled extension: `12ce1adfb932473ddea05393c1412bee39757d2dda82c8a25702e552a613e47b`

Raw data: [mixed 256k](results/amd_sparse_gather_256k.jsonl),
[mixed 1M](results/amd_sparse_gather_1m.jsonl), and
[BAGEL pack 1M](results/amd_sparse_gather_bagel_pack_1m.jsonl).

## Memory and limits

Query-head folding expands the query metadata and fixed-width BCSR grid by up to
the folding factor. Gathering KV also increases each call's KV allocation.
Partial bitmaps remain exact and shared across the kernel's heads/batches, but
folding duplicates query rows within those bitmaps. Each raw row records both
`max_rank_partial_bitmap_bytes` and `max_rank_plan_bytes`; the latter counts unique
plan tensor storages, including metadata, BCSR, bitmaps and permutation. It does
not include Q/K/V, online state or transient communication buffers.

This is a cached-plan, forward-only, single-node result. It adds no backward,
NVIDIA regression result, cross-node transport or guarantee for other shapes.
Planning/storage remain quadratic in native block counts. The ring path remains
available when gathering KV or storing folded plans is undesirable.

## Validation

The combined sparse/dense/classifier suite passed **122 tests**, with 24 opt-in
benchmarks skipped. New checks cover folded MQA with multiple batches and
permuted documents, different query schedules between online chunks, scaled
inputs, rejected unsupported storage, and 32-/64-bit exchange address paths.
The distributed exchange check passed Ulysses degrees 1, 2 and 4 on eight GPUs,
with two batches, multiple local KV heads and noncontiguous inputs.

RTNE/RTNA scaled-input tests retain the elementwise FP32-reference gate. RTZ uses
a 1% relative-L2 gate for the new scaled-input stress cases: rebuilding the
original commit confirmed it also fails the elementwise gate on those cases.
Existing RTZ regression tests and benchmark tolerances were not relaxed.
Repository pre-commit checks passed.

## Reproduction

Use the ROCm environment and extension-build instructions from
[the original appendix](AMD_SPARSE_FORWARD.md). From `projects/usp_attention`, use
fresh output files; the benchmark appends rows:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 benchmarks/bench_amd_sparse_forward.py --pack-mqa --kv-exchange gather --round-mode rtna --seq-len 262144 --output benchmarks/results/amd_sparse_gather_256k.jsonl
PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 benchmarks/bench_amd_sparse_forward.py --pack-mqa --kv-exchange gather --round-mode rtna --seq-len 1048576 --output benchmarks/results/amd_sparse_gather_1m.jsonl
PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 benchmarks/bench_amd_sparse_forward.py --pack-mqa --kv-exchange gather --round-mode rtna --seq-len 1048576 --layout 'file:benchmarks/layouts/packs/bagel_mix_1024k.json.gz#0' --output benchmarks/results/amd_sparse_gather_bagel_pack_1m.jsonl
.venv/bin/python benchmarks/report_amd_sparse_forward.py benchmarks/results/amd_sparse_gather_256k.jsonl benchmarks/results/amd_sparse_gather_1m.jsonl benchmarks/results/amd_sparse_gather_bagel_pack_1m.jsonl

PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/online_flash/test_sparse_amd.py tests/online_flash/test_online_flash_amd.py tests/ring/test_block_sparse.py -m 'not benchmark' -q
PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 tests/distributed/check_amd_exchange.py
```
