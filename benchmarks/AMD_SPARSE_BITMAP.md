# AMD sparse forward: cached partial-tile masks

Historical measurements. See [fused Q/KV exchange results](AMD_SPARSE_GATHER.md)
for the subsequent scheduling and distributed-forward work.

This iteration targets register pressure in the general packed sparse forward
kernel. It preserves BCSR traversal, PTRR, ring/Ulysses communication, and fp32
online accumulation. It does not add backward or use dense-varlen decomposition.

## Change and diagnostic evidence

The original kernel evaluates four metadata predicates for every score in a
partial tile. On a single MI325X, profiling a 32,768-token mixed layout with seven
query heads and one KV head attributed 99.85% of GPU time to attention, at
10.46 ms per call; V transpose took about 16 microseconds. The compiled rtna BSHD
sparse specialization reported 57 VGPR spills and 216 bytes of private segment
per thread, versus four spills for the dense online specialization.

The planner now caches exact masks only for partial tiles, using two 32-bit words
per query row. The HIP kernel loads those words and masks scores with bit tests.
Full and empty tiles allocate no bitmap. The bitmap is independent of Q/K/V and
shared across heads, batches, and layers. Original positions, document isolation,
and generation-group isolation retain the same semantics.

The first bitmap variant reduced that microprofile to 4.36 ms, with 14 VGPR spills
and 60 private-segment bytes. The final variant uses negative infinity for rejected
scores, allowing exponentiation to produce zero without per-score sentinel checks;
it measured 4.38 ms, effectively unchanged. Row maxima retain a finite sentinel
so fully masked rows do not introduce an infinity-minus-infinity operation.
An earlier attempt to disable predicate-loop unrolling was slower and discarded.
These microprofiles are diagnostic, not the distributed headline results below.

## Method

The hardware, masks, seeds, padding, correctness gates, and three-trial timing
protocol match [the preceding measurements](AMD_SPARSE_FORWARD.md). Every
configuration, including Ulysses+DP, uses the optimized sparse backend. Planning,
permutation, and initial sharding remain outside forward timing. Raw rows add
`max_rank_partial_bitmap_bytes`, the maximum total bitmap storage over all ring
steps on any rank; this excludes the existing BCSR and metadata storage.

Each partial tile costs 3 KiB. Unlike a dense token mask, full and empty tiles cost
no bitmap storage; however, an adversarial layout with every tile partial can
still require quadratic bitmap storage. This optimization should be presented
as an AMD backend tradeoff, not as the metadata-only NVIDIA planning method.

## Results

Forward latency in ms; brackets are the range of trial medians.

| Workload | Ulysses-4 + DP | Contiguous ring-8 | PTRR ring-8 | USP 2×4 | U+DP / USP latency | USP / U+DP throughput |
|---|---:|---:|---:|---:|---:|---:|
| mixed 256k | 94.6 [94.6, 94.7] | 106.7 [106.7, 106.8] | 53.6 [53.5, 53.7] | 55.5 [55.5, 55.6] | 1.70× | 0.85× |
| mixed 1024k | 1273.1 [1271.9, 1274.1] | 1385.6 [1385.4, 1385.7] | 637.7 [637.6, 638.1] | 666.4 [666.2, 666.6] | 1.91× | 0.96× |
| BAGEL pack 1024k | 114.7 [114.7, 114.7] | 116.1 [116.1, 116.2] | 59.4 [59.4, 59.6] | 65.9 [65.8, 65.9] | 1.74× | 0.87× |

Same-configuration speedup over the preceding metadata-predicate kernel:

| Workload | Ulysses+DP | Contiguous ring | PTRR ring | USP | Max bitmap MiB/rank (U+DP / contiguous / PTRR / USP) |
|---|---:|---:|---:|---:|---|
| mixed 256k | 1.79× | 1.64× | 2.21× | 2.06× | 26.4 / 7.3 / 6.0 / 13.9 |
| mixed 1M | 1.43× | 1.34× | 1.57× | 1.50× | 105.4 / 29.1 / 30.2 / 56.3 |
| BAGEL pack 1M | 1.88× | 1.69× | 2.15× | 2.05× | 238.7 / 39.7 / 34.2 / 123.1 |

PTRR ring is fastest on all three workloads. USP remains faster per sequence
than Ulysses+DP, while Ulysses+DP retains higher aggregate throughput. No claim
that USP is universally fastest is warranted. Planning/sharding times include
first-use compilation effects and are recorded per row, not included above.

## Validation and reproduction

The dense/sparse/classifier regression suite passed 104 tests (24 opt-in
microbenchmarks skipped); an additional direct bitmap reconstruction test passed,
checking exact predicate bits, both 32-bit halves, permuted metadata and padding.
All 12 distributed configurations passed both reference gates. Long-shape
sampled relative L2 error ranges from 0.0022617 to 0.0023459.
Repository pre-commit checks passed. No backward or NVIDIA regression run was
performed for this forward-only AMD change.

All rows were measured from the same working-tree implementation on base commit
`1043760d1efbb36bc0b91c385db619a5863cb36b` and explicitly record `dirty=true`.
The implementation is preserved with this report; its source fingerprint is
`d93b46bf13fd205e06cb56d94a106a1f10ddd2ee9a6c018b51569743f0666eff` and its compiled-extension fingerprint is
`0987052b9f72da604e00c26a4b7fa5d1dedee7745a9ddd7d7cf97c4f0fc36c80`. Report preparation verified all three files use
identical source and binary hashes and the same metadata hashes as their baseline.


Build the extension and use the same ROCm environment as the preceding report.
Run from `projects/usp_attention`, with a fresh JSONL output for each command:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 benchmarks/bench_amd_sparse_forward.py --seq-len 262144 --output benchmarks/results/amd_sparse_bitmap_256k.jsonl
PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 benchmarks/bench_amd_sparse_forward.py --seq-len 1048576 --output benchmarks/results/amd_sparse_bitmap_1m.jsonl
PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 benchmarks/bench_amd_sparse_forward.py --seq-len 1048576 --layout 'file:benchmarks/layouts/packs/bagel_mix_1024k.json.gz#0' --output benchmarks/results/amd_sparse_bitmap_bagel_pack_1m.jsonl
.venv/bin/python benchmarks/report_amd_sparse_forward.py benchmarks/results/amd_sparse_bitmap_256k.jsonl benchmarks/results/amd_sparse_bitmap_1m.jsonl benchmarks/results/amd_sparse_bitmap_bagel_pack_1m.jsonl
```
