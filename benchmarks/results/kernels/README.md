# Mask cost on partial tiles (H100, 2026-10-18)

How the mask predicate's cost on partially masked tiles was found and removed
(commits b3827c0b253, a3c88f759a4, 7b9e20d9f31). All runs are forward +
backward attention on 8 or 16 H100s, `bench_attention.py`, median of
5-10 iterations; A/Bs run ABBA in one session (two runs per arm).

## Finding

FA4 evaluates a `mask_mod` once per accumulator element of a partial tile.
On 8 H100s at R1U8, Molmo2 video 512k took 140.2 ms against FlashMask
Ulysses-8's 132.3 ms; a `mask_mod` that always returns true (wrong outputs,
same block sparsity) took 133.2 ms (`freemask_r1u8.jsonl`), so the per-element
mask was the whole gap. The profiles (`kernel_profiles.txt`; `profile_*`)
put FlashMask's forward kernel at 28.2 vs 29.0 ms and its backward at 82.2 vs
89.0 ms; the Ulysses all-to-alls and copies match. On ring and sparse
all-to-all steps the always-true `mask_mod` is 1.12-1.23x faster
(`freemask_ring.jsonl`).

## Fixes and their A/Bs

1. Column ranges, then hoisted (a3c88f759a4, `hoist_column_range_*`): in
   vision-group order every key's visible queries form one range
   `[lo, hi)`; loading `lo`/`hi` once per key column of a tile instead of per
   element gives 1.04-1.07x at R1U8.
2. Hoisting the position-aware predicate (8 loads per element) helped the
   forward but slowed FA4's backward tiles (`separable_*`,
   `ring_kernel_profile.jsonl`, `separable_fwd_only.jsonl`): 0.93x on BAGEL
   R4U4 at 16 GPUs. It now hoists only on forward-shaped tiles.
3. Position ranges on every ring and sparse all-to-all step (7b9e20d9f31,
   `position_ranges_*`): with each token's index in an order where every
   key's visible queries are one range, `lo[k] <= pos[q] < hi[k]` is exact in
   any token order. At R8U1 on Molmo2 mixed 1M it recovers 88% of the
   always-true bound (248.7 vs 244.7 ms, base 278.3).

Accuracy (`*_accuracy.jsonl`, `tests/distributed/check_usp.py`, 32k tokens):
every configuration's output and gradient errors against fp32 equal the
single-GPU FA4 reference's.

### position_ranges_16gpu.jsonl

| layout | GPUs | config | new ms | base ms | speedup |
|---|---:|---|---:|---:|---:|
| molmo2_mixed_2048k | 16 | R2U8 | 244.8 | 263.1 | 1.075x |
| molmo2_mixed_2048k | 16 | R2U8 sparse A2A | 179.2 | 196.6 | 1.097x |
| molmo2_video_2048k | 16 | R2U8 | 344.6 | 366.8 | 1.064x |
| bagel_mix_2048k | 16 | R4U4 | 240.6 | 296.9 | 1.234x |
| bagel_mix_2048k | 16 | R4U4 sparse A2A | 159.9 | 220.8 | 1.381x |
| qwen38_mixed_2048k | 16 | R4U4 | 350.5 | 365.4 | 1.042x |
| qwen38_mixed_2048k | 16 | R4U4 sparse A2A | 288.0 | 298.6 | 1.037x |
| gemma3_mixed_2048k | 16 | R2U8 sparse A2A W1024 | 52.0 | 65.2 | 1.253x |

### position_ranges_8gpu.jsonl

| layout | GPUs | config | new ms | base ms | speedup |
|---|---:|---|---:|---:|---:|
| molmo2_mixed_1024k | 8 | R8U1 | 248.7 | 278.3 | 1.119x |
| molmo2_mixed_1024k | 8 | R2U4 | 194.9 | 214.5 | 1.100x |
| molmo2_mixed_1024k | 8 | R2U4 sparse A2A | 184.4 | 202.0 | 1.096x |
| molmo2_mixed_1024k | 8 | R1U8 | 179.6 | 187.5 | 1.044x |

### separable_16gpu.jsonl (previous commit, both passes)

| layout | GPUs | config | new ms | base ms | speedup |
|---|---:|---|---:|---:|---:|
| molmo2_mixed_2048k | 16 | R2U8 | 265.5 | 263.0 | 0.991x |
| molmo2_mixed_2048k | 16 | R2U8 sparse A2A | 197.3 | 195.9 | 0.993x |
| molmo2_video_2048k | 16 | R2U8 | 369.3 | 366.2 | 0.992x |
| bagel_mix_2048k | 16 | R4U4 | 322.1 | 298.2 | 0.926x |

### separable_8gpu.jsonl

| layout | GPUs | config | new ms | base ms | speedup |
|---|---:|---|---:|---:|---:|
| molmo2_mixed_1024k | 8 | R8U1 | 273.3 | 279.4 | 1.022x |
| molmo2_mixed_1024k | 8 | R2U4 | 216.1 | 213.9 | 0.990x |
| molmo2_mixed_1024k | 8 | R2U4 sparse A2A | 202.3 | 201.5 | 0.996x |
| molmo2_mixed_1024k | 8 | R1U8 | 179.3 | 187.9 | 1.048x |

### hoist_column_range_r1u8.jsonl

| layout | GPUs | config | new ms | base ms | speedup |
|---|---:|---|---:|---:|---:|
| molmo2_video_512k | 8 | R1U8 | 134.6 | 140.6 | 1.045x |
| molmo2_mixed_512k | 8 | R1U8 | 87.7 | 92.5 | 1.055x |
| molmo2_interleaved_512k | 8 | R1U8 | 48.0 | 51.1 | 1.065x |
| molmo2_video_1024k | 8 | R1U8 | 259.7 | 269.3 | 1.037x |
| molmo2_mixed_1024k | 8 | R1U8 | 180.1 | 188.2 | 1.045x |

### freemask_ring.jsonl (always-true mask_mod; wrong outputs)

| layout | GPUs | config | new ms | base ms | speedup |
|---|---:|---|---:|---:|---:|
| molmo2_mixed_1024k | 8 | R8U1 | 244.7 | 278.3 | 1.137x |
| molmo2_mixed_1024k | 8 | R2U4 | 190.0 | 213.3 | 1.123x |
| molmo2_mixed_1024k | 8 | R2U4 sparse A2A | 180.7 | 201.8 | 1.117x |
| molmo2_interleaved_1024k | 8 | R8U1 | 168.5 | 199.3 | 1.183x |
| molmo2_interleaved_1024k | 8 | R2U4 | 102.9 | 126.2 | 1.227x |
| molmo2_interleaved_1024k | 8 | R2U4 sparse A2A | 93.3 | 113.5 | 1.216x |
| qwen3vl_mixed_1024k | 8 | R8U1 | 301.2 | 323.4 | 1.074x |
| qwen3vl_mixed_1024k | 8 | R2U4 | 245.7 | 263.7 | 1.074x |
