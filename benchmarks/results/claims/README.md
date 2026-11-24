# Paper claims: CPU reproductions

Each file here comes from a CPU-only script in `benchmarks/`. The scripts rebuild
the planner's own plans; no GPU, FA4 or process groups are needed. Run them from
`projects/usp_attention` with `PYTHONPATH=src:benchmarks`.

## 1. Remote keys the sparse all-to-all avoids (seqdim.tex)

Paper: "on our tested sequences we avoid sending 55--100% of the remote keys
that ring attention would send."

```
python benchmarks/claim_remote_keys.py --out benchmarks/results/claims/remote_keys.jsonl \
    --check benchmarks/results/rerun_oct26/mlsys > benchmarks/results/claims/remote_keys.txt
```

- Definition: `1 - fetched / ((R - 1) * seq / R)` per ring rank. The fetched keys
  come from the gather's own plan: the block-sparse halo (128-token blocks), and
  also the token-exact dense gather where the planner would use it (Qwen3.8 /
  Qwen3-VL, head_dim 256).
- Packs: every gemma3, gemma3ps, molmo2, qwen38 and qwen3vl pack at 1-4M tokens.
  Rings: 4, 8 and 16.
- Validation: ring rank 0's fetch equals the `halo_len` logged by all 43
  measured gather runs in `rerun_oct26/mlsys`.
- Result: on the 43 measured (pack, ring) configurations the range is
  **54.7-100%**. Over every pack and ring size it is 50.4-100%; the minimum is
  R=16 on the 1M-token Qwen video packs, a configuration that was not measured.
  At R=16 with packs of 2M tokens or more, the minimum is 65.4%. Each row
  reports the minimum, median and maximum over ranks.

## 2. Partial tiles with vision-group order (seqdim.tex)

Paper: "partially marked tiles on Molmo2 video at 512k tokens drop from 49% to 4%."

```
python benchmarks/claim_partial_tiles.py --out benchmarks/results/claims/partial_tiles.jsonl
```

- Definition: partial tiles as a share of non-empty 128x128 tiles (empty tiles
  are excluded), over the whole-sequence mask.
- Mesh: ring plans permute only whole 128-token blocks, so every ring size and
  every Ulysses degree has the same share.
- Validation: `tests/test_claim_partial_tiles.py` checks the counts against the
  dense reference mask.
- Result: molmo2_video_512k goes from **49.4% to 3.7%**. All Molmo2 video packs
  (128k-4M) go from 49% to 4-5%. Molmo2 mixed goes from 31-44% to 6-7%, and
  Molmo2 interleaved from 16-20% to 10-17%. The built-in `vlm_group` and
  `video_group` layouts at 512k go from 33% / 38% to under 1%.

## 3. Chunked-ring cost model (ring.tex)

Paper: "We fit tau on nine BAGEL runs at 1--8M tokens, and C predicts the
runtime eight out of nine times within 3%."

```
python benchmarks/fit_chunk_cost.py benchmarks/results/rerun_oct26/mlsys/*.jsonl \
    benchmarks/results/rerun_oct26/extra/QL.jsonl --out benchmarks/results/claims/chunk_cost_fit.jsonl
```

- Plans: each run's exact plan is rebuilt, with the same ring size, chunk count
  and refine flag. Every rebuilt plan's `ring_imbalance` equals the logged value.
- Counting: `W` is counted exactly.
- Fit: `runtime = a * C`, by least squares on relative error.
- Runs: the nine paper sources (bagel32k:1-4, bagel32g:5, bagel64d:1,
  bagel64e:1-3), with repeats averaged.
- With the planner's tau = 6.4e6 pairs, 8/9 runs are within 3%. The miss is
  2M/R8 at +3.08%.
- A fresh fit gives tau = 1.14e7 pairs, and 9/9 runs are within 3% (max 1.75%).
  Leave-one-out also gives 9/9 (max 2.28%).
- `chunk_cost_fit_all_rows.jsonl` (`--all-rows`) fits all 16 rows. With the
  fresh fit (tau = 1.21e7) all 16 are within 3%. With tau = 6.4e6, 15/16 are.
