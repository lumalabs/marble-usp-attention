# Grid gap-fill (2026-10-30)

Fills the cells that Table 4 / Table S8 left empty, on the 10-28 measurement settings (image NCCL 2.29.7 via
`LD_PRELOAD`, `NCCL_NET_PLUGIN=none`, `--halo-overlap` on our configs, FlashMask at `6a150fc`). H100 nodes of 8
GPUs, one job per `benchmarks/sweep.py` `gap*` sweep. Rows were taken from the `RESULT` lines of each job's rank-0
log; every row carries its provenance.

| file | job | GPUs | what |
|---|---|---|---|
| `gap32_fm.jsonl` | `worker-7a41e31d` | 32 | FM + Ulysses (U = 8, 16) on Gemma 3 sliding window 1M, mixed and video |
| `gap64_fm.jsonl` | `worker-48de39e5` | 64 | FM + Ulysses (U = 8, 16) on Gemma 3 sliding window 2M and 4M |
| `gap128.jsonl` | `worker-6c21a888` | 128 | ours: Gemma 3 sliding window 4M (R16U8, R32U4); BAGEL long video 8M (dense ring and sparse all-to-all, R32U4) |
| `gap128_fm.jsonl` | `worker-a6365af0` | 128 | FM + Ulysses on the sliding-window 4M packs; FlashMask CP and FM + Ulysses (U = 4) on BAGEL video 8M |
| `gapb64.jsonl`, `gapb128.jsonl` | `worker-c4842f54`, `worker-e63ad6f9` | 64, 128 | ours on BAGEL's packed mix, 2M and 4M at 64 GPUs, 4M at 128 (sparse all-to-all, U = 4) |
| `gapb64_fm.jsonl`, `gapb128_fm.jsonl` | `worker-8aaf0cb3`, `worker-2920b2cf` | 64, 128 | FlashMask CP cells that were never run (Molmo2 mixed and video 4M at 64, Molmo2 / Qwen3.8 / Qwen3-VL mixed 4M at 128); FlashMask CP and FM + Ulysses (U = 4) on the BAGEL mix |

All 38 configurations finished. FlashMask CP on Gemma 3 at 4M remains out of memory at every GPU count (not rerun).

- `MLSYS_RESULTS.md`: `report_tables.py results/rerun_oct28/mlsys --extra 'results/rerun_oct30_gapfill/*.jsonl'
  'results/rerun_oct28_scaling/rerun_magi*.jsonl'`. 60 cases; vs FM + Ulysses
  faster in 59 (1.05-12.6x), level in 1; vs FlashMask CP faster in all 52 it can run (1.23-46.2x); vs MagiAttention
  43 cases (the MagiAttention rows tune the chunk size only, overlap degree 1), faster in 41 (1.03-12.2x), level in 2.
- `strategy_selector_score.txt`: `fit_strategy.py --layouts benchmarks/layouts/packs ../rerun_oct28/mlsys/*.jsonl
  <these rows>`: still 49 scored cases; in-sample 98% within 5%, held-out model 98%, held-out GPU count 92%, top 3 100%.
- Per-doubling medians on the layouts all four systems run (2M 32->64 n = 8, 2M 64->128 n = 4, 4M 64->128 n = 4):
  ours 1.90 / 1.79 / 1.91x (1.91x over all 16, 95% parallel efficiency), MagiAttention 1.48 / 1.46 / 1.57x,
  FM + Ulysses 1.51 / 0.92 / 1.20x, FlashMask CP 0.97 / 0.97 / 1.05x.
