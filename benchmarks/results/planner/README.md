# Dense-ring planner: pricing ranges in refinement swaps

The paper's layout cost is `C = sum_s max_r W + tau R k`. Until 89ab94fa01e,
refinement swaps lowered only the work term and `tau R k` entered only when
picking the chunk count. 89ab94fa01e let swaps continue on the full `C`;
measured here, that buys nothing on H100 and costs 2.4% on one B300 case, so
it is now opt-in (`ring_chunk_assignment(..., price_ranges=True)`) and the
default plans match the old planner's on every paper configuration.

| File | What |
|---|---|
| `cpu_price_ranges.jsonl` | Old planner vs range-priced swaps on the paper's generation-ring configs: modeled `C` changes 0 to -1.44% |
| `cpu_default.jsonl` | Old planner vs the default now: identical plans on all 12 configs, faster planning |
| `h100_bagel_8192k_ab.jsonl` | 64 H100s, BAGEL video 8M, ABBA old / range-priced at R16U4, then R16U4 with 16 chunks and R32U2 |
| `h100_bagel_4096k_ab.jsonl` | 32 H100s, BAGEL video 4M, R8U4, ABBA |
| `b300_ab.jsonl` | 8 B300s, USP 4x2, exact paths: BAGEL video 1M, Qwen2.5-7B layout 256k and 1M, ABBA each |

Measured (median ms, old vs range-priced):

| Config | Old | Range-priced |
|---|---|---|
| H100 BAGEL 8M R16U4 | 2196.3, 2199.1 | 2198.4, 2198.1 |
| H100 BAGEL 8M R16U4, 16 chunks | 2236.4 | 2241.2 |
| H100 BAGEL 8M R32U2 | 2551.8 | 2542.3 |
| H100 BAGEL 4M R8U4 | 1869.2, 1873.3 | 1874.3, 1869.4 |
| B300 BAGEL 1M R4U2 | 739.7, 740.5 | 738.8, 739.4 |
| B300 Qwen2.5-7B 256k R4U2 | 80.2, 80.3 | 80.5, 80.7 |
| B300 Qwen2.5-7B 1M R4U2 | 1082.5, 1081.6 | 1107.6, 1107.5 |

On B300 the range-priced plan for the 1M Qwen2.5-7B layout holds 26 ranges per
rank instead of 14 for 0.8% better step balance; `tau` was fitted on H100 and
undercharges ranges there. Pricing ranges by default needs `tau` per GPU.

The H100 runs used `benchmarks/bench_attention.py` with the old planner tree
first on `PYTHONPATH` for "old" lines (labels `planner_old` / `planner_new`).
