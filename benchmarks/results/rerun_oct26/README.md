# Full re-measurement on the release code (2026-10-26)

Every H100 number cited by the paper, re-measured on this branch's code with
provenance stamps (torch 2.11/cu129, FA4 4.0.0b34, CuTe DSL 4.7.1, NCCL 2.28.9).

- `mlsys/<file>.jsonl`: fresh rows for 32-256 H100s; each row's
  `reproduction_source` is the `file:line` of the legacy row in `../mlsys/` it
  re-measures. The sweep was trimmed to each case's top 3 legacy configurations,
  the best of every exchange mode, and every row the paper cites. `cur_*.jsonl`
  are today's earlier rows (halo overlap, same-session end to end, 256 GPUs).
- `MLSYS_RESULTS.md`: `report_tables.py` over `mlsys/` (legacy FA3-default file
  flags disabled; fresh rows all ran the FA4 paths).
- `node/<preset>.jsonl`: `benchmarks/sweep.py <preset>` on 8 or 16 H100s;
  `accuracy*.jsonl` (`tests/distributed/check_usp.py`, 32k, 16/4 heads, 4x2;
  `accuracy2` = exact chunked ring), `fmcheck.jsonl`
  (`baselines/flashmask_cp_bench.py --check`), `magicheck.jsonl`
  (`baselines/magi_attn.py --check`), `planning.jsonl` (`bench_planning.py`,
  CPU, 12 threads). `yunchang8` errored in a fresh environment (FA3 3.0.0 renamed
  `out` to `out_`); the paper keeps the legacy yunchang rows.

Known failures (not cited): e2e64_pl:3 (R2U32) hangs; e2e64g:2 and two 256-GPU
reruns hit intermittent NCCL remote errors (the 256-GPU rows in `cur_b256`
measure the same configurations).
