# NCCL 2.29.7 re-measurement (2026-10-27/28)

Every H100 number the paper cites, re-measured on this branch's code with the image's NCCL 2.29.7
preloaded (`LD_PRELOAD=/opt/nccl/build/lib/libnccl.so`; `NCCL_DEBUG=VERSION` confirms 2.29.7 at runtime,
while provenance's `nccl` field reports torch's compile-time 2.28.9, so each row carries its own `nccl` tag).
The 2.28.9 measurement of the same rows is `../rerun_oct26/`.

- `mlsys/<file>.jsonl`: 32-256 H100 rows, one file per legacy results file, `reproduction_source` = the
  legacy `file:line` each row replays (halo-overlap-on rows in `cur_halo_overlap`). Only the FlashMask 256-GPU
  rows in `b256.jsonl` that carry `nccl: 2.28.9` are carried over; `extra/b256_one_session.jsonl` re-measures
  them and our 256-GPU rows in one 2.29.7 session on the same 32 pods (784 / 1283 / 675 ms), which the paper cites.
- `MLSYS_RESULTS.md`: `report_tables.py` over `mlsys/`.
- `node/<preset>.jsonl`: one row per `benchmarks/sweep.py <preset>` label on 8 or 16 H100s.
- `extra/nccl_ab_{a,b}.jsonl`: same-pod A/B of the six largest 2.29.7 gains, alternating versions, 2 reps
  each (`study_source` without `n29:` = 2.28.9): 25--37% faster on 2.29.7 in every pair.
- `extra/combine_ring32.jsonl`: online kernel vs per-step fp32 merge (`python`) vs per-step bf16 merge
  (`flash`) on a 32-way ring, BAGEL 1--2M tokens, same pods, 2 reps: +3.0--3.9% and +5.6--9.7%.
- `extra/accuracy_combine_modes.jsonl`: `tests/distributed/check_usp.py` (32k, 16/4 heads, 4x2, every mask
  type) with `--combine online|python|flash`: output error 1.00x, 1.25--1.34x and 1.20--1.44x a single GPU's.
- `extra/dense_paths_accuracy.jsonl`: `check_usp.py` on `bagel_video_32k_f1k` with the generation (chunked) ring and the dense gather (`--balance chunks --dense-ring on --exact-dense-ring`) at 8, 16 and 32 ring steps: output error 0.67--0.68x and gradient errors within 1.5% of a single GPU's (the dense calls run in fp16 on scaled inputs; the earlier DC8 rows without these flags ran the block-sparse paths).
- `extra/strategy_selector_score.txt`: `benchmarks/fit_strategy.py --layouts <packs> mlsys/*.jsonl` on these rows: 49 scored cases, top pick within 5% of the best in 98% (94% / 92% with the model / GPU count held out), best always in the top 3 (paper Table tab:selector).
- `extra/table7_q38.jsonl`: Table 7's Qwen3.8 bf16 chunked-ring cells (abl32:13/14).
- `b300fix/`: B300 follow-ups (NCCL 2.28.9; the image's NCCL lacks sm_103): exact 7B end-to-end rows,
  the naive block-sparse ring rows, exact accuracy, fp32 parity on the block-sparse ring
  (`fp32_parity_bs`, the paper's 2.08--2.14e-3), and `ab_b300.jsonl` (current vs `--no-column-ranges` vs the
  PR base f8fd4b1, explaining the Table 8 change).

Known failures (not cited): 16-GPU nccl4py rows (cross-node window registration unsupported, as on 2.28.9);
1M-token 7B rows on 8 H100s (OOM on both versions); e2e64_pl:3 and e2e64g:2 (fail on both versions).
Two first-collective startup hangs (fmu64:17, rerun64:32) passed on rerun.

Known issue: with NCCL 2.29.7 the online block-sparse ring hangs in its first P2P on `check_usp.py --ring 16 --ulysses 1` over two nodes (32k tokens, 2k per rank), reproducibly (2 of 2); the same row passes on 2.28.9 (1.000x error, 78 s). No cited configuration hit it.
