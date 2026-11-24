# FlashMask baseline runs

These are the attention-only launchers behind every FlashMask number in
`benchmarks/MLSYS_RESULTS.md`: FlashMask CP, the FM + Ulysses hybrid, and
FlashMask's Blackwell overlap path. All three use one script,
[`../flashmask_cp_bench.py`](../flashmask_cp_bench.py), run in FlashMask's own
venv, not ours.

| Baseline | Flags | `impl` in the rows |
|---|---|---|
| FlashMask CP (FlashMask's balancer, K/V all-gather, dK/dV reduce-scatter) | `--mode balanced` (`--ipo` on H100) | `flashmask_cp_balanced_allgather` |
| FM + Ulysses (Ulysses over U consecutive ranks, FlashMask CP across the groups; our construction) | `--ulysses U` (`--replicate-kv 2` when U exceeds the KV heads) | `flashmask_cp_balanced_allgather_ulyssesU` |
| FlashMask overlap (dual-chunk, one Blackwell node) | `--mode overlap` | `flashmask_cp_overlap_dualchunk` |

Masks come from `../flashmask_masks.py`, which converts our layouts to
FlashMask's per-column row ranges exactly (documents with vision groups are
reordered once, `reordered_for_groups` in the row).

## Setup

```bash
benchmarks/baselines/flashmask_runs/setup_flashmask.sh h100 ..   # or b300
```

This builds FlashMask at commit `6a150fc` with the package versions of the
original venvs (pinned in the script; the H100 and B300 venvs matched). By
default the scripts below expect `../flashmask` and `../fm-venv` next to the
project root.

## H100 (32-128 GPUs)

Each list line is one argument set for `flashmask_cp_bench.py`. Start
`run_list.sh` on every node at the same time, each with its own node rank:

```bash
# node r of N, from the usp_attention project root
benchmarks/baselines/flashmask_runs/run_list.sh r N MASTER_IP \
    benchmarks/baselines/flashmask_runs/lists/fmu32b.list
```

It writes `benchmarks/results/rerun/<list>.jsonl`, which you can compare with the
committed `benchmarks/results/mlsys/<list>.jsonl`. Lines with no committed row
produced no result (no `RESULT` line), and the tables show `–` for them.
`report_tables.py` takes the best FlashMask row per cell across all files.

| List | GPUs | What | Rows |
|---|---|---|---|
| `fm32`, `fm64`, `q128_fm` | 32, 64, 128 | FlashMask CP, Qwen3.8-27B and Qwen3-VL-30B-A3B (`q128_fm` also Molmo2-8B) | 8/8, 8/8, 3/5 |
| `mg32_fm4`, `mg64_fm` | 32, 64 | FlashMask CP, Molmo2-8B and Gemma 3 27B (global and sliding window) | 12/12, 6/12 |
| `mg32_fm` | 8 | the same list, run on one node; `mg32_fm4` is the 32-GPU run | 8/12 |
| `fmu32`, `fmu32b`, `q38k_fm` | 32 | FM + Ulysses (U = 4, 8, 16), all models | 18/18, 20/20, 2/2 |
| `fmu64` | 64 | FM + Ulysses (U = 4, 8, 16) | 32/32 |
| `fmu128` | 128 | FM + Ulysses (U = 8) | 8/8 |
| `bagel_fm32`, `bagel_fm32e` | 32 | BAGEL mix and video, 1-4M; CP and hybrid | 12/12, 2/2 |
| `bagel_fm64`, `bagel_fm64b` | 64 | BAGEL video 8M (`b` re-measures in the same session as `bagel64e`) | 2/2, 2/2 |
| `s8_fm_a`, `s8_fm_b` | 8 | FlashMask CP, one node, 512k and 1M | 14/14, 9/14 |

Multi-node runs set `NCCL_NET_PLUGIN=none`: on our pods the libfabric plugin
fell back to TCP (about 2 GB/s) across nodes, and torch's bundled NCCL 2.28.9
handled the inter-node traffic.

## B300 (one node)

```bash
benchmarks/baselines/flashmask_runs/b300.sh WORKDIR   # WORKDIR from setup_flashmask.sh b300
```

This runs every FlashMask command from the B300 evaluation, including the
overlap path, with NCCL 2.32.3 from `WORKDIR/nccl13` on `LD_LIBRARY_PATH`. The
B300 runs used FlashMask's default balancer (no `--ipo`). The script's
three parts match `benchmarks/results/mlsys/b300_eval.jsonl`, `b300_eval2.jsonl`
and `b300_eval3.jsonl`, where the FlashMask rows sit next to ours.

## FlashMask-3 on Hopper (PaddlePaddle)

FlashMask-3's C++ Hopper kernel ships in PaddlePaddle 3.4 (`FLAGS_flash_attn_version=3`).
[`../flashmask3_paddle_bench.py`](../flashmask3_paddle_bench.py) times it, attention
only, with the same arguments as `flashmask_cp_bench.py`:

| Baseline | Flags | `impl` in the rows |
|---|---|---|
| FM3 CP: PaddleFleet's balanced all-gather (dual-chunk split, K/V all-gather, dK/dV reduce-scatter) | default (`--mode allgather`) | `flashmask3_paddle_cp_balanced_allgather` |
| FM3 + Ulysses (our construction, as for FlashMask-4) | `--ulysses U` | `..._ulyssesU` (no `cp_` when the CP group has one rank) |
| FM3 overlap forward: NVSHMEM overlap that fetches only the needed K/V, all-gather backward | `--mode overlap_fwd` | `flashmask3_paddle_cp_overlap_fwd_allgather_bwd` |
| FM3 overlap forward and backward (backward unverified, see below) | `--mode overlap` | `flashmask3_paddle_cp_overlap_unverified_bwd` |

```bash
benchmarks/baselines/flashmask_runs/setup_paddle.sh paddlepaddle_gpu-3.4.0-cp312-cp312-linux_x86_64.whl ..
# node r of N, from the project root
benchmarks/baselines/flashmask_runs/run_paddle_list.sh r N MASTER_IP \
    benchmarks/baselines/flashmask_runs/lists/fm3_64_hyb.list
```

Checks (`--check`, fp32 reference on 256 sampled rows): every mode is within
2.1-2.3e-3 of fp32 on 8 H100s (Molmo2 video 1M, vision-group 512k), and plain
CP and the hybrid also on 16 GPUs. The overlap backward
(`flashmask_attention_v2_grad`) matched the all-gather gradients to 7.5e-5 on all
8 ranks of Molmo2 video 1M, but disagreed by 2-26% on ranks 1-6 of an earlier
16-rank configuration, so `--mode overlap` rows stay labeled unverified. The
overlap needs local lengths of 4k-128k tokens and at most 64 ranks.

Results are in `benchmarks/results/flashmask3/` (not in `results/mlsys/`, so the
FlashMask-4 tables keep their own rows):

| List | GPUs | What |
|---|---|---|
| `fm3_32_plain_a`, `fm3_32_plain_b` | 32 | FM3 CP |
| `fm3_32_hyb_a`, `fm3_32_hyb_b` | 32 | FM3 + Ulysses (U = 4, 8, 16) |
| `fm3_64_plain`, `fm3_64_hyb` | 64 | FM3 CP, FM3 + Ulysses |
| `fm3o_64_plain`, `fm3o_64_hyb` | 64 | overlap forward (runs where local lengths fit) |
| `fm3u_64_plain` | 64 | overlap forward and backward (unverified backward) |

The one-node, same-session comparison (FlashMask-4, FlashMask-3 and ours) is in
`benchmarks/results/single_node/`.
