# Vision-language masks on H100, October 14-15 2026

8x H100 nodes (up to 7 at once), FA4 4.0.0b34, cutlass-dsl 4.7, PyTorch 2.11
cu129, code at 89fc4b8e2aa. MagiAttention de13b2a with its default Hopper FFA kernel.
Two-node runs set `NCCL_NET_PLUGIN=none`: the image's AWS OFI plugin otherwise sends
cross-node NCCL over Libfabric TCP (2 GB/s instead of 421 GB/s on RoCE).

Layouts (`usp_attention.layouts`): `vlm` (8 interleaved image-text documents, every
256-token crop bidirectional, as in Gemma 3), `video` (2 long videos of 256-token frames),
`vlm_group` and `video_group` (all image tokens of a document one bidirectional group, as in
Molmo2). Mask density at 1M: 0.10, 0.37, 0.15, 0.73.

| file | sweep / script | what |
|---|---|---|
| `check_vlm.jsonl`, `check_group.jsonl`, `check_vgroup.jsonl` | `tests/distributed/check_usp.py` | accuracy vs fp32 and single-GPU FA4, 32k tokens, every combine and split |
| `check_vlm256.jsonl` | `check_usp.py --no-fa4-reference` | head_dim 256 on the dense paths, vs fp32 only (single-GPU FA4 needs block sparsity) |
| `vlm8.jsonl`, `video8.jsonl`, `vlm16.jsonl`, `video16.jsonl` | `vlm8`, `video8`, `vlm16`, `video16` | Gemma 3 4B / 12B / 27B and Molmo2-8B head layouts, every legal split, 8 and 16 GPUs |
| `vlm8_dense.jsonl` | `vlm8_dense` | Gemma 3 27B with the dense paths forced on |
| `vlm8_hd128.jsonl`, `vlm16_hd128.jsonl` | `vlm8_hd128`, `vlm16_hd128` | Gemma 3 4B / 12B head counts at head_dim 128 (control, see below) |
| `vlm_ablation.jsonl` | `vlm_ablation` | 32-head ladder on `vlm`, `video`, `vlm_group` |
| `video_group8.jsonl` | `video_group8` | 32-head ladder and Molmo2-8B on `video_group` |
| `torchcp8_vlm.jsonl` | `torchcp8_vlm` | PyTorch flex all-gather context parallelism, 32 heads |
| `magi8_vlm32.jsonl`, `magi8_vlm.jsonl` | `magi8_vlm32`, `magi8_vlm` | MagiAttention defaults, 32 heads and model layouts |
| `magi_tune/` | `baselines/sweep_magi.py --cells-set vlm` | MagiAttention tuning (chunk size x overlap degree, then switches), winners in `magi8_tuned` / `magi8_models_tuned` |
| `model8_vlm.jsonl` | `model8_vlm` | Molmo2-8B's language model (Qwen3-8B, 36 layers) end to end on `vlm_group` |
| `diag_video.jsonl` | ad hoc | Gemma 3 4B heads, head_dim 256 vs 128, every layout, 512k and 1M |

Failed rows keep their `error` code; their stderr tails were dropped (they name pods).

Findings:

- The block-sparse ring with the online kernel is 1.000x the single-GPU FA4 error on all four
  masks at every split. Per-step bf16 merges cost 1.17-1.65x on the output, the dense ring
  1.11-1.20x.
- MagiAttention's Hopper kernel asserts head_dim <= 128: it cannot run Gemma 3 4B / 12B.
  Its single-process tuning screen runs out of memory at 512k-1M; `sweep_magi.py` now
  retries each setting in its own process.
- FA4's SM90 backward at head_dim 256 slows with call length. Traces of 1M `video`, Gemma 3
  4B heads: forward 0.55 s in every layout; backward 3.3 s (ring 8), 6.0 s (USP 2x4), 7.2 s
  (Ulysses 8 with replicated KV). head_dim 256 runs reach 170-370 TFLOP/s per GPU, head_dim
  128 about 600 in every layout. Long-call layouts (Ulysses) are penalized most, so the
  head_dim 128 control grids give the like-for-like head-cap comparison.
- `nccl4py` rows on some 16-GPU configs fail because nccl4py loaded NCCL 2.28.9 (it needs
  2.29.7); the default NCCL transport runs.
