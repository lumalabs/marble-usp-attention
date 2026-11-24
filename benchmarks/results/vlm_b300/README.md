# Vision-language masks on 8x B300, October 15 2026

One 8x B300 node, FA4 4.0.0b34, cutlass-dsl 4.7, PyTorch 2.11 cu129, code at
da917cf59bd. Ring configurations run with `sweep.py --combine fp32` (the online combine kernel
is SM90 only; labels keep the `online` names). MagiAttention de13b2a with its `FFA_FA4`
Blackwell backend (`MAGI_ATTENTION_KERNEL_BACKEND=fa4`). Layouts as in `../vlm_h100/`.
`vlm_ablation` and `video_group8` ran only the Ulysses-8, USP 2x4 / 4x2, ring + PTRR and
naive ring rungs.

| file | what |
|---|---|
| `check_b300.jsonl` | accuracy vs fp32 and single-GPU FA4, every mask, ring 8 / USP 2x4 / Ulysses 8 |
| `vlm_ablation.jsonl`, `video_group8.jsonl`, `torchcp8_vlm.jsonl` | 32 heads: our rungs and PyTorch flex all-gather |
| `vlm8.jsonl`, `video8.jsonl`, `video_group8.jsonl` | Gemma 3 4B / 12B / 27B and Molmo2-8B head layouts |
| `magi8_vlm32.jsonl`, `magi8_vlm.jsonl`, `magi_tune/` | MagiAttention defaults and tuning |
| `model8_vlm.jsonl` | Molmo2-8B's language model (Qwen3-8B, 36 layers) end to end |

Failed rows keep their `error` code; their stderr tails were dropped.

Findings:

- Accuracy: the dense paths (Blackwell's default for image and video masks) cost 1.10-1.20x
  the single-GPU output error and 1.2-1.4x on gradients; the block-sparse ring with the fp32
  merge (vision groups) is 0.96-1.00x.
- Gemma 3 4B / 12B (head_dim 256) fail on every configuration of ours: FA4's SM100 backward
  at head_dim 256 does not support local attention, which one dense-path call uses
  ("SM100 backward with head_dim=256 does not support local attention"). MagiAttention runs
  them. A head_dim 256 fallback on Blackwell is a follow-up.
- MagiAttention's `FFA_FA4` backend hits an illegal memory access on the vision groups:
  the image vision group at 512k, the video vision group at 128k and Molmo2-8B at 256k and
  1M. The image vision group at 128k ran untuned (5.5 s, 113x our time); its tuning screen
  timed out.
- Tuned, MagiAttention is 1.04-1.4x our time on the image and video masks (32 heads) and
  1.1-1.25x on Gemma 3 27B; PyTorch flex all-gather is 2.5-4.4x.
- With 8 KV heads or more, Ulysses-8 is our fastest layout on one B300 node for every VLM
  mask (32 heads). USP is 3-41% slower and ring + PTRR 1.1-2.9x slower. On the image and
  video masks the dense ring runs the same with or without PTRR; on the vision groups
  (block-sparse) PTRR makes the ring 1.4-2.0x faster than contiguous shards.
