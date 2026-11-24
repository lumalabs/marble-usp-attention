# Current stack on 8x B300 (October 8 2026)

Our numbers in the evaluation of the MARBLE blog post come from this folder only.

- flash-attn-4 4.0.0b34, nvidia-cutlass-dsl 4.7.0, quack-kernels 0.6.5,
  PyTorch 2.11 (cu129), PyTorch's bundled NCCL 2.28.9.
- usp_attention at commit 918698600f1 or later: 256-row Q blocks for SM100
  block sparsity, one CTA per tile for the block-sparse forward
  (`install_sm100_fwd_scheduler_patch`), 128-wide sparse KV blocks.
- Ring configurations run with `--combine python` (the online kernel is
  SM90-only); labels keep their H100 names.
- Defaults on Blackwell at 400134d759f: packed causal documents run as
  varlen and packed multimodal masks as two dense calls without a ring
  (`interval_masks`), and as dense ring steps with a ring (`dense_ring`).
  Rows replaced from `../oct8/` accordingly (`varlen_*`, `interval_*`,
  `interval_qwen7b_dp`, `dense_ring3_*`).
- October 9 (`../oct9/`, code ff92640cef3 and 4f26517d7cb: fused Triton merge,
  2.5% chunk balance threshold, contiguous KV fix): replaced the Qwen2.5-7B-layout USP 2x4
  rows of `heads8.jsonl` (`../oct9/h2h2_ours_c`, code 5c66a2ddb9e),
  the `mixed_*_ulysses8` and `mixed_*_usp_r2u4` rows of `ablation.jsonl`, and the
  `ring8_online_ptrr` / `usp_r2u4_online_ptrr` rows of `model8.jsonl` (now on the
  dense ring). `../magi8_models.jsonl` Qwen2.5-7B-layout rows are the same-session reruns
  (`h2h2_magi_c`, back to back with `h2h2_ours_c`). The `ring8_python_contig` rows of
  `model8.jsonl` are still the October 8 block-sparse contiguous ring.

Files: `torchcp8_match.jsonl` (plain causal / full), `ablation.jsonl` (mixed,
docs, causal, full at 128k / 512k, mixed 1M), `heads8.jsonl` (model head
layouts). Same environment notes as `../README.md` (no `LD_PRELOAD` NCCL).
