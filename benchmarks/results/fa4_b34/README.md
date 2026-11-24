# FA4 4.0.0b34 on 8x H100 (October 8 2026)

flash-attn-4 4.0.0b34, nvidia-cutlass-dsl 4.7.0, quack-kernels 0.6.5, plus
flash-attn-3 3.0.0, usp_attention at 4e01737fee8 (online kernel reads b34's
block-sparse config).

- `parity.jsonl`: `tests/distributed/check_usp.py` (16k tokens) for ring 8,
  USP 2x4 and Ulysses 8 on mixed / docs / causal / full; online combine at
  single-GPU error, as with b21.
- `torchcp8_match.jsonl` (FA3 used for native Ulysses, causal and full, at
  that commit), `torchcp8_match_nofa3.jsonl` (FA4 only), `ablation.jsonl`
  (mixed / docs / causal / full).

Against b21: block-sparse masks unchanged (mixed 512k Ulysses-8 678.9 vs
678.8 ms), causal 2-3% faster (1595 vs 1637 ms at 512k, the same as FA3 on
b21), full masks 2-4% slower (3385 vs 3262 ms). The H100 stack therefore
stays on b21 with FA3 for causal masks (`../current/`).
