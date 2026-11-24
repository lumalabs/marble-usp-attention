# Current stack on 8x H100 (October 8 2026)

Our numbers in the evaluation of the MARBLE blog post come from this folder only.

- flash-attn-4 4.0.0b21, nvidia-cutlass-dsl 4.6.0 (the lockfile), PyTorch
  2.11 (cu129), plus flash-attn-3 3.0.0 for native causal Ulysses
  (`USPConfig.fa3_on_hopper`).
- `torchcp8_match.jsonl`: plain causal / full. Causal rows were rerun on
  October 8 with FA3 installed; full rows are
  the October 6 FA4 runs (FA3 is slower than FA4 on full masks and is not
  used there).
- `ablation.jsonl`: mixed and docs rows of `../ablation.jsonl` (block-sparse
  path, unaffected by FA3 or the SM100-only forward scheduler patch).
- `heads8.jsonl`: copy of `../heads8.jsonl`.

FA4 4.0.0b34 was also measured on H100 (`../fa4_b34/`): block-sparse masks
unchanged, causal 2-3% faster (same as FA3), full masks 2-4% slower, so b21
stays the H100 stack.
