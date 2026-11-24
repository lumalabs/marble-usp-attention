# 8x B300 runs of October 8 2026 (raw)

FA4 4.0.0b34, cutlass-dsl 4.7.0, PyTorch 2.11 cu129; MagiAttention de13b2a
with its FFA_FA4 backend (cutlass-dsl 4.4.2, quack 0.4.1). Ring rows use
`--combine python`.

| files | code | what |
|---|---|---|
| `torchcp8_match`, `ablation`, `heads8` | 918698600f1 | forward scheduler patch, block-sparse everywhere |
| `varlen_*` | 14fef9296de | packed causal documents as varlen (Ulysses) |
| `interval_*` | 265d8cec7c6 | packed multimodal masks as two dense calls (Ulysses) |
| `interval_qwen7b_dp` | 60f3b799390 | Qwen2.5-7B layout, Ulysses-4 x 2 data-parallel sequences, dense path |
| `dense_ring_*` | c57bba20243 | dense ring steps, contiguous shards |
| `dense_ring2_*` | 6fcbd9c3b56 | dense ring, 16 chunks per rank (balanced, slower) |
| `dense_ring3_*` | 400134d759f | dense ring, fewest chunks within 3% balance (4 here) |
| `model8` | 918698600f1 | 7B multimodal MoT decoder, 28 layers end to end (block-sparse ring) |
| `magi8`, `magi8_models`, `magi_checks_models` | | MagiAttention rerun, model head layouts, GQA checks |

Qwen2.5-7B 28q/4kv USP 2x4 (ms): block-sparse 92.4 / 1212 (256k / 1M), dense ring
contiguous 75.7 / 1088.6, 16 chunks 81.1 / 1066.6, 4 chunks 76.5 / 1071.0;
MagiAttention 72.0 / 1038.8.
