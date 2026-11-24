# 8x B300 runs of October 9 2026

Same node and stack as `../oct8/` (FA4 4.0.0b34, cutlass-dsl 4.7.0, PyTorch
2.11 cu129; MagiAttention de13b2a with its FFA_FA4 backend). Our rows use
`--combine python`.

| files | code | what |
|---|---|---|
| `parity.jsonl` | ff92640cef3 | Qwen2.5-7B 28q/4kv USP 2x4 dense ring: default, `USP_DENSE_RING_DKV_OVERLAP=1`, `USP_DENSE_RING_INPLACE=1` |
| `qwen7b_dense*`, `qwen7b_blocksparse` | ff92640cef3 | Qwen2.5-7B 28q/4kv USP 2x4, one switch at a time (2 trials) |
| `interval`, `mixed_usp_dense` | ff92640cef3 | 32-head mixed mask, Ulysses-8 two-call path and USP 2x4 dense ring |
| `model8` | ff92640cef3 | 7B multimodal MoT decoder end to end, USP 2x4 dense ring (ring 8 rows failed, fixed in 4f26517d7cb) |
| `h2h_*` | ff92640cef3 | same-session head to head, interleaved ours / Magi / ours, 5 trials each |

Head to head, Qwen2.5-7B 28q/4kv, fwd+bwd ms:

| | ours, 4 chunks | ours, 8 chunks | ours, 16 chunks | MagiAttention |
|---|---|---|---|---|
| 256k | 72.0 / 72.2 | 74.9 | 76.0 | 72.3 / 72.5 |
| 1M | 1050.8 / 1050.8 | 1039.5 | 1046.6 | 1043.7 / 1042.8 |

The 2.5% chunk balance threshold (4f26517d7cb) picks 4 chunks at 256k and 8
at 1M. On the 32-head mixed mask 8 chunks do not help (512k 338.5 vs 334.9 ms,
1M 1252.4 vs 1253.1), and the threshold keeps 4 at 512k.

Switches at 1M (Qwen2.5-7B layout, 2 trials): fused merge 1046.8 vs PyTorch merge 1060.6;
overlapped dK/dV 1054.8 (slower, stays opt-in); in-place query rows 1063.8
(slower, stays opt-in); block-sparse ring 1212.3.

## Afternoon: fused gradient accumulation (5c66a2ddb9e)

CUDA-event timing of the dense ring (Qwen2.5-7B layout, 256k: 16.1 ms forward, 49.1 ms
backward, of which 3.4 ms was fp32 gradient accumulation) led to a fused
Triton accumulate. `acc_parity.jsonl`: ring 2x4, ring 8 and Ulysses-8
interval checks unchanged. `acc_acc*`: Qwen2.5-7B layout 70.0 / 70.2 ms (256k), 1028.6 /
1029.4 ms (1M). `acc_interval_u8`: 32-head mixed Ulysses-8 22.2 / 304.3 ms.

Back to back (`h2h2_*`), MagiAttention then ours, 5 trials each:

| Qwen2.5-7B 28q/4kv | ours | MagiAttention |
|---|---|---|
| 256k | **70.4** | 72.3 |
| 1M | **1028.8** | 1040.9 |

Tried and dropped (no gain or slower): FA4's CLC scheduler (`FA_CLC=1`),
`pack_gqa=False` in the forward, MHA-expanded K/V in the backward (75.5 /
1059.7 ms, larger dK error), USP 4x2 (76.3 / 1079.4 ms).
