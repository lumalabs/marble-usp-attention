# Dense ring on 8x H100 (October 8 2026, commit 30f05e31f1f)

Qwen2.5-7B 28q/4kv mixed mask, `--combine python`, `--dense-ring on` vs `off`
(PTRR + block sparsity). FA4 b21.

| ms | dense ring | block-sparse |
|---|---|---|
| USP 2x4, 256k | **152.3** | 156.1 |
| USP 2x4, 1M | **2259** | 2314 |
| ring 8, 256k | 202.9 | **178.2** |
| ring 8, 1M | 2418 | **2408** |

A pure ring of 8 (KV-only traffic, like MagiAttention) is slower with dense
steps: eight steps of calls and merges cost more than the dense kernels save.
`parity_ring8.jsonl`: ring 8 dense check on the Qwen2.5-7B head layout (output
error 2.54e-3 vs 2.11e-3 single-GPU, python-combine level).

## Later on October 8: fused merge and dK/dV exchange

Same pod and session, Qwen2.5-7B 28q/4kv USP 2x4, `--dense-ring on` (ms):

| file | commit | 256k | 1M |
|---|---|---|---|
| `oct8_qwen7b_f900.jsonl` | f900b2ae960 (before) | 149.8 | 2250.7 |
| `oct8_qwen7b_overlap_torchmerge.jsonl` | 685b715c306, `USP_DENSE_MERGE=torch` | 150.7 | 2284.8 |
| `oct8_qwen7b_overlap.jsonl` | 685b715c306 (overlapped dK/dV exchange) | 145.8 | 2266.0 |
| `oct8_qwen7b_fused.jsonl` | ff92640cef3 (default: fused merge, end reduce-scatter) | **145.7** | **2233.0** |

The fused Triton merge saves 2-3%. Sending each owner's dK/dV during the
remaining backward steps hides the reduce-scatter, but its NCCL kernel shares
SMs with those steps and is slower at 1M, so it is opt-in
(`USP_DENSE_RING_DKV_OVERLAP=1`). `oct8_parity_def.jsonl`: ring 2x4, 4x2 and
8 checks at ff92640cef3, error ratios to one GPU unchanged (2x4: 1.07x output,
1.13x dQ).
