# Online Flash Attention 4 (AMD CDNA3)

The **AMD** counterpart to [`online_flash`](../online_flash/) (which targets NVIDIA SM90 via
`flash_attn.cute`). This one runs on **MI300X / MI325X (CDNA3, gfx942)** and drives a standalone
HIP kernel — no CUTLASS, no `flash_attn.cute` dependency.

"Online" means the same thing as in the SM90 version: the kernel can be called on a series of K/V
chunks (e.g. from Ring Attention), threading the running softmax state (`out`, `max`, `sum`)
through HBM across calls, so partial attention outputs combine without a separate RingCombine step.


## Build

Requires a ROCm build of PyTorch with `hipcc` on PATH and a gfx942 device. Build the `_C`
extension once, in place:

```sh
cd src/usp_attention/online_flash_amd
python setup.py build_ext --inplace
```

## Usage

### Chunked KV (one call per chunk)

```python
import torch
from usp_attention.online_flash_amd import online_flash_attn_func

B, Sq, Sk, H, D = 1, 1024, 8192, 4, 128
q = torch.randn(B, Sq, H, D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(B, Sk, H, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(B, Sk, H, D, dtype=torch.bfloat16, device="cuda")

k_chunks = list(k.chunk(4, dim=1))
v_chunks = list(v.chunk(4, dim=1))

# First call: state / out allocated internally.
out, state = online_flash_attn_func(q, k_chunks[0], v_chunks[0], is_online_first=True)

# Middle calls: thread the state back in.
for k_c, v_c in zip(k_chunks[1:-1], v_chunks[1:-1]):
    out, state = online_flash_attn_func(q, k_c, v_c, out=out, state=state)

# Last call: finalize `out`.
out, state = online_flash_attn_func(q, k_chunks[-1], v_chunks[-1], out=out, state=state, is_online_last=True)
```

### Convenience wrapper

```python
from usp_attention.online_flash_amd import online_flash_attn_chunks

out = online_flash_attn_chunks(q, k_chunks, v_chunks)
```

### Full (non-online) attention

```python
from usp_attention.online_flash_amd import forward

out = forward(q, k, v)  # == online over 1 chunk; also aliased offline_flash_attn_func
```

## Constraints

- bf16 inputs / bf16 outputs; `head_dim == 128`.
- Unpacked GQA and MQA require `num_q_heads >= num_kv_heads` and
  `num_q_heads % num_kv_heads == 0`. Contiguous query groups map as
  `kv_head = query_head / (num_q_heads / num_kv_heads)`; output and online state remain
  query-head-sized.
- BSHD `(batch, seqlen, num_head, head_dim)` or BHSD `(batch, num_head, seqlen, head_dim)`; no transpose either way.
- Any `seq_len ≥ 1` for Q and K/V independently (cross-attention supported).
- `softmax_scale` fixed to `1/sqrt(head_dim)`.
- The running `state` is a single opaque per-thread fp32 tensor (the `o_acc` numerator plus
  per-lane `row_max` / `row_sum`), not separate `max_` / `sum_` tensors.

## Not yet exposed

- LSE output, sliding-window, general packed-GQA optimization, backward pass.
- Dense entry points remain unmasked; causal/packed masks use a sparse plan.

## GQA test coverage

The correctness suite covers MHA, GQA ratios 2/4/8 (including MQA), and the Omni-shaped
`Hq=32, Hkv=8, D=128` case in both BSHD and BHSD. It exercises full attention, one and multiple
online chunks, uneven chunk lengths, cross-attention, every rounding mode, caller outputs, and
state/shape validation against an independent fp32 PyTorch reference. The distributed
folded-MQA option is described below; general packed-GQA optimization remains unsupported.

## Numerics

Output is within 1–2 bf16 ULP of the full-sequence kernel (and of AITER) for every `num_chunks`,
layout, rounding mode, and cross-attention shape — the online path is numerically identical to a
single full-sequence call (`num_chunks == 1` degenerates to it). On the long-sequence shapes the
base kernel targets, single-chunk online runs at ~0.97× AITER (e.g. ~536 TFLOP/s at B2 H16 S8192,
vs AITER ~554); more chunks add per-chunk state load/store + V-transpose overhead.

## What's different from `online_flash/` (the SM90 module)

- **Backend**: standalone HIP kernel (this) vs. `flash_attn.cute` / CUTLASS SM90 (that).
- **Running state**: one opaque per-thread `state` tensor (this) vs. separate `out` / `max_` /
  `sum_` tensors (that).
- **Return values**: `online_flash_attn_func` returns `(out, state)` and
  `online_flash_attn_chunks` returns `out`; the SM90 versions return
  `(out, lse, max_, sum_)` and `(out, lse)`. The two are not drop-in replacements.
- **Feature surface**: `online_flash` inherits causal / window / block-sparsity / LSE from FA4;
  this module exposes dense and packed sparse online forward attention.

## Packed sparse forward (experimental)

`online_flash_attn_func(..., sparse_plan=plan)` now supports the general packed
mask. Build a plan once per query/KV window:

```python
from usp_attention.online_flash_amd import build_sparse_plan, online_flash_attn_func

# int32 tensors of shape (4, sequence_length), on the same GPU as Q/K/V.
# Rows: bidirectional segment id, noise group id, document id, original position.
# Use -1 for tokens outside a bidirectional segment or generation group.
plan = build_sparse_plan(q_metadata, kv_metadata)
out, state = online_flash_attn_func(
    q, k, v, sparse_plan=plan, is_online_first=True, is_online_last=True,
)
```

The native HIP loop traverses BCSR rows at 384 query × 64 KV granularity. Empty
tiles are omitted, full tiles bypass token masks, and partial tiles read exact
bitmaps cached by the planner from the four packed-mask clauses. Plans are shared across heads and batch entries;
different masks per batch entry are not supported. Positions travel with the
metadata through permutation. A fully masked row returns zero. Empty ring steps
initialize, preserve, or finalize state as appropriate. Treat plans as immutable:
the low-level extension checks tensor structure, but trusts block index values.

`usp_attention.online_flash_amd.distributed` provides the explicit forward-only
`build_forward_plan` and `sparse_usp_forward` entry points. They use existing
Ulysses process groups/all-to-all and PTRR scheduling, with RCCL P2P KV rotation.
The caller pads to a multiple of `384 * ring * ulysses`, isolates padding with a
separate document id, and applies `plan.shard` before attention. Outputs stay in
the same permuted local order; gather/invert only when the consumer needs it.
This entry point rejects grad-enabled inputs requiring gradients. It does not
change the existing NVIDIA `usp_attention` dispatcher or implement backward.

The planner accepts rectangular blocks and retains fixed-width BCSR indices,
tile classes, and offsets into packed partial-tile masks. Each partial tile adds
3 KiB (384 rows × two 32-bit words), shared across heads and reused across layers.
Full and empty tiles allocate no token bitmap. For adversarial layouts where
every tile is partial, bitmap storage can still grow quadratically; the appendix
benchmark records the maximum per-rank bitmap bytes. Planning,
permutation, allocation, V transpose and per-call state traffic still offer room
for optimization. The existing NVIDIA lockfile is **not** an AMD install recipe;
use a ROCm PyTorch environment and build this extension locally.

See `benchmarks/AMD_SPARSE_BITMAP.md` for optimized measurements, and
`benchmarks/AMD_SPARSE_FORWARD.md` for the original appendix experiments and
reproduction commands. The benchmark uses only the sparse path, including its
Ulysses+DP baseline; it does not compare against dense-varlen decomposition.

### Fused KV-gather and folded MQA

For an eight-GPU 2×4 mesh with 28 query and 4 KV heads, the distributed wrapper
can gather each rank's full KV head and fold its seven query heads into the query
axis. This uses one CP all-to-all for Q/K/V routing, one sparse HIP attention
call and the normal Ulysses output all-to-all:

```python
from usp_attention.online_flash_amd.distributed import build_forward_plan, sparse_usp_forward

# aux is padded to 384 * groups.cp_size and isolates padding in its own document.
plan = build_forward_plan(aux, groups, query_head_factor=7, kv_exchange="gather")
q_local, k_local, v_local = (plan.shard(x) for x in (q, k, v))
with torch.no_grad():
    out_local = sparse_usp_forward(q_local, k_local, v_local, plan, round_mode="rtna")
```

`query_head_factor` must equal the query-head count after Ulysses and requires
one KV head per rank after Ulysses. Its default is 1, using the ordinary head
mapping. `kv_exchange="ring"` remains the default for bounded KV storage. The
gather path uses more KV memory, and folding expands metadata/BCSR storage;
reuse plans across layers. Sparse query workgroups are scheduled longest-first
while online state stays indexed by logical query/head IDs.

See [fused-exchange measurements](../../../benchmarks/AMD_SPARSE_GATHER.md) for
latency, memory, correctness gates and reproduction commands.

## Full decoder forward benchmark

[AMD_MODEL_FORWARD.md](../../../benchmarks/AMD_MODEL_FORWARD.md) describes the
forward-only model benchmark that reuses the article's 28-layer multimodal
MoT decoder, including FSDP weight gathers, projections and MLPs. It compares
naive ring, ring + PTRR and USP without requiring an AMD attention backward.
