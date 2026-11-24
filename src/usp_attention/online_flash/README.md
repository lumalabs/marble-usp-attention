# Online Flash Attention 4 (SM90)

This directory packages our **online** forward variant of the Flash Attention 4
kernel for Hopper (SM90).

"Online" means that the kernel can be called on a series of K/V chunks e.g. from RingAttention. The nice thing about this kernel is that it avoids us having to do a separate RingCombine step to combine partial attention outputs. This allows us to have a more performant RingAttention implementation (especially for long-context block-sparse cases), whilst also guaranteeing that outputs are closer to regular flash attention.

It subclasses
`flash_attn.cute.flash_fwd_sm90.FlashAttentionForwardSm90`, reusing the bulk
of FA4's Hopper forward pipeline. Only the pieces that needed to change for
online accumulation (softmax state load, normalization on the last call only,
shared-storage layout, output state load / store path, interface signature)
are overridden.

## Requirements

- **SM 9.x (Hopper)** GPU.
- `flash-attn-4==4.0.0b34` and `quack-kernels` 0.6.x (pinned in this
  project's `pyproject.toml`; 4.0.0b21 also works). The kernel subclasses FA4's
  `flash_attn.cute.flash_fwd_sm90.FlashAttentionForwardSm90` and reuses its
  pack-GQA and block-sparse load helpers, so other FA4 versions may not work.
- `bfloat16` or `float16` inputs in BSHD layout: `(batch, seqlen, num_head, head_dim)`.
- `head_dim` ≤ 256 and a multiple of 8. Up to 128 the fp32 output state is
  staged through shared memory with TMA; above 128 an fp32 tile would not fit
  next to Q / K / V in Hopper's 227 KiB, so the state moves directly between
  global memory and the accumulator registers (and the O rescale is deferred to
  just before each PV GEMM, as upstream does for wide heads).

## Usage

### Chunked KV (one call per chunk)

```python
import torch
from usp_attention.online_flash import online_flash_attn_func

B, Sq, Sk, H, D = 1, 1024, 8192, 4, 128
q = torch.randn(B, Sq, H, D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(B, Sk, H, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(B, Sk, H, D, dtype=torch.bfloat16, device="cuda")

k_chunks = list(k.chunk(4, dim=1))
v_chunks = list(v.chunk(4, dim=1))

# First call: allocates max_ / sum_ / out internally.
out, _, max_, sum_ = online_flash_attn_func(
    q, k_chunks[0], v_chunks[0],
    is_online_first=True,
)

# Middle calls: thread the state back in.
for k_c, v_c in zip(k_chunks[1:-1], v_chunks[1:-1]):
    out, _, max_, sum_ = online_flash_attn_func(
        q, k_c, v_c, out=out, max_=max_, sum_=sum_,
    )

# Last call: finalize output + LSE.
out, lse, max_, sum_ = online_flash_attn_func(
    q, k_chunks[-1], v_chunks[-1],
    out=out, max_=max_, sum_=sum_,
    is_online_last=True,
    return_lse=True,
)
```

### Convenience wrapper

If you already have the KV split into a list, `online_flash_attn_chunks`
handles the first/last bookkeeping:

```python
from usp_attention.online_flash import online_flash_attn_chunks

out, lse = online_flash_attn_chunks(
    q, k_chunks, v_chunks,
    softmax_scale=D ** -0.5,
    return_lse=True,
)
```

### Comparing against regular FA4

A thin re-export of the upstream non-online path is provided so correctness
checks don't have to reach into `flash_attn.cute` directly:

```python
from usp_attention.online_flash import offline_flash_attn_func

ref_out, ref_lse = offline_flash_attn_func(q, k, v, return_lse=True)
```

## Tests

```bash
cd projects/usp_attention
uv run pytest tests/online_flash -q
```

`tests/online_flash/test_online_flash.py` sweeps `num_chunks ∈ {2, 4, 8}` ×
`seqlen_k ∈ {2048, 8192, 32768, 131072}` and asserts that the online output
and LSE match a single full-sequence FA4 call; `test_online_flash_wide.py`
covers head dims above 128. See the top of `test_online_flash.py` for
tolerance / dimension overrides.

## Numerics

Measured on H100 bf16 across the full 12-case sweep (against a single
full-sequence FA4 call). The default tolerances sit roughly 2× above the
empirical worst case, separately for `out` and `lse`:

| Tensor | `atol` | `rtol` | Worst required `atol` (at chosen `rtol`) |
|--------|-------:|-------:|-----------------------------------------:|
| `out`  | 1.5e-3 |   1e-2 | 7.67e-4 |
| `lse`  | 2.0e-5 |   1e-4 | 1.14e-5 |

**Only one case (`num_chunks=8, seqlen_k=2048`) sets the output floor.** The
other 11 cases pass at roughly half the output tolerance — at `rtol=1e-2` they
only need `atol ≲ 4.4e-4`. The outlier is the shortest-seq × most-chunks
configuration, which has the fewest keys per chunk and therefore the most
per-chunk rounding relative to the running row-sum. Output absolute error
*decreases* as `seqlen_k` grows because the softmax row-sum grows and dilutes
per-chunk noise.

LSE is much tighter (~100× below output) because it's only written on the
last call after a single `finalize`, so it skips the re-scaling rounds.

Tolerance overrides (see `tests/online_flash/test_online_flash.py`):
`ONLINE_FA4_OUT_ATOL`, `ONLINE_FA4_OUT_RTOL`, `ONLINE_FA4_LSE_ATOL`,
`ONLINE_FA4_LSE_RTOL`.

## What's different from `flash_fwd_sm90.py`

`online_flash_fwd_sm90.py` subclasses `FlashAttentionForwardSm90` and
overrides the minimum needed for chunked-KV accumulation. The substantive
deltas:

### 1. Softmax state gets loaded from HBM, not initialized from scratch

Regular FA4 creates fresh `row_max` / `row_sum` registers per tile. The online
kernel introduces `Softmax2TheSequel`, which can **load** the running state
from the `mMax` / `mSum` tensors the caller passes in:

```python
# online_flash_fwd_sm90.py — new class
@dataclass
class Softmax2TheSequel(Softmax):
    @cute.jit
    def load(self, mMax, mSum, seqlen, tiled_mma,
             tile_m, tile_hdimv, batch_idx, head_idx, tidx, m_block):
        gMax = cute.local_tile(mMax_cur, (tile_m,), (m_block,))
        gSum = cute.local_tile(mSum_cur, (tile_m,), (m_block,))
        # …partition into this warp's MMA slice and broadcast into registers…
        for m in cutlass.range(cute.size(taccOgMax.shape[1]), unroll_full=True):
            if t0accOcO[m, 0][0] < seqlen.seqlen_q - m_block * tile_m - taccOcO[0][0]:
                self.row_max[m] = taccOgMax[m, 0]
                self.row_sum[m] = taccOgSum[m, 0] if is_col0 else 0.0
            else:
                self.row_max[m] = -Float32.inf
                self.row_sum[m] = 0.0
```

`flash_fwd_sm90.py`'s `Softmax` has no analogous `load` — the running state
only exists for the lifetime of a single kernel invocation.

### 2. Normalization runs only on `is_last_call`

In regular FA4, `softmax.finalize(...)` always runs: it computes `1 / row_sum`
and the LSE before the epilogue, and the output is normalized. The online
kernel calls the same inherited `Softmax.finalize` (`Softmax2TheSequel` does
not override it) only on the last chunk. Earlier calls skip the
reciprocal / `log2`, write the warp-reduced `row_max` / `row_sum` to `mMax` /
`mSum` in the epilogue, and store the unnormalized output.

```python
# online_flash_fwd_sm90.py — only the last call normalizes
if const_expr(is_last_call):  # Only normalise if this is the last call
    row_scale = softmax.finalize(sink_val=sink_val)
    softmax.rescale_O(acc_O, row_scale)
# epilogue(): the last call writes the LSE, earlier calls write row_max /
# row_sum instead and store the unnormalized acc_O for the next call.
```

The running output stays fp32 between calls. The last call reads it from the
fp32 buffer (`mO_state`) and writes the normalized output straight to a new
tensor in the input dtype, so there is no separate conversion afterwards.

### 3. Shared storage has an fp32 `sO` tile and an O-reload barrier

Non-first online calls need to **read** the partial output back from HBM,
re-scale it by the new `row_max`, and keep accumulating. Up to head_dim 128
the running fp32 output goes through one extra SMEM region, `sO`, used both
to reload it (TMA, behind the new `mbar_ptr_O_load` barrier pair; the
producer's `load` names this view `sO_load`) and to stage the epilogue's
store (the `sO_store` view of the same region):

```diff
 @cute.struct
 class SharedStorageQKV:
     mbar_ptr_Q: mbar_ptr_Q_struct
     mbar_ptr_K: mbar_ptr_K_struct
     mbar_ptr_V: mbar_ptr_V_struct
+    mbar_ptr_O_load: mbar_ptr_O_load_struct
     sV: sV_struct
     sQ: sQ_struct
     sK: sK_struct
     sP: sP_struct
+    sO: sO_struct  # fp32 running output (reload + epilogue store)
```

Above head_dim 128 an fp32 tile does not fit next to Q / K / V, so `sO` is
empty and `copy_O_state` moves the state directly between global memory and
the accumulator registers. Upstream `flash_fwd_sm90.py` stages its bf16
output through Q's SMEM and has no fp32 output region.

### 4. Kernel entry-point signature carries the online state

`OnlineFlashAttentionForwardSm90.__call__` adds four parameters on top of
`FlashAttentionForwardSm90.__call__`:

```python
def __call__(
    self,
    mQ, mK, mV, mO, mLSE,
    mMax: cute.Tensor,                 # running row-max, HBM
    mSum: cute.Tensor,                 # running row-sum, HBM
    is_first_call: cutlass.Constexpr,  # skip load(), init fresh
    is_last_call: cutlass.Constexpr,   # call finalize(), write lse, scale out
    softmax_scale, …
)
```

The `is_first_call` / `is_last_call` flags are `Constexpr` so the compiler
specializes each variant — first-only, middle, last-only, and
first-and-last (for `num_chunks == 1`) compile to distinct cubins with the
dead branches pruned.

### 5. This module's Python interface is SM90-only

Upstream `_flash_attn_fwd` dispatches to SM80 / SM90 / SM100 / SM120 kernels
and handles GQA pack, paged KV, varlen, block sparsity, split-KV, score_mod,
and learnable sinks. The online kernel only implements the SM90 path, so
this directory's `interface.py` is a much shorter function — it reuses
`to_cute_tensor`, `torch2cute_dtype_map`, `_validate_tensor`,
`maybe_contiguous`, `_get_device_arch`, and the block-sparse
normalization helpers (`normalize_block_sparse_config`,
`to_cute_block_sparse_tensors`) from the upstream package, and only spells
out the SM90 compile/invocation branch locally.

## Feature support

The kernel (`OnlineFlashAttentionForwardSm90`) inherits most of its feature
matrix from `FlashAttentionForwardSm90`. What's hard-asserted off in the
online subclass:

- **GQA pack** — `__init__` asserts `not self.pack_gqa`.
- **Paged KV (non-TMA)** — `__init__` asserts `not paged_kv_non_tma`.

Exposed through this module's Python wrapper:

- Dense SM90 BSHD attention.
- Causal masking, sliding-window (`window_size`).
- **Block sparsity** via `block_sparse_tensors=` on both `online_flash_attn_func`
  and `online_flash_attn_chunks`. Pass the same `BlockSparseTensorsTorch` the
  upstream non-online interface would accept.
- `mask_mod` with its `aux_tensors` on `online_flash_attn_func` (the ring's
  block-sparse steps pass their cross Q/KV mask this way).

What the kernel still supports but **this module's Python wrapper does not
expose yet** (`interface.py` passes `None` for all of these):

- `score_mod`, learnable sinks
- Variable-length sequences (`cu_seqlens_q`, `cu_seqlens_k`, `seqused_*`)
- Split-KV (orthogonal to online chunking — you usually want one or the other,
  not both)

If you need any of these, the path of least resistance is to extend
`online_flash_attn_func` in `interface.py` to plumb the argument through to
the kernel's `__call__`; no kernel change required.

What simply does not exist yet:

- **Backward pass** — there is no online backward kernel.

### Block sparsity usage

```python
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
from usp_attention.online_flash import online_flash_attn_chunks

# Build a block-sparse mask describing the full Q x K (packed) space once —
# the same descriptor is reused across every online chunk.
sparse = BlockSparseTensorsTorch(...)

out, lse = online_flash_attn_chunks(
    q, k_chunks, v_chunks,
    softmax_scale=scale,
    block_sparse_tensors=sparse,
    return_lse=True,
)
```
