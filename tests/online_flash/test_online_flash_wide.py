# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Online FA4 (SM90) above head_dim 128 matches upstream FA4.

Above 128 the kernel keeps its fp32 state out of shared memory (it moves
directly between global memory and registers) and defers the O rescale to just
before each PV GEMM. Chunked KV threads that state through several calls. The
reference runs one upstream ``flash_attn_func`` call per chunk, with the same
mask, and merges them by log-sum-exp; with one chunk that is a single call.

- head_dim 200 is not a multiple of 16, so its padded tile takes the
  column-predicated state copy.
- Q length 1000 leaves a partial last tile (row-predicated copy).
- Causal and sliding-window masks use upstream's tile sizes for those masks.
- Block sparsity at (192, 128) is the one block-sparse configuration whose
  tiles still fit 128-token sparse blocks above head_dim 128.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cutlass")

from flash_attn.cute.interface import flash_attn_func  # noqa: E402

from usp_attention.online_flash.interface import (  # noqa: E402
    online_flash_attn_chunks,
    online_flash_attn_func,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
        reason="the online kernel targets SM90",
    ),
]

# Same tolerances as test_online_flash.py (bf16 output, fp32 LSE); chunked
# sliding-window runs merge more partials, so their output gets 2x.
OUT_ATOL, OUT_RTOL = 1.5e-3, 1e-2
LSE_ATOL, LSE_RTOL = 2e-5, 1e-4
HEAD_DIMS = [(160, 160), (192, 192), (192, 128), (200, 200), (256, 256)]
MASKS = {
    "none": {},
    "causal": {"causal": True},
    "window": {"window_size": (300, 0)},
}


def _qkv(seqlen_q: int, head_dim: int, head_dim_v: int, seqlen_k: int = 8192):
    gen = torch.Generator(device="cuda").manual_seed(0)
    shapes = [
        (1, seqlen_q, 4, head_dim),
        (1, seqlen_k, 4, head_dim),
        (1, seqlen_k, 4, head_dim_v),
    ]
    return [
        torch.randn(s, device="cuda", dtype=torch.bfloat16, generator=gen)
        for s in shapes
    ]


def _chunked_fa4(q, k, v, num_chunks: int, **mask):
    """One upstream FA4 call per KV chunk, merged by log-sum-exp."""
    outs, lses = [], []
    for k_chunk, v_chunk in zip(k.chunk(num_chunks, dim=1), v.chunk(num_chunks, dim=1)):
        out, lse = flash_attn_func(q, k_chunk, v_chunk, return_lse=True, **mask)
        outs.append(out.float())
        lses.append(lse.float())  # (B, H, S)
    stacked = torch.stack(lses)
    lse = torch.logsumexp(stacked, dim=0)
    weights = torch.exp(stacked - lse).nan_to_num(0.0)
    out = sum(o * w.transpose(1, 2)[..., None] for o, w in zip(outs, weights))
    return out, lse


def _check(out, lse, ref_out, ref_lse, out_scale: float = 1.0) -> None:
    assert lse is not None
    torch.testing.assert_close(
        out.float(), ref_out, atol=OUT_ATOL * out_scale, rtol=OUT_RTOL
    )
    torch.testing.assert_close(
        lse.float().reshape(ref_lse.shape), ref_lse, atol=LSE_ATOL, rtol=LSE_RTOL
    )


@pytest.mark.parametrize(("head_dim", "head_dim_v"), HEAD_DIMS)
@pytest.mark.parametrize("mask", list(MASKS))
@pytest.mark.parametrize("num_chunks", [1, 4])
def test_wide_heads_match_fa4(
    head_dim: int, head_dim_v: int, mask: str, num_chunks: int
) -> None:
    q, k, v = _qkv(1024, head_dim, head_dim_v)
    kw = MASKS[mask]
    out, lse = online_flash_attn_chunks(
        q, k.chunk(num_chunks, dim=1), v.chunk(num_chunks, dim=1), return_lse=True, **kw
    )
    ref_out, ref_lse = _chunked_fa4(q, k, v, num_chunks, **kw)
    _check(
        out, lse, ref_out, ref_lse, 2.0 if mask == "window" and num_chunks > 1 else 1.0
    )


@pytest.mark.parametrize(
    ("head_dim", "head_dim_v"), [(192, 192), (200, 200), (256, 256)]
)
@pytest.mark.parametrize("num_chunks", [1, 4])
def test_wide_heads_partial_last_tile(
    head_dim: int, head_dim_v: int, num_chunks: int
) -> None:
    q, k, v = _qkv(1000, head_dim, head_dim_v)
    out, lse = online_flash_attn_chunks(
        q, k.chunk(num_chunks, dim=1), v.chunk(num_chunks, dim=1), return_lse=True
    )
    _check(out, lse, *_chunked_fa4(q, k, v, num_chunks))


def test_block_sparse_192_128_matches_fa4() -> None:
    from usp_attention.fa4.flash4_attn import build_fa4_sparse_mask, flash4_attn
    from usp_attention.layouts import build_layout

    docs, splits, modes = build_layout("mixed", 8192, seed=1)
    mask = build_fa4_sparse_mask(docs, splits, modes, device="cuda")
    q, k, v = _qkv(8192, 192, 128)
    out, lse, _, _ = online_flash_attn_func(
        q,
        k,
        v,
        is_online_first=True,
        is_online_last=True,
        return_lse=True,
        block_sparse_tensors=mask.block_sparse,
        mask_mod=mask.mask_mod,
        aux_tensors=mask.aux_tensors,
    )
    ref_out, ref_lse = flash4_attn(q, k, v, mask, return_lse=True)
    _check(out, lse, ref_out.float(), ref_lse.float())
