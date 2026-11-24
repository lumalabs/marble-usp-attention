# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Sliding-window masks: key counts, block classification and plan fallbacks."""

from __future__ import annotations

import pytest
import torch

from usp_attention.fa4.flash4_block_sparse import _classify_self_attention_blocks_torch
from usp_attention.fa4.interval_mask import build_interval_plan
from usp_attention.layouts import build_layout
from usp_attention.reference import dense_mask_rows, visible_key_counts
from usp_attention.ring.block_sparse import build_block_sparse_from_cross_qkv_aux
from usp_attention.ring.packed_layout import build_packed_layout
from usp_attention.ring.ring_planning import build_cross_qkv_step_aux_tensors
from usp_attention.sliding_window import (
    BLOCK_EMPTY,
    BLOCK_FULL,
    segment_end_positions,
    windowed_cross_blocks,
    windowed_self_attention_blocks,
)

LAYOUTS = ["causal", "docs", "mixed", "vlm", "video", "vlm_group"]
WINDOWS = [1, 100, 256, 1024, 5000]
N = 8192


def _mask(name: str, window: int | None):
    docs, splits, modes = build_layout(name, N, ring_size=4, seed=3)
    layout = build_packed_layout(docs, splits, modes)
    return (
        (docs, splits, modes),
        layout,
        dense_mask_rows(layout, 0, N, sliding_window=window),
    )


@pytest.mark.parametrize("name", LAYOUTS)
@pytest.mark.parametrize("window", WINDOWS)
def test_window_mask_semantics(name: str, window: int) -> None:
    spec, layout, mask = _mask(name, window)
    full = dense_mask_rows(layout, 0, N)
    # A window only removes pairs, and only causal pairs at distance >= window.
    assert not (mask & ~full).any()
    q = torch.arange(N)[:, None]
    k = torch.arange(N)[None, :]
    removed = full & ~mask
    assert (q - k)[removed].min().item() >= window if removed.any() else True
    fn = layout.full_and_noise_seq_id
    same_seg = (fn[:, None] == fn[None, :]) & (fn[:, None] >= 0)
    assert not (removed & same_seg).any()
    # The closed-form counts match the dense mask.
    counts = visible_key_counts(*spec, sliding_window=window)
    assert torch.equal(counts, mask.sum(dim=1))


def _block_view(mask: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor, b: int):
    sub = mask[rows][:, cols]
    nq, nk = sub.shape[0] // b, sub.shape[1] // b
    blocks = sub.reshape(nq, b, nk, b)
    return blocks.any(dim=(1, 3)), blocks.all(dim=(1, 3))


def _check_sound(block_type, any_visible, all_visible):
    # Empty blocks hold no visible pair; full blocks hold only visible pairs.
    assert not (any_visible & (block_type == BLOCK_EMPTY)).any()
    assert all_visible[block_type == BLOCK_FULL].all()


@pytest.mark.parametrize("name", LAYOUTS)
@pytest.mark.parametrize("window", WINDOWS)
def test_self_blocks(name: str, window: int) -> None:
    _, layout, mask = _mask(name, window)
    result = _classify_self_attention_blocks_torch(
        layout.full_and_noise_seq_id, layout.noise_seq_id, layout.document_id
    )
    windowed = windowed_self_attention_blocks(
        result, layout.full_and_noise_seq_id, window
    )
    idx = torch.arange(N)
    any_visible, all_visible = _block_view(mask, idx, idx, 128)
    _check_sound(windowed.block_type, any_visible, all_visible)
    # Causal layouts: the window empties every block it can.
    if name in ("causal", "docs"):
        assert torch.equal(windowed.block_type != BLOCK_EMPTY, any_visible)
    # BCSR counts agree with the block types.
    assert windowed.kv_num_blocks.sum() == (windowed.block_type == 1).sum()
    assert windowed.full_kv_num_blocks.sum() == (windowed.block_type == 2).sum()


@pytest.mark.parametrize("name", LAYOUTS)
@pytest.mark.parametrize("window", [100, 1024])
def test_cross_blocks_permuted(name: str, window: int) -> None:
    """Ring steps over a random 128-block permutation stay sound."""
    _, layout, mask = _mask(name, window)
    gen = torch.Generator().manual_seed(0)
    perm = (
        torch.randperm(N // 128, generator=gen)[:, None] * 128 + torch.arange(128)
    ).flatten()
    aux = [
        layout.full_and_noise_seq_id[perm],
        layout.noise_seq_id[perm],
        layout.document_id[perm],
    ]
    pos = perm.clone()
    seg_end = segment_end_positions(layout.full_and_noise_seq_id)[perm]
    local = N // 4
    for q_start in range(0, N, local):
        for kv_start in range(0, N, local):
            step = build_cross_qkv_step_aux_tensors(
                global_aux_tensors=aux,
                q_start=q_start,
                q_len=local,
                kv_start=kv_start,
                kv_len=local,
                global_position_ids=pos,
            )
            result = build_block_sparse_from_cross_qkv_aux(step, block_size=128)
            windowed = windowed_cross_blocks(
                result, step, seg_end[kv_start : kv_start + local], window
            )
            any_visible, all_visible = _block_view(
                mask,
                perm[q_start : q_start + local],
                perm[kv_start : kv_start + local],
                128,
            )
            _check_sound(windowed.block_type, any_visible, all_visible)


def test_interval_plan_refuses_spans_longer_than_window() -> None:
    docs, splits, modes = build_layout("vlm", N, seed=3)
    longest = max(s for s, m in zip(splits, modes) if m == "full")
    assert build_interval_plan(
        docs, splits, modes, device="cpu", sliding_window=longest
    )
    assert (
        build_interval_plan(
            docs, splits, modes, device="cpu", sliding_window=longest - 1
        )
        is None
    )


def test_segment_end_positions() -> None:
    fn = torch.tensor([-1, 1, 1, -1, 4, 4, 4, 1, -1], dtype=torch.float32)
    assert segment_end_positions(fn).tolist() == [-1, 7, 7, -1, 6, 6, 6, 7, -1]


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="needs an H100 for FA4 block sparsity",
)
@pytest.mark.parametrize("window", [None, 300])
def test_flash4_attn_backward_matches_reference(window: int | None) -> None:
    """The FA4 custom op runs backward with and without a window, and matches fp32."""
    from usp_attention.fa4.flash4_attn import build_fa4_sparse_mask, flash4_attn
    from usp_attention.reference import reference_attention

    docs, splits, modes = build_layout("vlm", 8192, seed=1)
    gen = torch.Generator(device="cuda").manual_seed(0)
    q, k, v, dout = (
        torch.randn(1, 8192, h, 128, device="cuda", dtype=torch.bfloat16, generator=gen)
        for h in (8, 4, 4, 8)
    )
    mask = build_fa4_sparse_mask(
        docs, splits, modes, device="cuda", sliding_window=window
    )
    grads = []
    for fn in (
        lambda a, b, c: flash4_attn(a, b, c, mask),
        lambda a, b, c: reference_attention(
            a,
            b,
            c,
            document_lens=docs,
            split_lens=splits,
            attn_modes=modes,
            sliding_window=window,
        )[0],
    ):
        leaves = [t.clone().requires_grad_(True) for t in (q, k, v)]
        out = fn(*leaves)
        out.backward(dout.to(out.dtype))
        grads.append([out.float(), *(t.grad.float() for t in leaves)])
    for got, want in zip(*grads):
        rel = (got - want).abs().mean() / want.abs().mean()
        assert rel < 1e-2


def _single_rank_groups():
    from usp_attention.distributed import USPGroups

    return USPGroups(
        ring_size=1,
        ulysses_size=1,
        ring_rank=0,
        ulysses_rank=0,
        cp_rank=0,
        ring_group=None,
        ulysses_group=None,
        cp_group=None,
    )


def test_window_plan_keeps_base_token_order_without_ring() -> None:
    """A windowed layer must shard like the global layers it is interleaved with.

    Vision-group layouts are reordered even without a ring; the windowed plan
    used to drop that order and mask in sequence order.
    """
    from usp_attention import USPConfig, build_usp_plan

    docs, splits, modes = build_layout("vlm_group", N, seed=3)
    kw = dict(
        document_lens=docs,
        split_lens=splits,
        attn_modes=modes,
        groups=_single_rank_groups(),
        config=USPConfig(),
        device="cpu",
    )
    base = build_usp_plan(**kw)
    assert base.token_permutation is not None
    win = build_usp_plan(**kw, sliding_window=1024)
    assert win.token_permutation is not None
    assert torch.equal(win.token_permutation, base.token_permutation)
    # A given token order reaches the windowed plan too.
    order = torch.randperm(N, generator=torch.Generator().manual_seed(0))
    given = build_usp_plan(**kw, sliding_window=1024, token_permutation=order)
    assert torch.equal(given.token_permutation.cpu(), order)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="needs an H100 for FA4 block sparsity",
)
@pytest.mark.parametrize("order", ["vision_groups", "given_permutation"])
def test_window_layer_sharded_with_base_plan_matches_natural_order(order: str) -> None:
    """Shard once with the base plan, run a windowed layer, gather: same result
    (forward and gradients) as the windowed mask in natural token order. The
    base is reordered by vision groups, or by a token_permutation the caller
    passes to both plans."""
    from usp_attention import USPConfig, build_usp_plan, usp_attention

    layout = "vlm_group" if order == "vision_groups" else "mixed"
    docs, splits, modes = build_layout(layout, N, seed=3)
    kw = dict(
        document_lens=docs,
        split_lens=splits,
        attn_modes=modes,
        groups=_single_rank_groups(),
        device=torch.device("cuda"),
    )
    given = (
        {}
        if order == "vision_groups"
        else {"token_permutation": torch.randperm(N, device="cuda")}
    )
    base = build_usp_plan(**kw, config=USPConfig(), **given)
    win = build_usp_plan(**kw, config=USPConfig(), sliding_window=1024, **given)
    assert base.token_permutation is not None
    natural = build_usp_plan(
        **kw, config=USPConfig(reorder_vision_groups=False), sliding_window=1024
    )
    assert natural.token_permutation is None
    gen = torch.Generator(device="cuda").manual_seed(0)
    q, k, v, dout = (
        torch.randn(1, N, 8, 128, device="cuda", dtype=torch.bfloat16, generator=gen)
        for _ in range(4)
    )

    def run(shard_plan, attn_plan):
        leaves = [shard_plan.shard(t).clone().requires_grad_(True) for t in (q, k, v)]
        out = usp_attention(*leaves, attn_plan)
        out.backward(shard_plan.shard(dout))
        return [
            shard_plan.gather(t).float()
            for t in (out.detach(), *(x.grad for x in leaves))
        ]

    for got, want in zip(run(base, win), run(natural, natural)):
        rel = (got - want).abs().mean() / want.abs().mean()
        assert rel < 1e-2, rel
