# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Planner fixes: padding, position_ids, deterministic, wide heads, plan-time cost."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch

import usp_attention.usp as usp_module
from usp_attention.distributed import USPGroups
from usp_attention.fa4.flash4_attn import HAS_FLASH4
from usp_attention.layouts import build_layout
from usp_attention.reference import dense_mask_rows
from usp_attention.ring.packed_layout import build_packed_layout
from usp_attention.ring.ring_planning import build_ring_plan
from usp_attention.usp import (
    USPConfig,
    USPPlan,
    build_sliding_window_plan,
    build_usp_plan,
    pad_packed_layout,
    usp_attention,
)

needs_fa4 = pytest.mark.skipif(
    not HAS_FLASH4, reason="block-sparse planning needs flash_attn.cute"
)
N = 8192
# Takes the dense ring at ring 2 with forced chunks (test_dense_ring_config.py).
DENSE_SPEC = (
    [2048, 2048],
    [512, 1536, 512, 1536],
    ["causal", "full", "causal", "full"],
)
DENSE = dict(dense_ring=True, dense_ring_chunks=2)


def _groups(ring: int = 1) -> USPGroups:
    return USPGroups(
        ring_size=ring,
        ulysses_size=1,
        ring_rank=0,
        ulysses_rank=0,
        cp_rank=0,
        ring_group=None,
        ulysses_group=None,
        cp_group=None,
    )


def _plan(spec, *, ring: int = 1, config: USPConfig | None = None, **kwargs):
    docs, splits, modes = spec
    return build_usp_plan(
        document_lens=docs,
        split_lens=splits,
        attn_modes=modes,
        groups=_groups(ring),
        config=config or USPConfig(),
        device=torch.device("cpu"),
        **kwargs,
    )


def _sentinel_plan(**path) -> USPPlan:
    """A plan that only selects an execution path (its metadata is never read)."""
    fields = dict(
        config=USPConfig(),
        groups=_groups(),
        seq_len=N,
        token_permutation=None,
        restore_permutation=None,
        step_meta=None,
        rank_plan=None,
        fa4_mask=None,
    )
    fields.update(path)
    return USPPlan(**fields)


def _hopper(monkeypatch: pytest.MonkeyPatch, capability=(9, 0)) -> None:
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: capability
    )


# -- pad_packed_layout ---------------------------------------------------------


def test_padding_is_an_isolated_causal_document() -> None:
    docs, splits, modes = pad_packed_layout(
        [1000, 900], [300, 700, 900], ["causal", "full", "causal"], 1024
    )
    assert (docs[-1], splits[-1], modes[-1]) == (148, 148, "causal")
    layout = build_packed_layout(docs, splits, modes)
    mask = dense_mask_rows(layout, 0, sum(docs))
    real = torch.arange(sum(docs)) < 1900
    # No real query sees a padding key, and no padding query sees a real key.
    assert not mask[real][:, ~real].any()
    assert not mask[~real][:, real].any()
    # The padding queries still attend (causally) among themselves.
    assert mask[~real][:, ~real].sum() == 148 * 149 // 2


def test_padded_causal_documents_keep_the_varlen_path() -> None:
    spec = pad_packed_layout([3000, 4000], [3000, 4000], ["causal", "causal"], 1024)
    plan = _plan(spec)
    assert plan.native_mask == "causal" and plan.fa4_mask is None
    assert plan.native_cu_seqlens.tolist() == [0, 3000, 7000, 7168]


# -- position_ids --------------------------------------------------------------


def test_default_positions_are_the_same_as_none() -> None:
    spec = build_layout("docs", N, seed=0)
    plan = _plan(spec, position_ids=torch.arange(N))
    assert plan.native_mask == "causal"


def test_positions_must_match_the_sequence() -> None:
    with pytest.raises(ValueError, match="shape"):
        _plan(build_layout("docs", N, seed=0), position_ids=torch.arange(N - 1))


@pytest.mark.parametrize(
    ("layout", "config"),
    [
        ("docs", USPConfig()),  # varlen
        ("causal", USPConfig()),  # one native call
        ("mixed", USPConfig(interval_masks=True)),  # interval
    ],
)
def test_custom_positions_rejected_where_kernels_mask_by_index(
    layout: str, config: USPConfig
) -> None:
    positions = torch.arange(N).flip(0)
    with pytest.raises(ValueError, match="position_ids"):
        _plan(build_layout(layout, N, seed=0), config=config, position_ids=positions)


def test_custom_positions_rejected_on_the_dense_ring() -> None:
    with pytest.raises(ValueError, match="dense_ring path"):
        _plan(
            DENSE_SPEC,
            ring=2,
            config=USPConfig(**DENSE),
            position_ids=torch.arange(4096) // 2,
        )


@needs_fa4
def test_custom_positions_kept_on_the_position_aware_path() -> None:
    positions = torch.arange(N) // 2
    plan = _plan(
        build_layout("mixed", N, seed=0),
        position_ids=positions,
        token_permutation=torch.arange(N),
    )
    assert plan.step_meta is not None and plan.step_meta.native is None


def test_sliding_window_plan_rejects_custom_positions_in_natural_order() -> None:
    spec = build_layout("docs", N, seed=0)
    base = _plan(spec)
    with pytest.raises(ValueError, match="position_ids"):
        build_sliding_window_plan(
            base,
            sliding_window=512,
            document_lens=spec[0],
            split_lens=spec[1],
            attn_modes=spec[2],
            device=torch.device("cpu"),
            position_ids=torch.arange(N) // 2,
        )


# -- deterministic ---------------------------------------------------------------


def test_deterministic_backward_rejected_on_dense_paths() -> None:
    plan = _plan(DENSE_SPEC, ring=2, config=USPConfig(**DENSE, deterministic=True))
    assert plan.dense_ring is not None
    q, k, v = (
        torch.randn(1, plan.local_len, 2, 64, requires_grad=True) for _ in range(3)
    )
    with pytest.raises(ValueError, match="deterministic"):
        usp_attention(q, k, v, plan)


@pytest.mark.parametrize(
    "path",
    [
        {"generation_ring": object()},
        {"dense_gather": object()},
        {"dense_ring": object()},
    ],
)
def test_deterministic_checked_only_for_a_backward(path: dict) -> None:
    plan = _sentinel_plan(config=USPConfig(deterministic=True), **path)
    kw = dict(device=torch.device("cpu"), head_dim=64, head_dim_v=64)
    usp_module._check_plan_supported(plan, backward=False, **kw)
    with pytest.raises(ValueError, match="dense_ring=False"):
        usp_module._check_plan_supported(plan, backward=True, **kw)
    plan = dataclasses.replace(plan, config=USPConfig())
    usp_module._check_plan_supported(plan, backward=True, **kw)


@pytest.mark.parametrize(
    "path",
    [
        {"halo": object()},
        {"interval_plan": object()},
        {"native_mask": "causal"},
        {"fa4_mask": object()},  # flash4_attn takes deterministic=
    ],
)
def test_deterministic_accepted_where_kernels_take_it(path: dict) -> None:
    plan = _sentinel_plan(config=USPConfig(deterministic=True), **path)
    usp_module._check_plan_supported(
        plan, device=torch.device("cpu"), head_dim=64, head_dim_v=64, backward=True
    )


# -- FA4 block sparsity above head_dim 128 on H100 -------------------------------


@pytest.mark.parametrize(
    ("dims", "backward", "expected"),
    [
        ((128, 128), True, False),
        ((96, 96), True, False),
        ((256, 256), False, True),
        ((256, 256), True, True),
        ((160, 160), False, True),  # value head > 128: 112-wide KV tiles
        ((192, 128), False, False),  # 128-wide KV tiles: the forward runs
        ((192, 128), True, True),  # the backward's KV tiles are 96 wide
    ],
)
def test_sm90_block_sparsity_limits(
    monkeypatch: pytest.MonkeyPatch,
    dims: tuple[int, int],
    backward: bool,
    expected: bool,
) -> None:
    _hopper(monkeypatch)
    got = usp_module._fa4_lacks_block_sparsity("cuda:0", *dims, backward=backward)
    assert got is expected


@pytest.mark.parametrize("capability", [(10, 0), (8, 0)])
def test_block_sparsity_limit_is_hopper_only(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int]
) -> None:
    _hopper(monkeypatch, capability)
    assert not usp_module._fa4_lacks_block_sparsity("cuda:0", 256, 256, backward=True)
    assert not usp_module._fa4_lacks_block_sparsity("cpu", 256, 256, backward=True)


@pytest.mark.parametrize(
    "path", [{"halo": object()}, {"fa4_mask": object()}, {"step_meta": object()}]
)
def test_block_sparse_plans_rejected_at_head_dim_256_on_h100(
    monkeypatch: pytest.MonkeyPatch, path: dict
) -> None:
    _hopper(monkeypatch)
    if "step_meta" in path:  # block-sparse steps (no native mask)
        path = {"step_meta": SimpleNamespace(native=None)}
    plan = _sentinel_plan(**path)
    with pytest.raises(ValueError, match="dense_ring=True"):
        usp_module._check_plan_supported(
            plan,
            device=torch.device("cuda:0"),
            head_dim=256,
            head_dim_v=256,
            backward=True,
        )


@pytest.mark.parametrize(
    "path",
    [
        {"dense_gather": object()},
        {"dense_ring": object()},
        {"generation_ring": object()},
        {"interval_plan": object()},
        {"native_mask": "causal"},
    ],
)
def test_dense_plans_run_at_head_dim_256_on_h100(
    monkeypatch: pytest.MonkeyPatch, path: dict
) -> None:
    _hopper(monkeypatch)
    usp_module._check_plan_supported(
        _sentinel_plan(**path),
        device=torch.device("cuda:0"),
        head_dim=256,
        head_dim_v=256,
        backward=True,
    )


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="needs an H100",
)
def test_gather_at_head_dim_256_on_h100_points_to_dense_ring() -> None:
    """Regression: the default gather plan is a block-sparse halo."""
    docs, splits, modes = build_layout("mixed", N, seed=0)
    kw = dict(
        document_lens=docs,
        split_lens=splits,
        attn_modes=modes,
        groups=_groups(2),
        device=torch.device("cuda"),
    )
    plan = build_usp_plan(**kw, config=USPConfig(kv_exchange="gather"))
    assert plan.halo is not None
    q, k, v = (
        torch.randn(
            1, plan.local_len, 4, 256, device="cuda", dtype=torch.bfloat16
        ).requires_grad_(True)
        for _ in range(3)
    )
    with pytest.raises(ValueError, match="dense_ring=True"):
        usp_attention(q, k, v, plan)
    # The image spans here are short: dense gather also needs dense_min_span=0.
    dense = build_usp_plan(
        **kw, config=USPConfig(kv_exchange="gather", dense_ring=True, dense_min_span=0)
    )
    assert dense.dense_gather is not None


# -- plan-time cost ---------------------------------------------------------------


@pytest.mark.parametrize("name", ["mixed", "vlm", "video", "docs"])
@pytest.mark.parametrize("world", [2, 4])
def test_one_rank_plan_matches_the_full_build(name: str, world: int) -> None:
    docs, splits, modes = build_layout(name, N, seed=1)
    layout = build_packed_layout(docs, splits, modes)
    kw = dict(world_size=world, device="cpu", use_direct_planner_block_sparse=True)
    full = build_ring_plan(layout, **kw)
    assert len(full.rank_plans) == world
    for rank in range(world):
        one = build_ring_plan(layout, **kw, ranks=[rank])
        assert torch.equal(one.token_permutation, full.token_permutation)
        (got,), want = one.rank_plans, full.rank_plans[rank]
        assert (got.rank, got.local_block_start, got.local_block_end) == (
            want.rank,
            want.local_block_start,
            want.local_block_end,
        )
        for a, b in zip(got.steps, want.steps, strict=True):
            assert (a.step, a.kv_owner_rank) == (b.step, b.kv_owner_rank)
            assert a.estimated_block_intersections == b.estimated_block_intersections
            assert torch.equal(a.active_local_q_blocks, b.active_local_q_blocks)
    none = build_ring_plan(layout, **kw, ranks=())
    assert none.rank_plans == []
    assert torch.equal(none.token_permutation, full.token_permutation)


@needs_fa4
def test_ring_plan_keeps_this_ranks_step_plan() -> None:
    plan = _plan(build_layout("mixed", N, seed=0), ring=2, config=USPConfig())
    assert plan.rank_plan is not None and plan.rank_plan.rank == 0
    assert len(plan.rank_plan.steps) == 2


@needs_fa4
def test_gather_builds_its_base_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Short image spans: the gather base skips the dense ring up front."""
    builds = []
    real = usp_module._build_usp_plan

    def counting(**kwargs):
        builds.append(kwargs["config"])
        return real(**kwargs)

    monkeypatch.setattr(usp_module, "_build_usp_plan", counting)
    plan = _plan(
        build_layout("vlm", N, seed=0),
        ring=2,
        config=USPConfig(kv_exchange="gather", dense_ring=True),
    )
    assert plan.halo is not None
    # The public call, then one order-only base without the dense ring.
    assert len(builds) == 2
    assert builds[1].order_only and builds[1].dense_ring is False


@needs_fa4
@pytest.mark.parametrize("exchange", ["ring", "gather"])
def test_sliding_window_base_is_order_only(
    monkeypatch: pytest.MonkeyPatch, exchange: str
) -> None:
    spec = build_layout("mixed", N, seed=0)
    config = USPConfig(dense_ring=False, kv_exchange=exchange)
    base = _plan(spec, ring=2, config=config)
    built = []
    real = usp_module.build_precomputed_cross_qkv_aux_plan

    def counting(**kwargs):
        built.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(usp_module, "build_precomputed_cross_qkv_aux_plan", counting)
    win = _plan(spec, ring=2, config=config, sliding_window=512)
    # Only the windowed plan's own step masks (none for a halo).
    assert len(built) <= 1 and all(kw.get("sliding_window") for kw in built)
    assert win.config == config and win.sliding_window == 512
    assert torch.equal(win.token_permutation, base.token_permutation)
    derived = build_sliding_window_plan(
        base,
        sliding_window=512,
        document_lens=spec[0],
        split_lens=spec[1],
        attn_modes=spec[2],
        device=torch.device("cpu"),
    )
    assert usp_module._execution_path(win) == usp_module._execution_path(derived)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_reference_planner_order_without_retained_block_mask() -> None:
    """PTRR without the direct planner builds the same BlockMask itself."""
    from usp_attention.fa4.flash4_attn import build_fa4_sparse_mask

    docs, splits, modes = build_layout("mixed", N, seed=0)
    device = torch.device("cuda")
    plan = build_usp_plan(
        document_lens=docs,
        split_lens=splits,
        attn_modes=modes,
        groups=_groups(2),
        config=USPConfig(dense_ring=False, direct_planner_block_sparse=False),
        device=device,
    )
    mask = build_fa4_sparse_mask(
        docs, splits, modes, device=device, retain_block_mask=True
    )
    layout = build_packed_layout(
        docs, splits, modes, device=device, aux_tensors=mask.aux_tensors
    )
    reference = build_ring_plan(
        layout, world_size=2, device=device, block_mask=mask.block_mask
    )
    assert torch.equal(plan.token_permutation, reference.token_permutation)
