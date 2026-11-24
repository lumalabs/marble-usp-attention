# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Output-equivalence tests: CuTe DSL block-sparse vs the torch reference.

``build_block_sparse_from_cross_qkv_aux_cutedsl`` is not a line-by-line
kernelization of the torch op -- the classify phase replaces the
``torch.unique`` membership tests with per-block id bitmaps -- so this file
checks bit-exact equality of every ``BlockSparseResult`` field (via
``torch.equal``, including the argsort tails of the BCSR index tensors)
across a wide sweep of production-shaped inputs: packed multi-document
layouts with causal/full/noise splits as built by ``_build_common_tensors``
in ``usp_attention.fa4.flash4_attn``, with both self-attention
(q == kv) and cross-shard (ring context-parallel) q/kv sides, split and
document boundaries that do not align with blocks, tail padding, and large
split counts (multi-word bitmaps).

Run with::

    cd projects/usp_attention
    source .venv/bin/activate
    pytest tests/ring/test_block_sparse_equivalence.py -v
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cutlass")

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
    pytest.skip(
        "CuTe DSL block-sparse kernels require an sm_90+ GPU (production runs "
        "H100s; CuTe DSL cannot compile for e.g. sm_75 T4 CI runners)",
        allow_module_level=True,
    )

from usp_attention.ring.block_sparse import (
    build_block_sparse_from_cross_qkv_aux,
)
from usp_attention.ring.block_sparse_cutedsl import (
    build_block_sparse_from_cross_qkv_aux_cutedsl,
)

_RESULT_TENSOR_FIELDS = (
    "block_type",
    "kv_num_blocks",
    "kv_indices",
    "full_kv_num_blocks",
    "full_kv_indices",
    "q_num_blocks",
    "q_indices",
    "full_q_num_blocks",
    "full_q_indices",
)

# Mode pools a layout draws its splits from. Single-mode pools exercise the
# reference's empty-unique-set branches (no noise ids / no fn ids anywhere).
_MODE_POOLS = (
    ("causal",),
    ("full",),
    ("noise",),
    ("causal", "full"),
    ("causal", "noise"),
    ("full", "noise"),
    ("causal", "full", "noise"),
)


def _make_packed_layout(
    seed: int,
    *,
    max_docs: int = 6,
    max_splits_per_doc: int = 6,
    max_split_len: int = 384,
    align: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(fn, noise, doc, pos) for a random packed multi-document layout.

    Mirrors production ``_build_common_tensors``: fn/noise are float32
    split indices (-1 outside full/noise splits), doc is 1-indexed int64,
    pos is the global arange. ``align`` rounds split lengths down to a
    multiple (production splits are block-aligned); None leaves them
    ragged so split and document boundaries fall inside blocks.
    """
    gen = torch.Generator().manual_seed(seed)

    def _randint(low: int, high: int) -> int:
        """Uniform int in [low, high]."""
        return int(torch.randint(low, high + 1, (1,), generator=gen).item())

    mode_pool = _MODE_POOLS[_randint(0, len(_MODE_POOLS) - 1)]
    fn_parts: list[torch.Tensor] = []
    noise_parts: list[torch.Tensor] = []
    doc_parts: list[torch.Tensor] = []
    split_idx = 0
    for doc_id in range(1, _randint(1, max_docs) + 1):
        for _ in range(_randint(1, max_splits_per_doc)):
            length = _randint(1, max_split_len)
            if align is not None:
                length = max(align, length - length % align)
            mode = mode_pool[_randint(0, len(mode_pool) - 1)]
            fn_val = float(split_idx) if mode in ("full", "noise") else -1.0
            noise_val = float(split_idx) if mode == "noise" else -1.0
            fn_parts.append(torch.full((length,), fn_val))
            noise_parts.append(torch.full((length,), noise_val))
            doc_parts.append(torch.full((length,), doc_id, dtype=torch.int64))
            split_idx += 1
    fn = torch.cat(fn_parts)
    noise = torch.cat(noise_parts)
    doc = torch.cat(doc_parts)
    pos = torch.arange(fn.numel(), dtype=torch.int64)
    return fn, noise, doc, pos


def _fixed_long_layout(
    total_min: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic layout of at least ``total_min`` tokens.

    Split lengths and modes cycle through fixed patterns (including
    single-token splits), with a new document every third split.
    """
    lengths = (1, 7, 64, 128, 300, 33)
    modes = ("causal", "full", "noise")
    fn_parts: list[torch.Tensor] = []
    noise_parts: list[torch.Tensor] = []
    doc_parts: list[torch.Tensor] = []
    split_idx = 0
    total = 0
    while total < total_min:
        length = lengths[split_idx % len(lengths)]
        mode = modes[split_idx % len(modes)]
        fn_val = float(split_idx) if mode in ("full", "noise") else -1.0
        noise_val = float(split_idx) if mode == "noise" else -1.0
        fn_parts.append(torch.full((length,), fn_val))
        noise_parts.append(torch.full((length,), noise_val))
        doc_parts.append(torch.full((length,), 1 + split_idx // 3, dtype=torch.int64))
        split_idx += 1
        total += length
    fn = torch.cat(fn_parts)
    noise = torch.cat(noise_parts)
    doc = torch.cat(doc_parts)
    pos = torch.arange(fn.numel(), dtype=torch.int64)
    return fn, noise, doc, pos


def _cross_aux(
    fn: torch.Tensor,
    noise: torch.Tensor,
    doc: torch.Tensor,
    pos: torch.Tensor,
    q_span: tuple[int, int],
    kv_span: tuple[int, int],
) -> list[torch.Tensor]:
    """8-tensor aux list with the q/kv sides sliced from one packed layout."""
    device = torch.device("cuda")
    q_lo, q_hi = q_span
    kv_lo, kv_hi = kv_span
    return [
        fn[q_lo:q_hi].to(device),
        fn[kv_lo:kv_hi].to(device),
        noise[q_lo:q_hi].to(device),
        noise[kv_lo:kv_hi].to(device),
        doc[q_lo:q_hi].to(device),
        doc[kv_lo:kv_hi].to(device),
        pos[q_lo:q_hi].to(device),
        pos[kv_lo:kv_hi].to(device),
    ]


def _assert_identical(aux: list[torch.Tensor], block_size: int, label: str) -> None:
    reference = build_block_sparse_from_cross_qkv_aux(aux, block_size=block_size)
    actual = build_block_sparse_from_cross_qkv_aux_cutedsl(aux, block_size=block_size)
    assert actual.n_q_blocks == reference.n_q_blocks, label
    assert actual.n_kv_blocks == reference.n_kv_blocks, label
    assert actual.block_size == reference.block_size, label
    for name in _RESULT_TENSOR_FIELDS:
        expected = getattr(reference, name)
        got = getattr(actual, name)
        assert got.dtype == expected.dtype, f"{label}: {name} dtype {got.dtype}"
        assert got.shape == expected.shape, f"{label}: {name} shape {got.shape}"
        assert torch.equal(expected, got), (
            f"{label}: {name} differs in {(expected != got).sum().item()} of "
            f"{expected.numel()} elements"
        )


class TestCuteDslMatchesReference:
    @pytest.mark.parametrize("seed", range(25))
    def test_self_attention_layouts(self, seed: int):
        """q == kv (the non-ring path), ragged split/document boundaries."""
        fn, noise, doc, pos = _make_packed_layout(seed)
        n = fn.numel()
        aux = _cross_aux(fn, noise, doc, pos, (0, n), (0, n))
        _assert_identical(aux, 128, f"self seed={seed} n={n}")

    @pytest.mark.parametrize("seed", range(25))
    def test_cross_shard_layouts(self, seed: int):
        """q/kv are different shards of one sequence (ring CP steps)."""
        fn, noise, doc, pos = _make_packed_layout(seed)
        n = fn.numel()
        gen = torch.Generator().manual_seed(seed + 10_000)

        def _span() -> tuple[int, int]:
            lo = int(torch.randint(0, n, (1,), generator=gen).item())
            hi = int(torch.randint(lo + 1, n + 1, (1,), generator=gen).item())
            return lo, hi

        aux = _cross_aux(fn, noise, doc, pos, _span(), _span())
        _assert_identical(aux, 128, f"cross seed={seed} n={n}")

    @pytest.mark.parametrize("block_size", [64, 128, 256])
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_block_sizes_aligned_splits(self, block_size: int, seed: int):
        """Block-aligned splits (production layouts) at several block sizes."""
        fn, noise, doc, pos = _make_packed_layout(seed, align=block_size)
        n = fn.numel()
        half = n // 2
        aux = _cross_aux(fn, noise, doc, pos, (0, n), (0, n))
        _assert_identical(aux, block_size, f"aligned self bs={block_size} seed={seed}")
        aux = _cross_aux(fn, noise, doc, pos, (0, half), (half, n))
        _assert_identical(aux, block_size, f"aligned split bs={block_size} seed={seed}")

    @pytest.mark.parametrize("mode", ["causal", "full", "noise"])
    def test_uniform_mode_sequence(self, mode: str):
        """One document, one 3000-token split of a single mode (tail-padded)."""
        length = 3000
        fn_val = -1.0 if mode == "causal" else 0.0
        noise_val = 0.0 if mode == "noise" else -1.0
        fn = torch.full((length,), fn_val)
        noise = torch.full((length,), noise_val)
        doc = torch.ones(length, dtype=torch.int64)
        pos = torch.arange(length, dtype=torch.int64)
        aux = _cross_aux(fn, noise, doc, pos, (0, length), (0, length))
        _assert_identical(aux, 128, f"uniform mode={mode}")

    def test_many_splits_multi_word_bitmaps(self):
        """~700 small splits: ids span >512 so bitmaps need many words."""
        gen = torch.Generator().manual_seed(7)
        fn_parts: list[torch.Tensor] = []
        noise_parts: list[torch.Tensor] = []
        doc_parts: list[torch.Tensor] = []
        modes = ("causal", "full", "noise")
        for split_idx in range(700):
            length = int(torch.randint(1, 17, (1,), generator=gen).item())
            mode = modes[split_idx % len(modes)]
            fn_val = float(split_idx) if mode in ("full", "noise") else -1.0
            noise_val = float(split_idx) if mode == "noise" else -1.0
            fn_parts.append(torch.full((length,), fn_val))
            noise_parts.append(torch.full((length,), noise_val))
            doc_parts.append(
                torch.full((length,), 1 + split_idx // 50, dtype=torch.int64)
            )
        fn = torch.cat(fn_parts)
        noise = torch.cat(noise_parts)
        doc = torch.cat(doc_parts)
        pos = torch.arange(fn.numel(), dtype=torch.int64)
        n = fn.numel()
        aux = _cross_aux(fn, noise, doc, pos, (0, n), (0, n))
        _assert_identical(aux, 128, "many splits")

    @pytest.mark.parametrize(
        ("q_len", "kv_len"),
        [
            (1, 1),
            (1, 500),
            (500, 1),
            (127, 129),
            (128, 128),
            (129, 4097),
        ],
    )
    def test_extreme_shard_lengths(self, q_len: int, kv_len: int):
        """Degenerate and off-by-one shard lengths from a fixed long layout."""
        fn, noise, doc, pos = _fixed_long_layout(total_min=max(q_len, kv_len) + 1)
        n = fn.numel()
        aux = _cross_aux(fn, noise, doc, pos, (0, q_len), (n - kv_len, n))
        _assert_identical(aux, 128, f"extreme q={q_len} kv={kv_len}")
