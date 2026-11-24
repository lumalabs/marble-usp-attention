# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Segment, noise and document IDs past the CuTe DSL classifiers' bitmaps.

Both CuTe DSL classifiers (single-rank ``fa4`` and the opt-in cross-QKV
``ring`` copy) record each block's IDs in 16384-bit bitmaps. An ID past that
would drop out and turn attending blocks empty, so large IDs must reach the
PyTorch classifiers instead (or be rejected when the kernel is called
directly). Each layout below has one block pair that only a shared large ID
makes attend.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cutlass")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
        reason="the CuTe DSL classifiers need an sm_90+ GPU",
    ),
]

from usp_attention.fa4.flash4_block_sparse import (  # noqa: E402
    BLOCK_PARTIAL,
    classify_self_attention_blocks,
)
from usp_attention.fa4.flash4_block_sparse_cutedsl import (  # noqa: E402
    BITMAP_ID_LIMIT,
    BitmapIdRangeError,
    classify_self_attention_blocks_cute_dsl,
)

BLOCK = 128
IDS = [5, BITMAP_ID_LIMIT - 1, BITMAP_ID_LIMIT, 70_000]


def _segment_across_blocks(segment: int) -> list[torch.Tensor]:
    """One causal document; tokens 64..191 form one full segment.

    Q block 0 attends KV block 1 only through that segment.
    """
    fn = torch.full((2 * BLOCK,), -1.0, device="cuda")
    fn[64:192] = float(segment)
    noise = torch.full((2 * BLOCK,), -1.0, device="cuda")
    doc = torch.ones(2 * BLOCK, dtype=torch.int64, device="cuda")
    return [fn, noise, doc]


def _document_across_blocks(first_doc: int) -> list[torch.Tensor]:
    """Two causal documents, the first ending inside block 1.

    Q block 1 attends KV block 0 only through the first document.
    """
    fn = torch.full((2 * BLOCK,), -1.0, device="cuda")
    noise = torch.full((2 * BLOCK,), -1.0, device="cuda")
    doc = torch.full((2 * BLOCK,), first_doc, dtype=torch.int64, device="cuda")
    doc[192:] = first_doc + 1
    return [fn, noise, doc]


@pytest.mark.parametrize("segment", IDS)
def test_fa4_large_segment_ids_keep_blocks_attending(segment: int) -> None:
    result = classify_self_attention_blocks(
        *_segment_across_blocks(segment), block_size=BLOCK
    )
    assert result.block_type[0, 1].item() == BLOCK_PARTIAL


@pytest.mark.parametrize("first_doc", IDS)
def test_fa4_large_document_ids_keep_blocks_attending(first_doc: int) -> None:
    result = classify_self_attention_blocks(
        *_document_across_blocks(first_doc), block_size=BLOCK
    )
    assert result.block_type[1, 0].item() == BLOCK_PARTIAL


@pytest.mark.parametrize(
    ("build", "value"),
    [
        (_segment_across_blocks, BITMAP_ID_LIMIT),
        (_document_across_blocks, BITMAP_ID_LIMIT),
        (_document_across_blocks, -2),
    ],
    ids=["segment", "document", "negative_document"],
)
def test_fa4_cute_dsl_rejects_ids_past_its_bitmaps(build, value: int) -> None:
    with pytest.raises(BitmapIdRangeError):
        classify_self_attention_blocks_cute_dsl(*build(value), block_size=BLOCK)


def _cross_aux(segment: int) -> list[torch.Tensor]:
    """``_segment_across_blocks`` as the ring classifier's 8 cross-QKV tensors."""
    fn, noise, doc = _segment_across_blocks(segment)
    pos = torch.arange(fn.numel(), dtype=torch.int64, device="cuda")
    return [fn, fn.clone(), noise, noise.clone(), doc, doc.clone(), pos, pos.clone()]


@pytest.mark.parametrize("segment", IDS)
def test_ring_cute_dsl_large_ids_match_torch_op(segment: int) -> None:
    from usp_attention.ring.block_sparse import (
        BLOCK_PARTIAL as RING_BLOCK_PARTIAL,
    )
    from usp_attention.ring.block_sparse import (
        build_block_sparse_from_cross_qkv_aux,
    )
    from usp_attention.ring.block_sparse_cutedsl import (
        build_block_sparse_from_cross_qkv_aux_cutedsl,
    )

    aux = _cross_aux(segment)
    got = build_block_sparse_from_cross_qkv_aux_cutedsl(aux, block_size=BLOCK)
    want = build_block_sparse_from_cross_qkv_aux(aux, block_size=BLOCK)
    assert torch.equal(got.block_type, want.block_type)
    assert got.block_type[0, 1].item() == RING_BLOCK_PARTIAL
