# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Oracle and property tests for single-rank FA4 block classification."""

from __future__ import annotations

import math
import random

import pytest
import torch

from usp_attention.fa4.flash4_block_sparse import (
    BLOCK_EMPTY,
    BLOCK_FULL,
    BLOCK_PARTIAL,
    classify_self_attention_blocks,
)


def _element_mask(
    full_and_noise_seq_id: torch.Tensor,
    noise_seq_id: torch.Tensor,
    document_id: torch.Tensor,
) -> torch.Tensor:
    """Independent dense implementation of the CuTe element predicate."""
    seq_len = full_and_noise_seq_id.numel()
    positions = torch.arange(seq_len)
    causal = positions[:, None] >= positions[None, :]
    full_noise = (full_and_noise_seq_id[:, None] == full_and_noise_seq_id[None, :]) & (
        full_and_noise_seq_id[:, None] >= 0
    )
    remove_noise = ~(
        (noise_seq_id[None, :] >= 0) & (noise_seq_id[:, None] != noise_seq_id[None, :])
    )
    same_document = document_id[:, None] == document_id[None, :]
    return (causal | full_noise) & remove_noise & same_document


def _oracle_block_types(dense_mask: torch.Tensor, *, block_size: int) -> torch.Tensor:
    seq_len = dense_mask.shape[0]
    n_blocks = math.ceil(seq_len / block_size)
    block_type = torch.full((n_blocks, n_blocks), BLOCK_PARTIAL, dtype=torch.int32)
    for q_block in range(n_blocks):
        q_start = q_block * block_size
        q_end = min(q_start + block_size, seq_len)
        for kv_block in range(n_blocks):
            kv_start = kv_block * block_size
            kv_end = min(kv_start + block_size, seq_len)
            block = dense_mask[q_start:q_end, kv_start:kv_end]
            if block.all():
                block_type[q_block, kv_block] = BLOCK_FULL
            elif not block.any():
                block_type[q_block, kv_block] = BLOCK_EMPTY
    return block_type


def _bcsr_to_dense(
    num_blocks_in_row: torch.Tensor,
    col_indices: torch.Tensor,
    *,
    n_blocks: int,
) -> torch.Tensor:
    counts = num_blocks_in_row.reshape(-1, n_blocks)[0].to(torch.int64)
    indices = col_indices.reshape(-1, n_blocks, col_indices.shape[-1])[0]
    dense = torch.zeros(n_blocks, n_blocks, dtype=torch.bool)
    for row in range(n_blocks):
        count = int(counts[row])
        dense[row, indices[row, :count].long()] = True
    return dense


def _assert_conservative(
    full_and_noise_seq_id: torch.Tensor,
    noise_seq_id: torch.Tensor,
    document_id: torch.Tensor,
    *,
    block_size: int,
) -> None:
    result = classify_self_attention_blocks(
        full_and_noise_seq_id,
        noise_seq_id,
        document_id,
        block_size=block_size,
    )
    oracle = _oracle_block_types(
        _element_mask(full_and_noise_seq_id, noise_seq_id, document_id),
        block_size=block_size,
    )

    direct_full = result.block_type == BLOCK_FULL
    direct_empty = result.block_type == BLOCK_EMPTY
    direct_partial = result.block_type == BLOCK_PARTIAL
    assert not (direct_full & (oracle != BLOCK_FULL)).any()
    assert not (direct_empty & (oracle != BLOCK_EMPTY)).any()
    assert not (direct_full & direct_partial).any()
    assert not (direct_empty & direct_partial).any()
    torch.testing.assert_close(result.block_type, oracle, rtol=0, atol=0)

    forward_partial = _bcsr_to_dense(
        result.kv_num_blocks, result.kv_indices, n_blocks=result.n_blocks
    )
    forward_full = _bcsr_to_dense(
        result.full_kv_num_blocks,
        result.full_kv_indices,
        n_blocks=result.n_blocks,
    )
    backward_partial = _bcsr_to_dense(
        result.q_num_blocks, result.q_indices, n_blocks=result.n_blocks
    )
    backward_full = _bcsr_to_dense(
        result.full_q_num_blocks,
        result.full_q_indices,
        n_blocks=result.n_blocks,
    )
    torch.testing.assert_close(forward_partial, direct_partial, rtol=0, atol=0)
    torch.testing.assert_close(forward_full, direct_full, rtol=0, atol=0)
    torch.testing.assert_close(backward_partial, forward_partial.T, rtol=0, atol=0)
    torch.testing.assert_close(backward_full, forward_full.T, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["causal", "full", "noise"])
@pytest.mark.parametrize("seq_len", [1, 7, 16, 17, 127, 128, 129])
def test_single_mode_matches_dense_oracle(mode: str, seq_len: int) -> None:
    segment_id = -1.0 if mode == "causal" else 0.0
    noise_id = 0.0 if mode == "noise" else -1.0
    fn = torch.full((seq_len,), segment_id)
    noise = torch.full((seq_len,), noise_id)
    doc = torch.zeros(seq_len, dtype=torch.int64)

    for block_size in (4, 16, 128):
        _assert_conservative(fn, noise, doc, block_size=block_size)


def test_mixed_causal_full_noise_documents_match_dense_oracle() -> None:
    fn = torch.tensor([-1.0] * 5 + [1.0] * 7 + [2.0] * 4 + [-1.0] * 3 + [4.0] * 9)
    noise = torch.tensor([-1.0] * 5 + [-1.0] * 7 + [2.0] * 4 + [-1.0] * 3 + [4.0] * 9)
    doc = torch.tensor([0] * 16 + [1] * 12, dtype=torch.int64)

    for block_size in (4, 8, 16):
        _assert_conservative(fn, noise, doc, block_size=block_size)


def test_mixed_noise_ids_remain_partial_when_element_mask_varies() -> None:
    fn = torch.full((8,), 7.0)
    noise = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    doc = torch.zeros(8, dtype=torch.int64)

    result = classify_self_attention_blocks(fn, noise, doc, block_size=4)

    assert result.block_type[0, 0] == BLOCK_PARTIAL
    assert result.block_type[0, 1] == BLOCK_PARTIAL
    assert result.block_type[1, 0] == BLOCK_PARTIAL
    assert result.block_type[1, 1] == BLOCK_FULL
    _assert_conservative(fn, noise, doc, block_size=4)


def test_disjoint_noise_segments_are_empty_in_both_directions() -> None:
    fn = torch.tensor([0.0] * 4 + [1.0] * 4)
    noise = torch.tensor([0.0] * 4 + [1.0] * 4)
    doc = torch.zeros(8, dtype=torch.int64)

    result = classify_self_attention_blocks(fn, noise, doc, block_size=4)

    assert result.block_type[0, 1] == BLOCK_EMPTY
    assert result.block_type[1, 0] == BLOCK_EMPTY
    _assert_conservative(fn, noise, doc, block_size=4)


def test_many_noise_categories_use_exact_block_summaries() -> None:
    num_categories = 512
    noise = torch.arange(num_categories, dtype=torch.float32)
    fn = noise.clone()
    doc = torch.zeros(num_categories, dtype=torch.int64)

    _assert_conservative(fn, noise, doc, block_size=8)


def test_nonmonotonic_cosmos_document_layout_matches_dense_oracle() -> None:
    text_lens = [3, 5, 2]
    gen_lens = [7, 4, 6]
    fn = torch.tensor(
        [-1.0] * sum(text_lens)
        + [
            float(sample)
            for sample, length in enumerate(gen_lens)
            for _ in range(length)
        ]
    )
    noise = torch.full_like(fn, -1.0)
    doc = torch.tensor(
        [sample for sample, length in enumerate(text_lens) for _ in range(length)]
        + [sample for sample, length in enumerate(gen_lens) for _ in range(length)],
        dtype=torch.int64,
    )

    for block_size in (4, 8, 16):
        _assert_conservative(fn, noise, doc, block_size=block_size)


def _random_packed_layout(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    fn_parts: list[torch.Tensor] = []
    noise_parts: list[torch.Tensor] = []
    doc_parts: list[torch.Tensor] = []
    segment_id = 0
    for document in range(rng.randint(1, 5)):
        for _ in range(rng.randint(1, 6)):
            length = rng.randint(0, 24)
            mode = rng.choice(("causal", "full", "noise"))
            fn_id = float(segment_id) if mode in {"full", "noise"} else -1.0
            noise_id = float(segment_id) if mode == "noise" else -1.0
            fn_parts.append(torch.full((length,), fn_id))
            noise_parts.append(torch.full((length,), noise_id))
            doc_parts.append(torch.full((length,), document, dtype=torch.int64))
            segment_id += 1
    if sum(part.numel() for part in fn_parts) == 0:
        fn_parts.append(torch.tensor([-1.0]))
        noise_parts.append(torch.tensor([-1.0]))
        doc_parts.append(torch.tensor([0], dtype=torch.int64))
    return torch.cat(fn_parts), torch.cat(noise_parts), torch.cat(doc_parts)


@pytest.mark.parametrize("seed", range(64))
def test_random_packed_layout_is_conservative(seed: int) -> None:
    fn, noise, doc = _random_packed_layout(seed)
    for block_size in (4, 16, 64, 128):
        _assert_conservative(fn, noise, doc, block_size=block_size)


def test_invalid_aux_tensors_are_rejected() -> None:
    valid = torch.zeros(4)
    doc = torch.zeros(4, dtype=torch.int64)

    with pytest.raises(ValueError, match="positive"):
        classify_self_attention_blocks(valid, valid, doc, block_size=0)
    with pytest.raises(ValueError, match="one-dimensional"):
        classify_self_attention_blocks(valid[:, None], valid, doc)
    with pytest.raises(ValueError, match="equal lengths"):
        classify_self_attention_blocks(valid, valid[:-1], doc)
    with pytest.raises(ValueError, match="non-empty"):
        classify_self_attention_blocks(valid[:0], valid[:0], doc[:0])
