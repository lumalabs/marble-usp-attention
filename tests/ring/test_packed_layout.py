# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest
import torch

from usp_attention.ring.packed_layout import (
    build_packed_layout,
)


class TestBuildPackedLayout:
    def test_single_causal_document(self):
        layout = build_packed_layout(
            document_lens=[6],
            split_lens=[6],
            attn_modes=["causal"],
        )
        assert layout.seq_len == 6
        assert layout.full_and_noise_seq_id.shape == (6,)
        assert layout.noise_seq_id.shape == (6,)
        assert layout.document_id.shape == (6,)
        assert (layout.full_and_noise_seq_id == -1).all()
        assert (layout.noise_seq_id == -1).all()
        assert (layout.document_id == 1).all()

    def test_mixed_modes(self):
        layout = build_packed_layout(
            document_lens=[4, 4],
            split_lens=[2, 3, 3],
            attn_modes=["causal", "full", "noise"],
        )
        assert layout.seq_len == 8

        fn = layout.full_and_noise_seq_id
        assert (fn[:2] == -1).all(), "causal tokens should have fn_seq=-1"
        assert (fn[2:5] == 1).all(), "full tokens should share segment id 1"
        assert (fn[5:8] == 2).all(), "noise tokens should share segment id 2"

        ns = layout.noise_seq_id
        assert (ns[:5] == -1).all(), "non-noise tokens should have noise_seq=-1"
        assert (ns[5:8] == 2).all(), "noise tokens should share segment id 2"

        doc = layout.document_id
        assert (doc[:4] == 1).all()
        assert (doc[4:8] == 2).all()

    def test_with_aux_tensors(self):
        fn = torch.tensor([0.0, 0.0, -1.0, -1.0])
        ns = torch.tensor([-1.0, -1.0, -1.0, -1.0])
        doc = torch.tensor([1.0, 1.0, 2.0, 2.0])
        layout = build_packed_layout(
            document_lens=[2, 2],
            split_lens=[2, 2],
            attn_modes=["full", "causal"],
            aux_tensors=[fn, ns, doc],
        )
        assert layout.full_and_noise_seq_id is fn
        assert layout.noise_seq_id is ns
        assert layout.document_id is doc

    def test_mismatched_split_attn_lengths(self):
        with pytest.raises(ValueError, match="split_lens and attn_modes must match"):
            build_packed_layout(
                document_lens=[4],
                split_lens=[2, 2],
                attn_modes=["causal"],
            )

    def test_sum_mismatch(self):
        with pytest.raises(ValueError, match="Sum mismatch"):
            build_packed_layout(
                document_lens=[3],
                split_lens=[4],
                attn_modes=["causal"],
            )

    def test_zero_seq_len(self):
        with pytest.raises(ValueError, match="positive sequence length"):
            build_packed_layout(
                document_lens=[],
                split_lens=[],
                attn_modes=[],
            )

    def test_non_positive_split(self):
        with pytest.raises(ValueError, match="Split lengths must be positive"):
            build_packed_layout(
                document_lens=[4],
                split_lens=[0, 4],
                attn_modes=["causal", "causal"],
            )

    def test_wrong_aux_count(self):
        with pytest.raises(ValueError, match="aux_tensors must have exactly 3"):
            build_packed_layout(
                document_lens=[2],
                split_lens=[2],
                attn_modes=["causal"],
                aux_tensors=[torch.zeros(2), torch.zeros(2)],
            )


class TestPackedLayoutReordered:
    def test_reorder_preserves_metadata(self):
        layout = build_packed_layout(
            document_lens=[4],
            split_lens=[2, 2],
            attn_modes=["causal", "full"],
        )
        perm = torch.tensor([3, 2, 1, 0])
        reordered = layout.reordered(perm)

        assert reordered.seq_len == layout.seq_len
        assert reordered.document_lens == layout.document_lens
        assert reordered.split_lens == layout.split_lens
        assert reordered.attn_modes == layout.attn_modes

    def test_reorder_permutes_tensors(self):
        layout = build_packed_layout(
            document_lens=[4],
            split_lens=[2, 2],
            attn_modes=["causal", "full"],
        )
        perm = torch.tensor([3, 2, 1, 0])
        reordered = layout.reordered(perm)

        torch.testing.assert_close(
            reordered.full_and_noise_seq_id,
            layout.full_and_noise_seq_id[perm],
        )
        torch.testing.assert_close(
            reordered.noise_seq_id,
            layout.noise_seq_id[perm],
        )
        torch.testing.assert_close(
            reordered.document_id,
            layout.document_id[perm],
        )

    def test_double_reorder_roundtrip(self):
        layout = build_packed_layout(
            document_lens=[3, 3],
            split_lens=[3, 3],
            attn_modes=["causal", "noise"],
        )
        perm = torch.tensor([5, 4, 3, 2, 1, 0])
        reordered = layout.reordered(perm).reordered(perm)

        torch.testing.assert_close(
            reordered.full_and_noise_seq_id,
            layout.full_and_noise_seq_id,
        )
        torch.testing.assert_close(
            reordered.document_id,
            layout.document_id,
        )
