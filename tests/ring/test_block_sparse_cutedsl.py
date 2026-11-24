# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Tests for direct block-sparsity classification, CuTe DSL implementation.

Copy of ``block_sparse_test.py`` running against
``build_block_sparse_from_cross_qkv_aux_cutedsl``; requires a CUDA device.

Run with::

    cd projects/usp_attention
    source .venv/bin/activate
    pytest tests/ring/test_block_sparse_cutedsl.py -v

The performance and create_block_mask cross-validation classes are gated on
``torch.cuda.device_count() >= 4`` (no ``-m`` marker; they skip on smaller
nodes).
"""

from __future__ import annotations

import time

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
    BLOCK_EMPTY,
    BLOCK_FULL,
    BLOCK_PARTIAL,
)
from usp_attention.ring.block_sparse_cutedsl import (
    build_block_sparse_from_cross_qkv_aux_cutedsl,
)


def _make_causal_aux(seq_len: int, *, device: str = "cuda") -> list[torch.Tensor]:
    """Build cross-QKV aux tensors for a pure causal single-document mask."""
    fn = torch.full((seq_len,), -1.0, device=device)
    noise = torch.full((seq_len,), -1.0, device=device)
    doc = torch.ones(seq_len, dtype=torch.int64, device=device)
    pos = torch.arange(seq_len, dtype=torch.int64, device=device)
    return [fn, fn.clone(), noise, noise.clone(), doc, doc.clone(), pos, pos.clone()]


def _make_full_attn_aux(seq_len: int, *, device: str = "cuda") -> list[torch.Tensor]:
    """Build cross-QKV aux for a single full-attention segment (fn=0 everywhere)."""
    fn = torch.zeros(seq_len, dtype=torch.float32, device=device)
    noise = torch.full((seq_len,), -1.0, device=device)
    doc = torch.ones(seq_len, dtype=torch.int64, device=device)
    pos = torch.arange(seq_len, dtype=torch.int64, device=device)
    return [fn, fn.clone(), noise, noise.clone(), doc, doc.clone(), pos, pos.clone()]


def _make_two_doc_causal_aux(
    doc1_len: int, doc2_len: int, *, device: str = "cuda"
) -> list[torch.Tensor]:
    """Build cross-QKV aux for two causal documents packed together."""
    seq_len = doc1_len + doc2_len
    fn = torch.full((seq_len,), -1.0, device=device)
    noise = torch.full((seq_len,), -1.0, device=device)
    doc = torch.cat(
        [
            torch.ones(doc1_len, dtype=torch.int64, device=device),
            torch.full((doc2_len,), 2, dtype=torch.int64, device=device),
        ]
    )
    pos = torch.arange(seq_len, dtype=torch.int64, device=device)
    return [fn, fn.clone(), noise, noise.clone(), doc, doc.clone(), pos, pos.clone()]


def _make_multi_doc_mixed_aux(
    *,
    seed: int = 0,
    device: str = "cuda",
) -> list[torch.Tensor]:
    """Build cross-QKV aux for 4 documents with random causal/full/noise segments.

    Each document has 2-4 segments with randomly chosen attention modes.
    Documents are 128-384 tokens each (multiples of 128 for clean block
    boundaries). Total sequence is the concatenation of all documents.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    modes = ["causal", "full", "noise"]
    fn_parts: list[torch.Tensor] = []
    noise_parts: list[torch.Tensor] = []
    doc_parts: list[torch.Tensor] = []
    pos_parts: list[torch.Tensor] = []

    segment_id = 0
    offset = 0
    for doc_id in range(1, 5):
        n_segments = int(torch.randint(2, 5, (1,), generator=rng).item())
        for _ in range(n_segments):
            seg_len = int(torch.randint(1, 4, (1,), generator=rng).item()) * 128
            mode = modes[int(torch.randint(0, 3, (1,), generator=rng).item())]

            if mode == "causal":
                fn_parts.append(torch.full((seg_len,), -1.0, device=device))
                noise_parts.append(torch.full((seg_len,), -1.0, device=device))
            elif mode == "full":
                fn_parts.append(
                    torch.full((seg_len,), float(segment_id), device=device)
                )
                noise_parts.append(torch.full((seg_len,), -1.0, device=device))
            else:
                fn_parts.append(
                    torch.full((seg_len,), float(segment_id), device=device)
                )
                noise_parts.append(
                    torch.full((seg_len,), float(segment_id), device=device)
                )
            segment_id += 1

            doc_parts.append(
                torch.full((seg_len,), doc_id, dtype=torch.int64, device=device)
            )
            pos_parts.append(
                torch.arange(offset, offset + seg_len, dtype=torch.int64, device=device)
            )
            offset += seg_len

    fn = torch.cat(fn_parts)
    noise = torch.cat(noise_parts)
    doc = torch.cat(doc_parts)
    pos = torch.cat(pos_parts)
    return [fn, fn.clone(), noise, noise.clone(), doc, doc.clone(), pos, pos.clone()]


class TestBlockSparseFromAux:
    def test_pure_causal_block_types(self):
        """Pure causal: above-diagonal empty, diagonal partial, below-diagonal full."""
        block_size = 4
        seq_len = 16
        aux = _make_causal_aux(seq_len)
        result = build_block_sparse_from_cross_qkv_aux_cutedsl(
            aux, block_size=block_size
        )

        assert result.n_q_blocks == 4
        assert result.n_kv_blocks == 4
        bt = result.block_type

        for q in range(4):
            for kv in range(4):
                if q > kv:
                    assert bt[q, kv] == BLOCK_FULL, (
                        f"below-diagonal ({q},{kv}) should be full"
                    )
                elif q == kv:
                    assert bt[q, kv] == BLOCK_PARTIAL, (
                        f"diagonal ({q},{kv}) should be partial"
                    )
                else:
                    assert bt[q, kv] == BLOCK_EMPTY, (
                        f"above-diagonal ({q},{kv}) should be empty"
                    )

    def test_full_attention_all_full(self):
        """Single full-attention segment: all blocks should be full."""
        block_size = 4
        seq_len = 12
        aux = _make_full_attn_aux(seq_len)
        result = build_block_sparse_from_cross_qkv_aux_cutedsl(
            aux, block_size=block_size
        )

        assert (result.block_type == BLOCK_FULL).all()

    def test_two_documents_cross_empty(self):
        """Two documents: cross-document blocks should be empty."""
        block_size = 4
        aux = _make_two_doc_causal_aux(8, 8)
        result = build_block_sparse_from_cross_qkv_aux_cutedsl(
            aux, block_size=block_size
        )

        bt = result.block_type
        for q in range(4):
            for kv in range(4):
                q_is_doc1 = q < 2
                kv_is_doc1 = kv < 2
                if q_is_doc1 != kv_is_doc1:
                    assert bt[q, kv] == BLOCK_EMPTY, (
                        f"cross-doc ({q},{kv}) should be empty"
                    )

    def test_bcsr_format_consistency(self):
        """Partial BCSR indices should reconstruct the partial-only block set."""
        block_size = 4
        aux = _make_two_doc_causal_aux(8, 8)
        result = build_block_sparse_from_cross_qkv_aux_cutedsl(
            aux, block_size=block_size
        )

        partial_blocks = result.block_type == BLOCK_PARTIAL
        kv_counts = result.kv_num_blocks[0, 0]
        kv_idx = result.kv_indices[0, 0]
        for q in range(result.n_q_blocks):
            count = int(kv_counts[q].item())
            bcsr_set = set(kv_idx[q, :count].tolist())
            dense_set = set(torch.where(partial_blocks[q])[0].tolist())
            assert bcsr_set == dense_set, (
                f"row {q}: partial BCSR {bcsr_set} != dense {dense_set}"
            )

    def test_full_bcsr_format_consistency(self):
        """Full-only BCSR should match the full blocks in block_type."""
        block_size = 4
        aux = _make_causal_aux(16)
        result = build_block_sparse_from_cross_qkv_aux_cutedsl(
            aux, block_size=block_size
        )

        full_blocks = result.block_type == BLOCK_FULL
        full_counts = result.full_kv_num_blocks[0, 0]
        full_idx = result.full_kv_indices[0, 0]
        for q in range(result.n_q_blocks):
            count = int(full_counts[q].item())
            bcsr_set = set(full_idx[q, :count].tolist())
            dense_set = set(torch.where(full_blocks[q])[0].tolist())
            assert bcsr_set == dense_set, (
                f"row {q}: full BCSR {bcsr_set} != dense {dense_set}"
            )

    def test_backward_transpose_consistency(self):
        """Q-indexed (backward) BCSR should be the transpose of KV-indexed partial blocks."""
        block_size = 4
        aux = _make_two_doc_causal_aux(8, 8)
        result = build_block_sparse_from_cross_qkv_aux_cutedsl(
            aux, block_size=block_size
        )

        partial_blocks = result.block_type == BLOCK_PARTIAL
        transposed = partial_blocks.t()

        q_counts = result.q_num_blocks[0, 0]
        q_idx = result.q_indices[0, 0]
        for kv in range(result.n_kv_blocks):
            count = int(q_counts[kv].item())
            bcsr_set = set(q_idx[kv, :count].tolist())
            dense_set = set(torch.where(transposed[kv])[0].tolist())
            assert bcsr_set == dense_set, (
                f"bwd row {kv}: Q BCSR {bcsr_set} != dense {dense_set}"
            )

    def test_validation_error(self):
        with pytest.raises(ValueError, match="Expected 8 aux tensors"):
            build_block_sparse_from_cross_qkv_aux_cutedsl([torch.zeros(4)] * 3)

    def test_non_divisible_padded(self):
        """Seq length not divisible by block_size should be padded correctly."""
        block_size = 4
        seq_len = 10
        aux = _make_causal_aux(seq_len)
        result = build_block_sparse_from_cross_qkv_aux_cutedsl(
            aux, block_size=block_size
        )
        assert result.n_q_blocks == 3
        assert result.n_kv_blocks == 3

    def test_mixed_causal_and_full(self):
        """Mixed mask: 4 causal tokens + 4 full-attention tokens (same doc)."""
        block_size = 4
        seq_len = 8
        fn = torch.tensor([-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0], device="cuda")
        noise = torch.full((seq_len,), -1.0, device="cuda")
        doc = torch.ones(seq_len, dtype=torch.int64, device="cuda")
        pos = torch.arange(seq_len, dtype=torch.int64, device="cuda")
        aux = [fn, fn.clone(), noise, noise.clone(), doc, doc.clone(), pos, pos.clone()]
        result = build_block_sparse_from_cross_qkv_aux_cutedsl(
            aux, block_size=block_size
        )

        bt = result.block_type
        assert bt[1, 1] == BLOCK_FULL, "full-attn block (1,1) should be full"
        assert bt[0, 0] == BLOCK_PARTIAL, (
            "causal diagonal block (0,0) should be partial"
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_multi_doc_mixed_runs(self, seed: int):
        """Multi-document with random modes: no crashes, valid block_type range."""
        aux = _make_multi_doc_mixed_aux(seed=seed)
        result = build_block_sparse_from_cross_qkv_aux_cutedsl(aux, block_size=128)
        assert (result.block_type >= BLOCK_EMPTY).all()
        assert (result.block_type <= BLOCK_FULL).all()
        n_q = result.n_q_blocks
        n_kv = result.n_kv_blocks
        assert result.block_type.shape == (n_q, n_kv)
        partial_count = int(result.kv_num_blocks.flatten().sum().item())
        full_count = int(result.full_kv_num_blocks.flatten().sum().item())
        assert partial_count + full_count <= n_q * n_kv

    def test_noise_masking_different_noise_ids(self):
        """Different KV noise IDs block all attention, even below diagonal.

        Block (0,1): fn=0 vs fn=1, different segments. full_noise fails
        (fn_q != fn_kv), so mask = causal only. Since q_pos_max=3 < kv_pos_min=4,
        the block is empty. Block (1,0): q_pos_min=4 > kv_pos_max=3, but the
        KV block has noise id 0 while the Q block has noise id 1, so
        ``remove_noise`` is false for all element pairs and the block is empty.
        """
        block_size = 4
        seq_len = 8
        fn = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], device="cuda")
        noise = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], device="cuda")
        doc = torch.ones(seq_len, dtype=torch.int64, device="cuda")
        pos = torch.arange(seq_len, dtype=torch.int64, device="cuda")
        aux = [fn, fn.clone(), noise, noise.clone(), doc, doc.clone(), pos, pos.clone()]
        result = build_block_sparse_from_cross_qkv_aux_cutedsl(
            aux, block_size=block_size
        )

        bt = result.block_type
        assert bt[0, 0] == BLOCK_FULL, "same segment block (0,0) should be full"
        assert bt[1, 1] == BLOCK_FULL, "same segment block (1,1) should be full"
        assert bt[0, 1] == BLOCK_EMPTY, (
            "cross-segment above-diagonal (0,1) should be empty"
        )
        assert bt[1, 0] == BLOCK_EMPTY, (
            "cross-segment below-diagonal (1,0) should be empty"
        )

    def test_noise_masking_kv_without_noise_is_full(self):
        """A KV block without noise passes regardless of Q noise IDs."""
        block_size = 4
        seq_len = 8
        fn = torch.tensor([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0], device="cuda")
        noise = torch.tensor(
            [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0], device="cuda"
        )
        doc = torch.ones(seq_len, dtype=torch.int64, device="cuda")
        pos = torch.arange(seq_len, dtype=torch.int64, device="cuda")
        aux = [fn, fn.clone(), noise, noise.clone(), doc, doc.clone(), pos, pos.clone()]
        result = build_block_sparse_from_cross_qkv_aux_cutedsl(
            aux, block_size=block_size
        )

        bt = result.block_type
        assert bt[0, 1] == BLOCK_EMPTY, "above-diagonal block (0,1) should be empty"
        assert bt[1, 0] == BLOCK_FULL, "below-diagonal block (1,0) should be full"

    def test_full_attn_segment_with_mixed_q_noise_is_partial(self):
        """Full-attn block with mixed Q noise IDs must not be marked FULL.

        Reference remove_noise for a pair is
        ``(kv_noise < 0) | (q_noise == kv_noise)``. For "every pair in the
        block satisfies remove_noise", we need either every KV token to
        have noise < 0, or every Q token to share the same positive
        noise ID that is also the only positive KV noise ID present.

        Here both Q and KV have noise IDs {0, 1} inside a single
        full-attention fn_seq segment. The block-level subset check
        ``kv_subset_of_q`` trivially holds, but pairs like
        ``(q_noise=0, kv_noise=1)`` fail remove_noise. The classifier
        must mark such blocks PARTIAL (so the kernel evaluates mask_mod),
        not FULL.
        """
        block_size = 4
        seq_len = 8
        fn = torch.full((seq_len,), 7.0, device="cuda")
        noise = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], device="cuda")
        doc = torch.ones(seq_len, dtype=torch.int64, device="cuda")
        pos = torch.arange(seq_len, dtype=torch.int64, device="cuda")
        aux = [fn, fn.clone(), noise, noise.clone(), doc, doc.clone(), pos, pos.clone()]
        result = build_block_sparse_from_cross_qkv_aux_cutedsl(
            aux, block_size=block_size
        )

        bt = result.block_type
        assert bt[0, 0] == BLOCK_PARTIAL, (
            "diagonal full-attn block with mixed Q noise ids must be PARTIAL, "
            f"got {bt[0, 0].item()}"
        )
        assert bt[0, 1] == BLOCK_PARTIAL, (
            "above-diagonal full-attn block where Q has mixed positive noise "
            "ids and KV has a strict positive noise subset must be PARTIAL "
            f"(kv_subset_of_q alone is insufficient), got {bt[0, 1].item()}"
        )
        assert bt[1, 0] == BLOCK_PARTIAL, (
            "below-diagonal block with mixed KV noise ids and uniform-0 Q "
            f"noise must be PARTIAL, got {bt[1, 0].item()}"
        )
        assert bt[1, 1] == BLOCK_FULL, (
            "both Q and KV uniformly noise=0 inside a full-attn segment "
            f"should be FULL, got {bt[1, 1].item()}"
        )


class TestBlockSparsePerformance:
    """Benchmark direct classifier vs dense create_block_mask."""

    @pytest.fixture(autouse=True)
    def _require_cuda(self):
        if (
            not hasattr(torch, "cuda")
            or not hasattr(torch.cuda, "is_available")
            or not hasattr(torch.cuda, "device_count")
        ):
            pytest.skip("CUDA not available")
        if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
            pytest.skip(
                "CUDA not available or less than 4 GPUs (only run on nodes with at least 4 GPUs to prevent long running tests)"
            )

    @pytest.mark.parametrize("seq_len", [512, 1024, 2048, 4096])
    def test_direct_speed(self, seq_len: int):
        aux = _make_two_doc_causal_aux(seq_len // 2, seq_len // 2)

        n_warmup = 2
        n_iters = 10
        for _ in range(n_warmup):
            build_block_sparse_from_cross_qkv_aux_cutedsl(aux, block_size=128)

        t0 = time.perf_counter()
        for _ in range(n_iters):
            build_block_sparse_from_cross_qkv_aux_cutedsl(aux, block_size=128)
        elapsed = (time.perf_counter() - t0) / n_iters * 1000

        print(f"\nseq_len={seq_len}: direct={elapsed:.2f}ms")


def _compare_bcsr_sets(
    num_blocks_a: torch.Tensor,
    indices_a: torch.Tensor,
    num_blocks_b: torch.Tensor,
    indices_b: torch.Tensor,
    label: str,
) -> None:
    """Assert that two BCSR representations encode the same block sets per row."""
    counts_a = num_blocks_a.flatten()
    counts_b = num_blocks_b.flatten()
    assert counts_a.shape == counts_b.shape, (
        f"{label}: row count shapes differ: {counts_a.shape} vs {counts_b.shape}"
    )
    idx_a = indices_a.reshape(counts_a.shape[0], -1)
    idx_b = indices_b.reshape(counts_b.shape[0], -1)
    for row in range(counts_a.shape[0]):
        ca = int(counts_a[row].item())
        cb = int(counts_b[row].item())
        set_a = set(idx_a[row, :ca].tolist())
        set_b = set(idx_b[row, :cb].tolist())
        assert set_a == set_b, (
            f"{label} row {row}: direct {set_a} (count={ca}) "
            f"vs reference {set_b} (count={cb})"
        )


class TestBlockSparseVsReference:
    """Cross-validate direct classifier against create_block_mask reference.

    These tests require a CUDA GPU because ``build_step_block_sparse_tensors_from_aux``
    calls ``torch.compile(create_block_mask)`` which needs GPU.

    Run with::

        pytest -m gpu tests/ring/test_block_sparse_cutedsl.py -v -s
    """

    @pytest.fixture(autouse=True)
    def _require_cuda(self):
        if (
            not hasattr(torch, "cuda")
            or not hasattr(torch.cuda, "is_available")
            or not hasattr(torch.cuda, "device_count")
        ):
            pytest.skip("CUDA not available")
        if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
            pytest.skip(
                "CUDA not available or less than 4 GPUs (only run on nodes with at least 4 GPUs to prevent long running tests)"
            )

    def _compare(self, aux: list[torch.Tensor], block_size: int = 128) -> None:
        """Run both paths and assert safe block-sparse output.

        The direct classifier is conservative: it may mark some full blocks
        as partial (safe -- kernel evaluates mask_mod unnecessarily) but must
        never mark a non-empty block as empty (unsafe -- kernel skips it).

        We verify:
        1. The union (partial + full) matches exactly per row -- no blocks
           are incorrectly skipped or invented.
        2. Our full set is a subset of the reference full set (we never
           skip mask_mod on a block that needs it).
        """
        from usp_attention.ring.flash4_ring_attn import (
            build_step_block_sparse_direct,
            build_step_block_sparse_tensors_from_aux,
        )

        cuda_aux = [t.cuda() for t in aux]
        direct = build_step_block_sparse_direct(cuda_aux, block_size=block_size)
        reference = build_step_block_sparse_tensors_from_aux(cuda_aux)

        assert direct.block_size == reference.block_size, (
            f"block_size: {direct.block_size} vs {reference.block_size}"
        )

        for (
            label_prefix,
            d_partial_cnt,
            d_partial_idx,
            d_full_cnt,
            d_full_idx,
            r_partial_cnt,
            r_partial_idx,
            r_full_cnt,
            r_full_idx,
        ) in [
            (
                "fwd",
                direct.mask_block_cnt,
                direct.mask_block_idx,
                direct.full_block_cnt,
                direct.full_block_idx,
                reference.mask_block_cnt,
                reference.mask_block_idx,
                reference.full_block_cnt,
                reference.full_block_idx,
            ),
            (
                "bwd",
                direct.bwd_mask_block_cnt,
                direct.bwd_mask_block_idx,
                direct.bwd_full_block_cnt,
                direct.bwd_full_block_idx,
                reference.bwd_mask_block_cnt,
                reference.bwd_mask_block_idx,
                reference.bwd_full_block_cnt,
                reference.bwd_full_block_idx,
            ),
        ]:
            if r_partial_cnt is None or d_partial_cnt is None:
                continue

            d_p_counts = d_partial_cnt.flatten()
            d_p_idx = d_partial_idx.reshape(d_p_counts.shape[0], -1)
            r_p_counts = r_partial_cnt.flatten()
            r_p_idx = r_partial_idx.reshape(r_p_counts.shape[0], -1)

            d_f_counts = (
                d_full_cnt.flatten()
                if d_full_cnt is not None
                else torch.zeros_like(d_p_counts)
            )
            d_f_idx = (
                d_full_idx.reshape(d_f_counts.shape[0], -1)
                if d_full_idx is not None
                else d_p_idx
            )
            r_f_counts = (
                r_full_cnt.flatten()
                if r_full_cnt is not None
                else torch.zeros_like(r_p_counts)
            )
            r_f_idx = (
                r_full_idx.reshape(r_f_counts.shape[0], -1)
                if r_full_idx is not None
                else r_p_idx
            )

            for row in range(d_p_counts.shape[0]):
                d_partial = set(d_p_idx[row, : int(d_p_counts[row])].tolist())
                d_full = set(d_f_idx[row, : int(d_f_counts[row])].tolist())
                r_partial = set(r_p_idx[row, : int(r_p_counts[row])].tolist())
                r_full = set(r_f_idx[row, : int(r_f_counts[row])].tolist())

                d_union = d_partial | d_full
                r_union = r_partial | r_full

                assert d_union == r_union, (
                    f"{label_prefix} row {row}: union mismatch. "
                    f"direct={d_union} vs reference={r_union}. "
                    f"Missing from direct: {r_union - d_union}. "
                    f"Extra in direct: {d_union - r_union}."
                )

                assert d_full <= r_full, (
                    f"{label_prefix} row {row}: direct marks blocks as full that "
                    f"reference does not: {d_full - r_full}. "
                    f"direct_full={d_full}, ref_full={r_full}"
                )

    def test_causal_single_doc(self):
        aux = _make_causal_aux(512)
        self._compare(aux)

    def test_full_attention(self):
        aux = _make_full_attn_aux(512)
        self._compare(aux)

    def test_two_documents(self):
        aux = _make_two_doc_causal_aux(256, 256)
        self._compare(aux)

    def test_mixed_modes(self):
        seq_len = 512
        half = seq_len // 2
        fn = torch.cat(
            [
                torch.full((half,), -1.0),
                torch.zeros(half),
            ]
        )
        noise = torch.full((seq_len,), -1.0, device="cuda")
        doc = torch.ones(seq_len, dtype=torch.int64, device="cuda")
        pos = torch.arange(seq_len, dtype=torch.int64, device="cuda")
        aux = [fn, fn.clone(), noise, noise.clone(), doc, doc.clone(), pos, pos.clone()]
        self._compare(aux)

    @pytest.mark.parametrize("seq_len", [256, 512, 1024])
    def test_various_lengths(self, seq_len: int):
        aux = _make_two_doc_causal_aux(seq_len // 2, seq_len // 2)
        self._compare(aux)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_multi_doc_random_modes(self, seed: int):
        """4 documents with random causal/full/noise segment permutations."""
        aux = _make_multi_doc_mixed_aux(seed=seed)
        self._compare(aux)
