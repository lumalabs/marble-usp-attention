# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""CuTe DSL kernels for exact FA4 self-attention block classification."""

from __future__ import annotations

import functools
from collections.abc import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, select_

from usp_attention.fa4.cute_dsl_utils import (
    fmin_f32,
    get_ptr_as_int64,
    imax_s64,
    imin_s64,
    ld_global_v2_u64,
    ld_global_v4_f32,
    max_min_fp32,
    max_min_i64,
    warp_reduce,
)
from usp_attention.fa4.flash4_block_sparse import (
    BLOCK_EMPTY,
    BLOCK_FULL,
    BLOCK_PARTIAL,
    SelfAttentionBlockSparseResult,
    _pad_to_blocks,
)

_FP32_VEC_SIZE = 4
_I64_VEC_SIZE = 2
_NUM_SM_BLOCKS = 4
_MAX_THREADS_PER_BLOCK = 1024
_MIN_THREADS = 128
_MAX_THREADS = 256
_W_CAP = 512
# Per-block ID bitmaps hold _W_CAP 32-bit words: IDs 0 .. BITMAP_ID_LIMIT - 1.
BITMAP_ID_LIMIT = _W_CAP * 32
_CLASSIFY_TILE_KV = 128
_BCSR_WARPS_PER_CTA = 4


class BitmapIdRangeError(ValueError):
    """An ID does not fit the classifier's per-block bitmaps."""


def _check_ids_fit_bitmaps(
    full_and_noise_seq_id: torch.Tensor,
    noise_seq_id: torch.Tensor,
    document_id: torch.Tensor,
) -> None:
    """Raise unless every ID fits the per-block bitmaps (one host sync).

    The kernels record each block's segment, noise and document IDs as bits
    in ``BITMAP_ID_LIMIT``-bit bitmaps. An ID outside the bitmap would drop
    out, and the blocks holding it would be classified as if it were absent
    (for example empty although they attend).
    """
    fits = (
        (full_and_noise_seq_id.max() < BITMAP_ID_LIMIT)
        & (noise_seq_id.max() < BITMAP_ID_LIMIT)
        & (document_id.min() >= 0)
        & (document_id.max() < BITMAP_ID_LIMIT)
    )
    if not bool(fits):
        raise BitmapIdRangeError(
            "The CuTe DSL block classifier needs segment and noise IDs below "
            f"{BITMAP_ID_LIMIT} and document IDs in [0, {BITMAP_ID_LIMIT}); "
            "larger IDs would drop out of its per-block bitmaps and misclassify "
            "blocks. classify_self_attention_blocks falls back to the PyTorch "
            "classifier for such layouts."
        )


@cute.jit
def _ext_words(max_value: Float32) -> Int32:
    raw = (Int32(max_value) >> 5) + 1
    extent = Int32(select_(max_value >= _W_CAP * 32.0, Int32(_W_CAP), raw))
    return Int32(select_(max_value >= 0.0, extent, Int32(0)))


@cute.jit
def _ext_words_i64(max_value: Int64) -> Int32:
    raw = (Int32(max_value) >> 5) + 1
    extent = Int32(select_(max_value >= _W_CAP * 32, Int32(_W_CAP), raw))
    return Int32(select_(max_value >= 0, extent, Int32(0)))


def _compute_optimal_threads(block_size: int) -> tuple[int, int]:
    """Match the proven cross-QKV stats-kernel launch geometry."""
    threads_per_row = block_size // _FP32_VEC_SIZE
    allowed = (1, 2, 4, 8, 16, 32)
    threads_per_row = max(value for value in allowed if value <= threads_per_row)
    num_threads = (_MAX_THREADS // threads_per_row) * threads_per_row
    if num_threads >= _MIN_THREADS:
        return num_threads, threads_per_row
    return _MAX_THREADS, threads_per_row


class SelfAttentionClassifyKernel:
    """Compute block summaries and exact block classes in two launches."""

    def __init__(self, block_size: int):
        self.block_size = block_size
        self.num_threads, self.threads_per_row = _compute_optimal_threads(block_size)
        self.rows_per_cta = self.num_threads // self.threads_per_row
        self.col_chunks = block_size // (self.threads_per_row * _FP32_VEC_SIZE)
        self.bitmap_threads = max(32, min(block_size, _MAX_THREADS_PER_BLOCK))
        self.bitmap_elements = (
            block_size + self.bitmap_threads - 1
        ) // self.bitmap_threads
        self.tile_kv = _CLASSIFY_TILE_KV

    @cute.jit
    def __call__(
        self,
        fn: cute.Tensor,
        noise: cute.Tensor,
        doc: cute.Tensor,
        f32_stats: cute.Tensor,
        i64_stats: cute.Tensor,
        bits: cute.Tensor,
        block_type: cute.Tensor,
        n_blocks: Int32,
        n_stats_ctas: Int32,
        n_classify_tiles: Int32,
        valid_tail: Int32,
        stream,
    ):
        self.stats_kernel(fn, noise, doc, f32_stats, i64_stats, n_blocks).launch(
            grid=[n_stats_ctas, 1, 1],
            block=[self.num_threads, 1, 1],
            max_number_threads=[_MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=_NUM_SM_BLOCKS,
            stream=stream,
        )
        self.bitmap_kernel(fn, noise, doc, f32_stats, i64_stats, bits, n_blocks).launch(
            grid=[n_blocks, 1, 1],
            block=[self.bitmap_threads, 1, 1],
            stream=stream,
        )
        self.classify_kernel(
            fn,
            noise,
            doc,
            f32_stats,
            i64_stats,
            bits,
            block_type,
            n_blocks,
            valid_tail,
        ).launch(
            grid=[n_classify_tiles, n_blocks, 1],
            block=[self.tile_kv, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def stats_kernel(
        self,
        fn: cute.Tensor,
        noise: cute.Tensor,
        doc: cute.Tensor,
        f32_stats: cute.Tensor,
        i64_stats: cute.Tensor,
        n_blocks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()
        row_in_cta = tidx // self.threads_per_row
        lane = tidx % self.threads_per_row
        row_batch = bidx
        row = row_batch * self.rows_per_cta + row_in_cta

        while row < n_blocks:
            fn_row = fn[row, None]
            noise_row = noise[row, None]
            doc_row = doc[row, None]
            f32_offset = lane * _FP32_VEC_SIZE
            i64_offset = lane * (2 * _I64_VEC_SIZE)

            fn_0, fn_1, fn_2, fn_3 = ld_global_v4_f32(
                get_ptr_as_int64(fn_row, f32_offset)
            )
            noise_0, noise_1, noise_2, noise_3 = ld_global_v4_f32(
                get_ptr_as_int64(noise_row, f32_offset)
            )
            doc_0, doc_1 = ld_global_v2_u64(get_ptr_as_int64(doc_row, i64_offset))
            doc_2, doc_3 = ld_global_v2_u64(
                get_ptr_as_int64(doc_row, i64_offset + Int32(2))
            )

            fn_max, fn_min = max_min_fp32(fn_0, fn_1, fn_2, fn_3)
            noise_max, noise_min = max_min_fp32(noise_0, noise_1, noise_2, noise_3)
            doc_max, doc_min = max_min_i64(doc_0, doc_1, doc_2, doc_3)

            for chunk in cutlass.range_constexpr(1, self.col_chunks):
                chunk_f32 = f32_offset + chunk * self.threads_per_row * _FP32_VEC_SIZE
                chunk_i64 = (
                    i64_offset + chunk * self.threads_per_row * 2 * _I64_VEC_SIZE
                )
                c_fn_0, c_fn_1, c_fn_2, c_fn_3 = ld_global_v4_f32(
                    get_ptr_as_int64(fn_row, chunk_f32)
                )
                c_noise_0, c_noise_1, c_noise_2, c_noise_3 = ld_global_v4_f32(
                    get_ptr_as_int64(noise_row, chunk_f32)
                )
                c_doc_0, c_doc_1 = ld_global_v2_u64(
                    get_ptr_as_int64(doc_row, chunk_i64)
                )
                c_doc_2, c_doc_3 = ld_global_v2_u64(
                    get_ptr_as_int64(doc_row, chunk_i64 + Int32(2))
                )
                c_fn_max, c_fn_min = max_min_fp32(c_fn_0, c_fn_1, c_fn_2, c_fn_3)
                c_noise_max, c_noise_min = max_min_fp32(
                    c_noise_0, c_noise_1, c_noise_2, c_noise_3
                )
                c_doc_max, c_doc_min = max_min_i64(c_doc_0, c_doc_1, c_doc_2, c_doc_3)
                fn_max = cute.arch.fmax(fn_max, c_fn_max)
                fn_min = fmin_f32(fn_min, c_fn_min)
                noise_max = cute.arch.fmax(noise_max, c_noise_max)
                noise_min = fmin_f32(noise_min, c_noise_min)
                doc_max = imax_s64(doc_max, c_doc_max)
                doc_min = imin_s64(doc_min, c_doc_min)

            fn_max = warp_reduce(fn_max, cute.arch.fmax, width=self.threads_per_row)
            fn_min = warp_reduce(fn_min, fmin_f32, width=self.threads_per_row)
            noise_max = warp_reduce(
                noise_max, cute.arch.fmax, width=self.threads_per_row
            )
            noise_min = warp_reduce(noise_min, fmin_f32, width=self.threads_per_row)
            doc_max = warp_reduce(doc_max, imax_s64, width=self.threads_per_row)
            doc_min = warp_reduce(doc_min, imin_s64, width=self.threads_per_row)

            if lane == 0:
                f32_stats[0, row] = fn_min
                f32_stats[1, row] = fn_max
                f32_stats[2, row] = noise_min
                f32_stats[3, row] = noise_max
                i64_stats[0, row] = doc_min
                i64_stats[1, row] = doc_max

            row_batch = row_batch + grid_dim_x
            row = row_batch * self.rows_per_cta + row_in_cta

    @cute.kernel
    def bitmap_kernel(
        self,
        fn: cute.Tensor,
        noise: cute.Tensor,
        doc: cute.Tensor,
        f32_stats: cute.Tensor,
        i64_stats: cute.Tensor,
        bits: cute.Tensor,
        n_blocks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        fn_extent = _ext_words(f32_stats[1, row])
        noise_extent = _ext_words(f32_stats[3, row])
        doc_extent = _ext_words_i64(i64_stats[1, row])
        extent = Int32(select_(fn_extent > noise_extent, fn_extent, noise_extent))
        extent = Int32(select_(extent > doc_extent, extent, doc_extent))

        word = Int32(tidx)
        while word < extent:
            bits[0, word, row] = Int32(0)
            bits[1, word, row] = Int32(0)
            bits[2, word, row] = Int32(0)
            word = word + self.bitmap_threads
        cute.arch.sync_threads()

        noise_base = Int32(_W_CAP) * n_blocks
        doc_base = Int32(2 * _W_CAP) * n_blocks
        for element in cutlass.range_constexpr(self.bitmap_elements):
            token = tidx + element * self.bitmap_threads
            if token < self.block_size:
                fn_value = Float32(fn[row, token])
                if (fn_value >= 0.0) & (fn_value < _W_CAP * 32.0):
                    fn_id = Int32(fn_value)
                    cute.arch.atomic_or(
                        bits.iterator + ((fn_id >> 5) * n_blocks + row),
                        Int32(1) << (fn_id & 31),
                    )
                noise_value = Float32(noise[row, token])
                if (noise_value >= 0.0) & (noise_value < _W_CAP * 32.0):
                    noise_id = Int32(noise_value)
                    cute.arch.atomic_or(
                        bits.iterator + (noise_base + (noise_id >> 5) * n_blocks + row),
                        Int32(1) << (noise_id & 31),
                    )
                doc_value = Int64(doc[row, token])
                if (doc_value >= 0) & (doc_value < _W_CAP * 32):
                    doc_id = Int32(doc_value)
                    cute.arch.atomic_or(
                        bits.iterator + (doc_base + (doc_id >> 5) * n_blocks + row),
                        Int32(1) << (doc_id & 31),
                    )

    @cute.kernel
    def classify_kernel(
        self,
        fn: cute.Tensor,
        noise: cute.Tensor,
        doc: cute.Tensor,
        f32_stats: cute.Tensor,
        i64_stats: cute.Tensor,
        bits: cute.Tensor,
        block_type: cute.Tensor,
        n_blocks: Int32,
        valid_tail: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        tile, q_block, _ = cute.arch.block_idx()
        kv_block = tile * self.tile_kv + tidx

        if kv_block < n_blocks:
            q_fn_min = Float32(f32_stats[0, q_block])
            q_fn_max = Float32(f32_stats[1, q_block])
            q_noise_min = Float32(f32_stats[2, q_block])
            q_noise_max = Float32(f32_stats[3, q_block])
            q_doc_min = Int64(i64_stats[0, q_block])
            q_doc_max = Int64(i64_stats[1, q_block])

            kv_fn_min = Float32(f32_stats[0, kv_block])
            kv_fn_max = Float32(f32_stats[1, kv_block])
            kv_noise_min = Float32(f32_stats[2, kv_block])
            kv_noise_max = Float32(f32_stats[3, kv_block])
            kv_doc_min = Int64(i64_stats[0, kv_block])
            kv_doc_max = Int64(i64_stats[1, kv_block])

            same_doc = (
                (q_doc_min == q_doc_max)
                & (kv_doc_min == kv_doc_max)
                & (q_doc_min == kv_doc_min)
            )
            all_full_attn = (
                same_doc
                & (q_fn_min == q_fn_max)
                & (kv_fn_min == kv_fn_max)
                & (q_fn_min >= 0.0)
                & (q_fn_min == kv_fn_min)
            )
            q_fn_extent = _ext_words(q_fn_max)
            q_noise_extent = _ext_words(q_noise_max)
            q_doc_extent = _ext_words_i64(q_doc_max)
            kv_fn_extent = _ext_words(kv_fn_max)
            kv_noise_extent = _ext_words(kv_noise_max)
            kv_doc_extent = _ext_words_i64(kv_doc_max)
            extent = Int32(
                select_(
                    kv_fn_extent > kv_noise_extent,
                    kv_fn_extent,
                    kv_noise_extent,
                )
            )
            extent = Int32(select_(extent > kv_doc_extent, extent, kv_doc_extent))
            kv_extra_noise = Int32(0)
            shared_noise = Int32(0)
            shared_fn = Int32(0)
            shared_doc = Int32(0)
            word = Int32(0)
            while word < extent:
                kv_fn_word = bits[0, word, kv_block]
                kv_noise_word = bits[1, word, kv_block]
                kv_doc_word = bits[2, word, kv_block]
                q_fn_word = Int32(
                    select_(
                        word < q_fn_extent,
                        bits[0, word, q_block],
                        Int32(0),
                    )
                )
                q_noise_word = Int32(
                    select_(
                        word < q_noise_extent,
                        bits[1, word, q_block],
                        Int32(0),
                    )
                )
                q_doc_word = Int32(
                    select_(
                        word < q_doc_extent,
                        bits[2, word, q_block],
                        Int32(0),
                    )
                )
                kv_extra_noise = kv_extra_noise | (kv_noise_word & ~q_noise_word)
                shared_noise = shared_noise | (kv_noise_word & q_noise_word)
                shared_fn = shared_fn | (kv_fn_word & q_fn_word)
                shared_doc = shared_doc | (kv_doc_word & q_doc_word)
                word = word + 1

            remove_noise_none = (kv_noise_min >= 0.0) & (shared_noise == 0)
            no_full_noise_overlap = shared_fn == 0
            q_single_positive_noise = (q_noise_min == q_noise_max) & (
                q_noise_min >= 0.0
            )
            remove_noise_all = (kv_noise_max < 0.0) | (
                q_single_positive_noise & (kv_extra_noise == 0)
            )
            full_attn_safe = all_full_attn & remove_noise_all
            below_diagonal_full = (q_block > kv_block) & same_doc & remove_noise_all
            singleton_diagonal_full = (
                (q_block == kv_block) & (q_block == n_blocks - 1) & (valid_tail == 1)
            )
            is_full = full_attn_safe | below_diagonal_full | singleton_diagonal_full

            doc_ranges_disjoint = (q_doc_max < kv_doc_min) | (kv_doc_max < q_doc_min)
            above_diagonal = q_block < kv_block
            is_empty = (
                doc_ranges_disjoint
                | remove_noise_none
                | (above_diagonal & no_full_noise_overlap)
            )

            both_without_noise = (q_noise_max < 0.0) & (kv_noise_max < 0.0)
            if (q_block != kv_block) & both_without_noise:
                active_without_noise = Int32(
                    select_(above_diagonal, shared_fn, shared_doc)
                )
                is_empty = active_without_noise == 0

            # Bitmaps settle uniform-document blocks except the unusual case
            # where fn and noise overlap independently but not on the same token
            # pair. Mixed document boundary blocks also need the composite check.
            needs_composite_scan = (both_without_noise == 0) & (
                (same_doc == 0) | (above_diagonal & (all_full_attn == 0))
            )
            if (
                (q_block != kv_block)
                & (is_full == 0)
                & (is_empty == 0)
                & needs_composite_scan
            ):
                active = Int32(0)
                q_token = Int32(0)
                while (q_token < self.block_size) & (active == 0):
                    q_fn = Float32(fn[q_block, q_token])
                    q_noise = Float32(noise[q_block, q_token])
                    q_doc = Int64(doc[q_block, q_token])
                    kv_token = Int32(0)
                    while (kv_token < self.block_size) & (active == 0):
                        kv_fn = Float32(fn[kv_block, kv_token])
                        kv_noise = Float32(noise[kv_block, kv_token])
                        kv_doc = Int64(doc[kv_block, kv_token])
                        remove_noise = (kv_noise < 0.0) | (q_noise == kv_noise)
                        full_noise = (q_fn >= 0.0) & (q_fn == kv_fn)
                        attends = (
                            (q_doc == kv_doc)
                            & remove_noise
                            & ((q_block > kv_block) | full_noise)
                        )
                        active = Int32(select_(attends, Int32(1), active))
                        kv_token = kv_token + 1
                    q_token = q_token + 1
                is_empty = active == 0

            value = Int32(select_(is_empty, Int32(BLOCK_EMPTY), Int32(BLOCK_PARTIAL)))
            value = Int32(select_(is_full, Int32(BLOCK_FULL), value))
            block_type[q_block, kv_block] = value


class BcsrKernel:
    """Convert a square block-type grid to forward/backward BCSR tensors."""

    def __init__(self):
        self.warps_per_cta = _BCSR_WARPS_PER_CTA
        self.threads = 32 * self.warps_per_cta

    @cute.jit
    def __call__(
        self,
        block_type: cute.Tensor,
        kv_num: cute.Tensor,
        kv_idx: cute.Tensor,
        full_kv_num: cute.Tensor,
        full_kv_idx: cute.Tensor,
        q_num: cute.Tensor,
        q_idx: cute.Tensor,
        full_q_num: cute.Tensor,
        full_q_idx: cute.Tensor,
        n_blocks: Int32,
        num_ctas: Int32,
        stream,
    ):
        self.kernel(
            block_type,
            kv_num,
            kv_idx,
            full_kv_num,
            full_kv_idx,
            q_num,
            q_idx,
            full_q_num,
            full_q_idx,
            n_blocks,
        ).launch(
            grid=[num_ctas, 1, 1],
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.jit
    def _write_row(
        self,
        block_type: cute.Tensor,
        num_out: cute.Tensor,
        idx_out: cute.Tensor,
        row: Int32,
        n_blocks: Int32,
        target: int,
        lane: Int32,
        transpose: bool,
    ):
        count = Int32(0)
        base = Int32(0)
        while base < n_blocks:
            column = base + lane
            safe_column = Int32(select_(column < n_blocks, column, Int32(0)))
            if cutlass.const_expr(transpose):
                value = block_type[safe_column, row]
            else:
                value = block_type[row, safe_column]
            member = (column < n_blocks) & (value == target)
            count = count + cute.arch.popc(cute.arch.vote_ballot_sync(member))
            base = base + 32
        if lane == 0:
            num_out[row] = count

        lower_lane_mask = (Int32(1) << lane) - 1
        carry = Int32(0)
        base = Int32(0)
        while base < n_blocks:
            column = base + lane
            safe_column = Int32(select_(column < n_blocks, column, Int32(0)))
            if cutlass.const_expr(transpose):
                value = block_type[safe_column, row]
            else:
                value = block_type[row, safe_column]
            member = (column < n_blocks) & (value == target)
            ballot = cute.arch.vote_ballot_sync(member)
            members_before = carry + cute.arch.popc(ballot & lower_lane_mask)
            position = Int32(
                select_(member, members_before, count + (column - members_before))
            )
            if column < n_blocks:
                idx_out[row, position] = column
            carry = carry + cute.arch.popc(ballot)
            base = base + 32

    @cute.kernel
    def kernel(
        self,
        block_type: cute.Tensor,
        kv_num: cute.Tensor,
        kv_idx: cute.Tensor,
        full_kv_num: cute.Tensor,
        full_kv_idx: cute.Tensor,
        q_num: cute.Tensor,
        q_idx: cute.Tensor,
        full_q_num: cute.Tensor,
        full_q_idx: cute.Tensor,
        n_blocks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        workgroup = bidx * self.warps_per_cta + tidx // 32
        lane = cute.arch.lane_idx()

        if workgroup < n_blocks:
            self._write_row(
                block_type,
                kv_num,
                kv_idx,
                workgroup,
                n_blocks,
                BLOCK_PARTIAL,
                lane,
                False,
            )
        elif workgroup < 2 * n_blocks:
            self._write_row(
                block_type,
                full_kv_num,
                full_kv_idx,
                workgroup - n_blocks,
                n_blocks,
                BLOCK_FULL,
                lane,
                False,
            )
        elif workgroup < 3 * n_blocks:
            self._write_row(
                block_type,
                q_num,
                q_idx,
                workgroup - 2 * n_blocks,
                n_blocks,
                BLOCK_PARTIAL,
                lane,
                True,
            )
        elif workgroup < 4 * n_blocks:
            self._write_row(
                block_type,
                full_q_num,
                full_q_idx,
                workgroup - 3 * n_blocks,
                n_blocks,
                BLOCK_FULL,
                lane,
                True,
            )


@functools.cache
def _get_compiled_classify_kernel(block_size: int) -> tuple[Callable, int, int]:
    kernel = SelfAttentionClassifyKernel(block_size)
    n_blocks = cute.sym_int()

    def _f32_blocked():
        return cute.runtime.make_fake_compact_tensor(
            Float32, (n_blocks, block_size), stride_order=(1, 0), assumed_align=4
        )

    fn = _f32_blocked()
    noise = _f32_blocked()
    doc = cute.runtime.make_fake_compact_tensor(
        Int64, (n_blocks, block_size), stride_order=(1, 0), assumed_align=8
    )
    f32_stats = cute.runtime.make_fake_compact_tensor(
        Float32, (4, n_blocks), stride_order=(1, 0), assumed_align=4
    )
    i64_stats = cute.runtime.make_fake_compact_tensor(
        Int64, (2, n_blocks), stride_order=(1, 0), assumed_align=8
    )
    bits = cute.runtime.make_fake_compact_tensor(
        Int32,
        (3, _W_CAP, n_blocks),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    block_type = cute.runtime.make_fake_compact_tensor(
        Int32, (n_blocks, n_blocks), stride_order=(1, 0), assumed_align=4
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        kernel,
        fn,
        noise,
        doc,
        f32_stats,
        i64_stats,
        bits,
        block_type,
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        stream,
        options="--enable-tvm-ffi",
    )
    return compiled, kernel.rows_per_cta, kernel.tile_kv


@functools.cache
def _get_compiled_bcsr_kernel() -> tuple[Callable, int]:
    kernel = BcsrKernel()
    n_blocks = cute.sym_int()

    def _i32_fake(*shape):
        return cute.runtime.make_fake_compact_tensor(
            Int32,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=4,
        )

    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        kernel,
        _i32_fake(n_blocks, n_blocks),
        _i32_fake(n_blocks),
        _i32_fake(n_blocks, n_blocks),
        _i32_fake(n_blocks),
        _i32_fake(n_blocks, n_blocks),
        _i32_fake(n_blocks),
        _i32_fake(n_blocks, n_blocks),
        _i32_fake(n_blocks),
        _i32_fake(n_blocks, n_blocks),
        Int32(1),
        Int32(1),
        stream,
        options="--enable-tvm-ffi",
    )
    return compiled, kernel.warps_per_cta


def classify_self_attention_blocks_cute_dsl(
    full_and_noise_seq_id: torch.Tensor,
    noise_seq_id: torch.Tensor,
    document_id: torch.Tensor,
    *,
    block_size: int = 128,
) -> SelfAttentionBlockSparseResult:
    """Classify an FA4 self-attention mask with exact CuTe DSL kernels.

    Segment, noise, and document IDs must be integral values smaller than
    ``BITMAP_ID_LIMIT`` (16384), and document IDs non-negative, matching the
    production split-index representation. The ID range is checked with one
    host sync.

    Args:
        full_and_noise_seq_id: Contiguous CUDA float32 segment IDs.
        noise_seq_id: Contiguous CUDA float32 noise IDs.
        document_id: Contiguous CUDA int64 document IDs.
        block_size: Positive power-of-two square block size.

    Returns:
        Exact block types and partial/full BCSR tensors for both directions.

    Raises:
        BitmapIdRangeError: If an ID does not fit the per-block bitmaps.
        ValueError: If the kernel input contract is not satisfied.
    """
    tensors = (full_and_noise_seq_id, noise_seq_id, document_id)
    if block_size < 4 or block_size & (block_size - 1):
        raise ValueError(
            f"block_size must be a power of two at least 4, got {block_size=}."
        )
    if block_size > 1024:
        raise ValueError(f"block_size must be at most 1024, got {block_size=}.")
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError(
            "FA4 self-attention aux tensors must be one-dimensional, got "
            f"{[tuple(tensor.shape) for tensor in tensors]}."
        )
    if len({tensor.numel() for tensor in tensors}) != 1 or not tensors[0].numel():
        raise ValueError(
            "FA4 self-attention aux tensors must have equal nonzero lengths."
        )
    if (
        full_and_noise_seq_id.dtype != torch.float32
        or noise_seq_id.dtype != torch.float32
    ):
        raise ValueError("FA4 full/noise and noise IDs must be float32 CUDA tensors.")
    if document_id.dtype != torch.int64:
        raise ValueError("FA4 document IDs must be int64 CUDA tensors.")
    device = full_and_noise_seq_id.device
    if any(not tensor.is_cuda or tensor.device != device for tensor in tensors):
        raise ValueError(
            f"FA4 self-attention aux tensors must share one CUDA device, got {device}."
        )
    _check_ids_fit_bitmaps(full_and_noise_seq_id, noise_seq_id, document_id)

    seq_len = full_and_noise_seq_id.numel()
    fn = _pad_to_blocks(full_and_noise_seq_id.contiguous(), block_size)
    noise = _pad_to_blocks(noise_seq_id.contiguous(), block_size)
    doc = _pad_to_blocks(document_id.contiguous(), block_size)
    n_blocks = fn.numel() // block_size
    fn = fn.reshape(n_blocks, block_size)
    noise = noise.reshape(n_blocks, block_size)
    doc = doc.reshape(n_blocks, block_size)

    f32_stats = torch.empty((4, n_blocks), dtype=torch.float32, device=device)
    i64_stats = torch.empty((2, n_blocks), dtype=torch.int64, device=device)
    bits = torch.empty((3, _W_CAP, n_blocks), dtype=torch.int32, device=device)
    block_type = torch.empty((n_blocks, n_blocks), dtype=torch.int32, device=device)
    classify, rows_per_cta, tile_kv = _get_compiled_classify_kernel(block_size)
    classify(
        fn,
        noise,
        doc,
        f32_stats,
        i64_stats,
        bits,
        block_type,
        n_blocks,
        (n_blocks + rows_per_cta - 1) // rows_per_cta,
        (n_blocks + tile_kv - 1) // tile_kv,
        seq_len - (n_blocks - 1) * block_size,
    )

    shape_1d = (1, 1, n_blocks)
    shape_2d = (1, 1, n_blocks, n_blocks)
    kv_num_blocks = torch.empty(shape_1d, dtype=torch.int32, device=device)
    kv_indices = torch.empty(shape_2d, dtype=torch.int32, device=device)
    full_kv_num_blocks = torch.empty(shape_1d, dtype=torch.int32, device=device)
    full_kv_indices = torch.empty(shape_2d, dtype=torch.int32, device=device)
    q_num_blocks = torch.empty(shape_1d, dtype=torch.int32, device=device)
    q_indices = torch.empty(shape_2d, dtype=torch.int32, device=device)
    full_q_num_blocks = torch.empty(shape_1d, dtype=torch.int32, device=device)
    full_q_indices = torch.empty(shape_2d, dtype=torch.int32, device=device)

    bcsr, warps_per_cta = _get_compiled_bcsr_kernel()
    bcsr(
        block_type,
        kv_num_blocks.view(n_blocks),
        kv_indices.view(n_blocks, n_blocks),
        full_kv_num_blocks.view(n_blocks),
        full_kv_indices.view(n_blocks, n_blocks),
        q_num_blocks.view(n_blocks),
        q_indices.view(n_blocks, n_blocks),
        full_q_num_blocks.view(n_blocks),
        full_q_indices.view(n_blocks, n_blocks),
        n_blocks,
        (4 * n_blocks + warps_per_cta - 1) // warps_per_cta,
    )
    return SelfAttentionBlockSparseResult(
        n_blocks=n_blocks,
        block_size=(block_size, block_size),
        block_type=block_type,
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=full_kv_num_blocks,
        full_kv_indices=full_kv_indices,
        q_num_blocks=q_num_blocks,
        q_indices=q_indices,
        full_q_num_blocks=full_q_num_blocks,
        full_q_indices=full_q_indices,
    )
