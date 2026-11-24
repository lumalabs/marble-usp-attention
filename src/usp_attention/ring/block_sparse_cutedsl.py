# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, select_
from torch import Tensor

from usp_attention.fa4.cute_dsl_utils import (
    device_support_pdl,
    fmin_f32,
    get_num_sm,
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
    _dense_to_ordered,
    _transpose_ordered,
)
from usp_attention.ring.block_sparse import (
    BLOCK_EMPTY,
    BLOCK_FULL,
    BLOCK_PARTIAL,
    BlockSparseResult,
    _pad_to_blocks,
    build_block_sparse_from_cross_qkv_aux,
)


@dataclass(kw_only=True, slots=True)
class BlockStats:
    """Per-block min/max stats for the 8 aux tensors.

    Each field is a 1D tensor of length ``n_q_blocks`` (``q_*``) or
    ``n_kv_blocks`` (``kv_*``); fn/noise stats are float32, doc/pos stats
    are int64, matching the aux dtypes.
    """

    q_doc_min: Tensor
    q_doc_max: Tensor
    kv_doc_min: Tensor
    kv_doc_max: Tensor
    q_fn_min: Tensor
    q_fn_max: Tensor
    kv_fn_min: Tensor
    kv_fn_max: Tensor
    q_noise_min: Tensor
    q_noise_max: Tensor
    kv_noise_min: Tensor
    kv_noise_max: Tensor
    q_pos_min: Tensor
    q_pos_max: Tensor
    kv_pos_min: Tensor
    kv_pos_max: Tensor


_AUX_NAMES = (
    "q_fn",
    "kv_fn",
    "q_noise",
    "kv_noise",
    "q_doc",
    "kv_doc",
    "q_pos",
    "kv_pos",
)
_AUX_DTYPES = (
    torch.float32,
    torch.float32,
    torch.float32,
    torch.float32,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
)

_FP32_VEC_SIZE = 4
_I64_VEC_SIZE = 2

# Blocks per SM for occupancy target
_NUM_SM_BLOCKS = 4

# Maximum threads per block
_MAX_THREADS_PER_BLOCK = 1024

# Thread count bounds for swizzled kernel
_MIN_THREADS = 128
_MAX_THREADS = 256


def _compute_optimal_threads(K: int) -> tuple[int, int]:
    threads_per_row = K // _FP32_VEC_SIZE

    # Snap threads_per_row down to the largest allowed power of two (max 32)
    allowed = (1, 2, 4, 8, 16, 32)
    tpr = 1
    for p in allowed:
        if p <= threads_per_row:
            tpr = p
        else:
            break

    # Largest multiple of tpr in [_MIN_THREADS, _MAX_THREADS]
    largest = (_MAX_THREADS // tpr) * tpr
    if largest >= _MIN_THREADS:
        return largest, tpr

    candidate = ((_MIN_THREADS + tpr - 1) // tpr) * tpr
    if candidate <= _MAX_THREADS:
        return candidate, tpr

    return _MAX_THREADS, tpr


class BlockStatsKernel:
    def __init__(
        self,
        block_size: int,
        enable_pdl: bool = False,
    ):
        self.block_size = block_size
        self.enable_pdl = enable_pdl

        self.num_threads, self.threads_per_row = _compute_optimal_threads(block_size)
        # threads_per_row is capped at 32 (shuffle width), so wide blocks need
        # each thread to fold multiple 4-element chunks before the reduce.
        self.col_chunks = block_size // (self.threads_per_row * _FP32_VEC_SIZE)

        if self.threads_per_row <= self.num_threads:
            self.rows_per_block = self.num_threads // self.threads_per_row
            self.needs_col_loop = False
        else:
            self.rows_per_block = 1
            self.needs_col_loop = True

    @cute.jit
    def __call__(
        self,
        q_fn: cute.Tensor,
        kv_fn: cute.Tensor,
        q_noise: cute.Tensor,
        kv_noise: cute.Tensor,
        q_doc: cute.Tensor,
        kv_doc: cute.Tensor,
        q_pos: cute.Tensor,
        kv_pos: cute.Tensor,
        q_f32_out: cute.Tensor,  # (4, n_q_blocks): fn_min, fn_max, noise_min, noise_max
        kv_f32_out: cute.Tensor,  # (4, n_kv_blocks): same rows
        q_i64_out: cute.Tensor,  # (4, n_q_blocks): doc_min, doc_max, pos_min, pos_max
        kv_i64_out: cute.Tensor,  # (4, n_kv_blocks): same rows
        num_blocks: Int32,
        q_len: Int32,
        kv_len: Int32,
        stream,
    ):
        self.kernel(
            q_fn,
            kv_fn,
            q_noise,
            kv_noise,
            q_doc,
            kv_doc,
            q_pos,
            kv_pos,
            q_f32_out,
            kv_f32_out,
            q_i64_out,
            kv_i64_out,
            q_len,
            kv_len,
        ).launch(
            grid=[num_blocks, 1, 1],
            block=[self.num_threads, 1, 1],
            max_number_threads=[_MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=_NUM_SM_BLOCKS,
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        q_fn: cute.Tensor,
        kv_fn: cute.Tensor,
        q_noise: cute.Tensor,
        kv_noise: cute.Tensor,
        q_doc: cute.Tensor,
        kv_doc: cute.Tensor,
        q_pos: cute.Tensor,
        kv_pos: cute.Tensor,
        q_f32_out: cute.Tensor,  # (4, n_q_blocks): fn_min, fn_max, noise_min, noise_max
        kv_f32_out: cute.Tensor,  # (4, n_kv_blocks): same rows
        q_i64_out: cute.Tensor,  # (4, n_q_blocks): doc_min, doc_max, pos_min, pos_max
        kv_i64_out: cute.Tensor,  # (4, n_kv_blocks): same rows
        q_len: Int32,
        kv_len: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        threads_per_row = self.threads_per_row
        rows_per_block = self.rows_per_block

        if cutlass.const_expr(self.needs_col_loop):
            pass
        else:
            row_in_block = tidx // threads_per_row
            lane_idx_in_row = tidx % threads_per_row

            row_batch_idx = bidx
            row_idx = row_batch_idx * rows_per_block + row_in_block

            while row_idx < q_len + kv_len:
                # Each thread consumes 4 elements per row for both dtypes:
                # one v4.f32 load, or two v2.u64 loads at +0 and +2.
                elem_base_fp32 = lane_idx_in_row * _FP32_VEC_SIZE
                elem_base_i64 = lane_idx_in_row * (2 * _I64_VEC_SIZE)

                if row_idx < q_len:
                    q_fn_row_input = q_fn[row_idx, None]
                    q_noise_row_input = q_noise[row_idx, None]
                    q_doc_row_input = q_doc[row_idx, None]
                    q_pos_row_input = q_pos[row_idx, None]

                    q_fn_ptr = get_ptr_as_int64(q_fn_row_input, elem_base_fp32)
                    q_noise_ptr = get_ptr_as_int64(q_noise_row_input, elem_base_fp32)
                    q_doc_ptr_0 = get_ptr_as_int64(q_doc_row_input, elem_base_i64)
                    q_doc_ptr_1 = get_ptr_as_int64(
                        q_doc_row_input, elem_base_i64 + Int32(2)
                    )
                    q_pos_ptr_0 = get_ptr_as_int64(q_pos_row_input, elem_base_i64)
                    q_pos_ptr_1 = get_ptr_as_int64(
                        q_pos_row_input, elem_base_i64 + Int32(2)
                    )

                    q_fn_0, q_fn_1, q_fn_2, q_fn_3 = ld_global_v4_f32(q_fn_ptr)
                    q_noise_0, q_noise_1, q_noise_2, q_noise_3 = ld_global_v4_f32(
                        q_noise_ptr
                    )
                    q_doc_0, q_doc_1 = ld_global_v2_u64(q_doc_ptr_0)
                    q_doc_2, q_doc_3 = ld_global_v2_u64(q_doc_ptr_1)
                    q_pos_0, q_pos_1 = ld_global_v2_u64(q_pos_ptr_0)
                    q_pos_2, q_pos_3 = ld_global_v2_u64(q_pos_ptr_1)

                    # Note LIP the max/mins and shf later
                    q_fn_max_t, q_fn_min_t = max_min_fp32(
                        q_fn_0, q_fn_1, q_fn_2, q_fn_3
                    )
                    q_noise_max_t, q_noise_min_t = max_min_fp32(
                        q_noise_0, q_noise_1, q_noise_2, q_noise_3
                    )
                    q_doc_max_t, q_doc_min_t = max_min_i64(
                        q_doc_0, q_doc_1, q_doc_2, q_doc_3
                    )
                    q_pos_max_t, q_pos_min_t = max_min_i64(
                        q_pos_0, q_pos_1, q_pos_2, q_pos_3
                    )

                    for c in cutlass.range_constexpr(1, self.col_chunks):
                        off_fp32 = elem_base_fp32 + c * threads_per_row * _FP32_VEC_SIZE
                        off_i64 = elem_base_i64 + c * threads_per_row * (
                            2 * _I64_VEC_SIZE
                        )
                        c_fn_0, c_fn_1, c_fn_2, c_fn_3 = ld_global_v4_f32(
                            get_ptr_as_int64(q_fn_row_input, off_fp32)
                        )
                        c_noise_0, c_noise_1, c_noise_2, c_noise_3 = ld_global_v4_f32(
                            get_ptr_as_int64(q_noise_row_input, off_fp32)
                        )
                        c_doc_0, c_doc_1 = ld_global_v2_u64(
                            get_ptr_as_int64(q_doc_row_input, off_i64)
                        )
                        c_doc_2, c_doc_3 = ld_global_v2_u64(
                            get_ptr_as_int64(q_doc_row_input, off_i64 + Int32(2))
                        )
                        c_pos_0, c_pos_1 = ld_global_v2_u64(
                            get_ptr_as_int64(q_pos_row_input, off_i64)
                        )
                        c_pos_2, c_pos_3 = ld_global_v2_u64(
                            get_ptr_as_int64(q_pos_row_input, off_i64 + Int32(2))
                        )
                        c_fn_max, c_fn_min = max_min_fp32(
                            c_fn_0, c_fn_1, c_fn_2, c_fn_3
                        )
                        c_noise_max, c_noise_min = max_min_fp32(
                            c_noise_0, c_noise_1, c_noise_2, c_noise_3
                        )
                        c_doc_max, c_doc_min = max_min_i64(
                            c_doc_0, c_doc_1, c_doc_2, c_doc_3
                        )
                        c_pos_max, c_pos_min = max_min_i64(
                            c_pos_0, c_pos_1, c_pos_2, c_pos_3
                        )
                        q_fn_max_t = cute.arch.fmax(q_fn_max_t, c_fn_max)
                        q_fn_min_t = fmin_f32(q_fn_min_t, c_fn_min)
                        q_noise_max_t = cute.arch.fmax(q_noise_max_t, c_noise_max)
                        q_noise_min_t = fmin_f32(q_noise_min_t, c_noise_min)
                        q_doc_max_t = imax_s64(q_doc_max_t, c_doc_max)
                        q_doc_min_t = imin_s64(q_doc_min_t, c_doc_min)
                        q_pos_max_t = imax_s64(q_pos_max_t, c_pos_max)
                        q_pos_min_t = imin_s64(q_pos_min_t, c_pos_min)

                    q_fn_max_row = warp_reduce(
                        q_fn_max_t, cute.arch.fmax, width=threads_per_row
                    )
                    q_fn_min_row = warp_reduce(
                        q_fn_min_t, fmin_f32, width=threads_per_row
                    )
                    q_noise_max_row = warp_reduce(
                        q_noise_max_t, cute.arch.fmax, width=threads_per_row
                    )
                    q_noise_min_row = warp_reduce(
                        q_noise_min_t, fmin_f32, width=threads_per_row
                    )

                    q_doc_max_row = warp_reduce(
                        q_doc_max_t, imax_s64, width=threads_per_row
                    )
                    q_doc_min_row = warp_reduce(
                        q_doc_min_t, imin_s64, width=threads_per_row
                    )
                    q_pos_max_row = warp_reduce(
                        q_pos_max_t, imax_s64, width=threads_per_row
                    )
                    q_pos_min_row = warp_reduce(
                        q_pos_min_t, imin_s64, width=threads_per_row
                    )

                    if lane_idx_in_row == 0:
                        q_f32_out[0, row_idx] = q_fn_min_row
                        q_f32_out[1, row_idx] = q_fn_max_row
                        q_f32_out[2, row_idx] = q_noise_min_row
                        q_f32_out[3, row_idx] = q_noise_max_row

                        q_i64_out[0, row_idx] = q_doc_min_row
                        q_i64_out[1, row_idx] = q_doc_max_row
                        q_i64_out[2, row_idx] = q_pos_min_row
                        q_i64_out[3, row_idx] = q_pos_max_row

                elif row_idx < q_len + kv_len:
                    kv_row_idx = row_idx - q_len
                    kv_fn_row_input = kv_fn[kv_row_idx, None]
                    kv_noise_row_input = kv_noise[kv_row_idx, None]
                    kv_doc_row_input = kv_doc[kv_row_idx, None]
                    kv_pos_row_input = kv_pos[kv_row_idx, None]

                    kv_fn_ptr = get_ptr_as_int64(kv_fn_row_input, elem_base_fp32)
                    kv_noise_ptr = get_ptr_as_int64(kv_noise_row_input, elem_base_fp32)
                    kv_doc_ptr_0 = get_ptr_as_int64(kv_doc_row_input, elem_base_i64)
                    kv_doc_ptr_1 = get_ptr_as_int64(
                        kv_doc_row_input, elem_base_i64 + Int32(2)
                    )
                    kv_pos_ptr_0 = get_ptr_as_int64(kv_pos_row_input, elem_base_i64)
                    kv_pos_ptr_1 = get_ptr_as_int64(
                        kv_pos_row_input, elem_base_i64 + Int32(2)
                    )

                    kv_fn_0, kv_fn_1, kv_fn_2, kv_fn_3 = ld_global_v4_f32(kv_fn_ptr)
                    kv_noise_0, kv_noise_1, kv_noise_2, kv_noise_3 = ld_global_v4_f32(
                        kv_noise_ptr
                    )
                    kv_doc_0, kv_doc_1 = ld_global_v2_u64(kv_doc_ptr_0)
                    kv_doc_2, kv_doc_3 = ld_global_v2_u64(kv_doc_ptr_1)
                    kv_pos_0, kv_pos_1 = ld_global_v2_u64(kv_pos_ptr_0)
                    kv_pos_2, kv_pos_3 = ld_global_v2_u64(kv_pos_ptr_1)

                    # Note: Increase ILP max/mins and shf
                    kv_fn_max_t, kv_fn_min_t = max_min_fp32(
                        kv_fn_0, kv_fn_1, kv_fn_2, kv_fn_3
                    )
                    kv_noise_max_t, kv_noise_min_t = max_min_fp32(
                        kv_noise_0, kv_noise_1, kv_noise_2, kv_noise_3
                    )
                    kv_doc_max_t, kv_doc_min_t = max_min_i64(
                        kv_doc_0, kv_doc_1, kv_doc_2, kv_doc_3
                    )
                    kv_pos_max_t, kv_pos_min_t = max_min_i64(
                        kv_pos_0, kv_pos_1, kv_pos_2, kv_pos_3
                    )

                    for c in cutlass.range_constexpr(1, self.col_chunks):
                        off_fp32 = elem_base_fp32 + c * threads_per_row * _FP32_VEC_SIZE
                        off_i64 = elem_base_i64 + c * threads_per_row * (
                            2 * _I64_VEC_SIZE
                        )
                        c_fn_0, c_fn_1, c_fn_2, c_fn_3 = ld_global_v4_f32(
                            get_ptr_as_int64(kv_fn_row_input, off_fp32)
                        )
                        c_noise_0, c_noise_1, c_noise_2, c_noise_3 = ld_global_v4_f32(
                            get_ptr_as_int64(kv_noise_row_input, off_fp32)
                        )
                        c_doc_0, c_doc_1 = ld_global_v2_u64(
                            get_ptr_as_int64(kv_doc_row_input, off_i64)
                        )
                        c_doc_2, c_doc_3 = ld_global_v2_u64(
                            get_ptr_as_int64(kv_doc_row_input, off_i64 + Int32(2))
                        )
                        c_pos_0, c_pos_1 = ld_global_v2_u64(
                            get_ptr_as_int64(kv_pos_row_input, off_i64)
                        )
                        c_pos_2, c_pos_3 = ld_global_v2_u64(
                            get_ptr_as_int64(kv_pos_row_input, off_i64 + Int32(2))
                        )
                        c_fn_max, c_fn_min = max_min_fp32(
                            c_fn_0, c_fn_1, c_fn_2, c_fn_3
                        )
                        c_noise_max, c_noise_min = max_min_fp32(
                            c_noise_0, c_noise_1, c_noise_2, c_noise_3
                        )
                        c_doc_max, c_doc_min = max_min_i64(
                            c_doc_0, c_doc_1, c_doc_2, c_doc_3
                        )
                        c_pos_max, c_pos_min = max_min_i64(
                            c_pos_0, c_pos_1, c_pos_2, c_pos_3
                        )
                        kv_fn_max_t = cute.arch.fmax(kv_fn_max_t, c_fn_max)
                        kv_fn_min_t = fmin_f32(kv_fn_min_t, c_fn_min)
                        kv_noise_max_t = cute.arch.fmax(kv_noise_max_t, c_noise_max)
                        kv_noise_min_t = fmin_f32(kv_noise_min_t, c_noise_min)
                        kv_doc_max_t = imax_s64(kv_doc_max_t, c_doc_max)
                        kv_doc_min_t = imin_s64(kv_doc_min_t, c_doc_min)
                        kv_pos_max_t = imax_s64(kv_pos_max_t, c_pos_max)
                        kv_pos_min_t = imin_s64(kv_pos_min_t, c_pos_min)

                    kv_fn_max_row = warp_reduce(
                        kv_fn_max_t, cute.arch.fmax, width=threads_per_row
                    )
                    kv_fn_min_row = warp_reduce(
                        kv_fn_min_t, fmin_f32, width=threads_per_row
                    )
                    kv_noise_max_row = warp_reduce(
                        kv_noise_max_t, cute.arch.fmax, width=threads_per_row
                    )
                    kv_noise_min_row = warp_reduce(
                        kv_noise_min_t, fmin_f32, width=threads_per_row
                    )

                    kv_doc_max_row = warp_reduce(
                        kv_doc_max_t, imax_s64, width=threads_per_row
                    )
                    kv_doc_min_row = warp_reduce(
                        kv_doc_min_t, imin_s64, width=threads_per_row
                    )
                    kv_pos_max_row = warp_reduce(
                        kv_pos_max_t, imax_s64, width=threads_per_row
                    )
                    kv_pos_min_row = warp_reduce(
                        kv_pos_min_t, imin_s64, width=threads_per_row
                    )

                    if lane_idx_in_row == 0:
                        kv_f32_out[0, kv_row_idx] = kv_fn_min_row
                        kv_f32_out[1, kv_row_idx] = kv_fn_max_row
                        kv_f32_out[2, kv_row_idx] = kv_noise_min_row
                        kv_f32_out[3, kv_row_idx] = kv_noise_max_row

                        kv_i64_out[0, kv_row_idx] = kv_doc_min_row
                        kv_i64_out[1, kv_row_idx] = kv_doc_max_row
                        kv_i64_out[2, kv_row_idx] = kv_pos_min_row
                        kv_i64_out[3, kv_row_idx] = kv_pos_max_row

                row_batch_idx = row_batch_idx + grid_dim_x
                row_idx = row_batch_idx * rows_per_block + row_in_block


@functools.cache
def _get_compiled_kernel(
    block_size: int = 128, enable_pdl: bool = False
) -> tuple[Callable, int]:
    kernel_obj = BlockStatsKernel(block_size, enable_pdl)

    sym_m_q = cute.sym_int()
    sym_m_kv = cute.sym_int()
    sym_nq = cute.sym_int()
    sym_nkv = cute.sym_int()

    q_fn_fake = cute.runtime.make_fake_compact_tensor(
        Float32, (sym_m_q, block_size), stride_order=(1, 0), assumed_align=16
    )
    kv_fn_fake = cute.runtime.make_fake_compact_tensor(
        Float32, (sym_m_kv, block_size), stride_order=(1, 0), assumed_align=16
    )

    q_noise_fake = cute.runtime.make_fake_compact_tensor(
        Float32, (sym_m_q, block_size), stride_order=(1, 0), assumed_align=16
    )
    kv_noise_fake = cute.runtime.make_fake_compact_tensor(
        Float32, (sym_m_kv, block_size), stride_order=(1, 0), assumed_align=16
    )

    q_doc_fake = cute.runtime.make_fake_compact_tensor(
        Int64, (sym_m_q, block_size), stride_order=(1, 0), assumed_align=32
    )
    kv_doc_fake = cute.runtime.make_fake_compact_tensor(
        Int64, (sym_m_kv, block_size), stride_order=(1, 0), assumed_align=32
    )

    q_pos_fake = cute.runtime.make_fake_compact_tensor(
        Int64, (sym_m_q, block_size), stride_order=(1, 0), assumed_align=32
    )
    kv_pos_fake = cute.runtime.make_fake_compact_tensor(
        Int64, (sym_m_kv, block_size), stride_order=(1, 0), assumed_align=32
    )

    q_f32_fake = cute.runtime.make_fake_compact_tensor(
        Float32, (4, sym_nq), stride_order=(1, 0), assumed_align=16
    )
    kv_f32_fake = cute.runtime.make_fake_compact_tensor(
        Float32, (4, sym_nkv), stride_order=(1, 0), assumed_align=16
    )

    q_i64_fake = cute.runtime.make_fake_compact_tensor(
        Int64, (4, sym_nq), stride_order=(1, 0), assumed_align=32
    )
    kv_i64_fake = cute.runtime.make_fake_compact_tensor(
        Int64, (4, sym_nkv), stride_order=(1, 0), assumed_align=32
    )

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        q_fn_fake,
        kv_fn_fake,
        q_noise_fake,
        kv_noise_fake,
        q_doc_fake,
        kv_doc_fake,
        q_pos_fake,
        kv_pos_fake,
        q_f32_fake,
        kv_f32_fake,
        q_i64_fake,
        kv_i64_fake,
        Int32(1),  # Dummy num_blocks
        Int32(1),  # Dummy q_len
        Int32(1),  # Dummy kv_len
        stream_fake,
        options="--enable-tvm-ffi",
    )

    return compiled_kernel, kernel_obj.rows_per_block


def compute_block_stats_cutedsl(
    aux_tensors: list[Tensor], block_size: int = 128, enable_pdl: bool = False
) -> BlockStats:
    if len(aux_tensors) != 8:
        raise ValueError(
            f"Expected 8 aux tensors for cross q/kv mask, got {len(aux_tensors)=}."
        )
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError(
            f"block_size must be a positive power of two, got {block_size=}."
        )
    device = aux_tensors[0].device
    for name, dtype, t in zip(_AUX_NAMES, _AUX_DTYPES, aux_tensors, strict=True):
        if t.dtype != dtype or t.dim() != 1 or not t.is_contiguous():
            raise ValueError(
                f"{name} must be a contiguous 1D {dtype} tensor, got "
                f"{t.dtype=}, {t.dim()=}, {t.is_contiguous()=}."
            )
        if not t.is_cuda or t.device != device:
            raise ValueError(f"{name} must be on {device}, got {t.device}.")

    q_fn, kv_fn, q_noise, kv_noise, q_doc, kv_doc, q_pos, kv_pos = aux_tensors

    if not (q_noise.numel() == q_doc.numel() == q_pos.numel()):
        raise ValueError("q-side aux tensors must all have the same length.")
    if not (kv_noise.numel() == kv_doc.numel() == kv_pos.numel()):
        raise ValueError("kv-side aux tensors must all have the same length.")

    if q_fn.dim() < 2:
        n_q_blocks = (q_fn.numel() + block_size - 1) // block_size
        q_fn = q_fn.reshape(n_q_blocks, block_size)
        q_noise = q_noise.reshape(n_q_blocks, block_size)
        q_doc = q_doc.reshape(n_q_blocks, block_size)
        q_pos = q_pos.reshape(n_q_blocks, block_size)

        n_kv_blocks = (kv_fn.numel() + block_size - 1) // block_size
        kv_fn = kv_fn.reshape(n_kv_blocks, block_size)
        kv_noise = kv_noise.reshape(n_kv_blocks, block_size)
        kv_doc = kv_doc.reshape(n_kv_blocks, block_size)
        kv_pos = kv_pos.reshape(n_kv_blocks, block_size)
    else:
        n_q_blocks = q_fn.shape[0]
        n_kv_blocks = kv_fn.shape[0]

    enable_pdl = device_support_pdl(device) if enable_pdl is not False else False
    target_grid = _NUM_SM_BLOCKS * get_num_sm(device)

    kernel_fn, rows_per_block = _get_compiled_kernel(block_size, enable_pdl)

    num_blocks = min(
        (n_q_blocks + n_kv_blocks + rows_per_block - 1) // rows_per_block, target_grid
    )

    q_f32 = torch.empty((4, n_q_blocks), dtype=torch.float32, device=device)
    kv_f32 = torch.empty((4, n_kv_blocks), dtype=torch.float32, device=device)
    q_i64 = torch.empty((4, n_q_blocks), dtype=torch.int64, device=device)
    kv_i64 = torch.empty((4, n_kv_blocks), dtype=torch.int64, device=device)

    kernel_fn(
        q_fn,
        kv_fn,
        q_noise,
        kv_noise,
        q_doc,
        kv_doc,
        q_pos,
        kv_pos,
        q_f32,
        kv_f32,
        q_i64,
        kv_i64,
        num_blocks,
        n_q_blocks,
        n_kv_blocks,
    )

    q_fn_min, q_fn_max, q_noise_min, q_noise_max = q_f32.unbind(0)
    kv_fn_min, kv_fn_max, kv_noise_min, kv_noise_max = kv_f32.unbind(0)
    q_doc_min, q_doc_max, q_pos_min, q_pos_max = q_i64.unbind(0)
    kv_doc_min, kv_doc_max, kv_pos_min, kv_pos_max = kv_i64.unbind(0)
    return BlockStats(
        q_doc_min=q_doc_min,
        q_doc_max=q_doc_max,
        kv_doc_min=kv_doc_min,
        kv_doc_max=kv_doc_max,
        q_fn_min=q_fn_min,
        q_fn_max=q_fn_max,
        kv_fn_min=kv_fn_min,
        kv_fn_max=kv_fn_max,
        q_noise_min=q_noise_min,
        q_noise_max=q_noise_max,
        kv_noise_min=kv_noise_min,
        kv_noise_max=kv_noise_max,
        q_pos_min=q_pos_min,
        q_pos_max=q_pos_max,
        kv_pos_min=kv_pos_min,
        kv_pos_max=kv_pos_max,
    )


# Bitmap words per (side, block): 512 words = 16384 ids.
_W_CAP = 512
_CLASSIFY_TILE_KV = 128


@cute.jit
def _ext_words(max_val: Float32) -> Int32:
    # Words needed for a block's id bitmap, from its (integral) fn/noise max.
    raw = (Int32(max_val) >> 5) + 1
    ext = Int32(select_(max_val >= _W_CAP * 32.0, Int32(_W_CAP), raw))
    return Int32(select_(max_val >= 0.0, ext, Int32(0)))


class BlockClassifyKernel:
    def __init__(self, block_size: int):
        self.block_size = block_size
        self.bitmap_threads = max(32, min(block_size, _MAX_THREADS_PER_BLOCK))
        self.bitmap_elems = (
            block_size + self.bitmap_threads - 1
        ) // self.bitmap_threads
        self.tile_kv = _CLASSIFY_TILE_KV

    @cute.jit
    def __call__(
        self,
        q_fn_b: cute.Tensor,
        kv_fn_b: cute.Tensor,
        q_noise_b: cute.Tensor,
        kv_noise_b: cute.Tensor,
        q_doc_min: cute.Tensor,
        q_doc_max: cute.Tensor,
        kv_doc_min: cute.Tensor,
        kv_doc_max: cute.Tensor,
        q_fn_min: cute.Tensor,
        q_fn_max: cute.Tensor,
        kv_fn_min: cute.Tensor,
        kv_fn_max: cute.Tensor,
        q_noise_min: cute.Tensor,
        q_noise_max: cute.Tensor,
        kv_noise_min: cute.Tensor,
        kv_noise_max: cute.Tensor,
        q_pos_min: cute.Tensor,
        q_pos_max: cute.Tensor,
        kv_pos_min: cute.Tensor,
        kv_pos_max: cute.Tensor,
        bits: cute.Tensor,  # (2, _W_CAP, n_q + n_kv): fn row 0, noise row 1
        block_type: cute.Tensor,  # (n_q, n_kv) int32 out
        n_q: Int32,
        n_kv: Int32,
        n_total: Int32,
        n_kv_tiles: Int32,
        stream,
    ):
        self.bitmap_kernel(
            q_fn_b,
            kv_fn_b,
            q_noise_b,
            kv_noise_b,
            q_fn_max,
            q_noise_max,
            kv_fn_max,
            kv_noise_max,
            bits,
            n_q,
            n_total,
        ).launch(
            grid=[n_total, 1, 1],
            block=[self.bitmap_threads, 1, 1],
            stream=stream,
        )
        self.classify_kernel(
            q_doc_min,
            q_doc_max,
            kv_doc_min,
            kv_doc_max,
            q_fn_min,
            q_fn_max,
            kv_fn_min,
            kv_fn_max,
            q_noise_min,
            q_noise_max,
            kv_noise_min,
            kv_noise_max,
            q_pos_min,
            q_pos_max,
            kv_pos_min,
            kv_pos_max,
            bits,
            block_type,
            n_q,
            n_kv,
        ).launch(
            grid=[n_kv_tiles, n_q, 1],
            block=[self.tile_kv, 1, 1],
            stream=stream,
        )

    @cute.jit
    def _build_side(
        self,
        fn_b: cute.Tensor,
        noise_b: cute.Tensor,
        fn_max: cute.Tensor,
        noise_max: cute.Tensor,
        bits: cute.Tensor,
        row: Int32,
        col: Int32,
        n_total: Int32,
        tidx: Int32,
    ):
        # This CTA exclusively owns column `col` of bits: zero the words the
        # classify kernel will read (extents derived from the same stats it
        # uses as loop bounds), then set bits with global atomics.
        ext_fn = _ext_words(fn_max[row])
        ext_noise = _ext_words(noise_max[row])
        w_hi = Int32(select_(ext_fn > ext_noise, ext_fn, ext_noise))

        w = Int32(tidx)
        while w < w_hi:
            bits[0, w, col] = Int32(0)
            bits[1, w, col] = Int32(0)
            w = w + self.bitmap_threads
        cute.arch.sync_threads()

        noise_base = Int32(_W_CAP) * n_total
        for e in cutlass.range_constexpr(self.bitmap_elems):
            t = tidx + e * self.bitmap_threads
            if t < self.block_size:
                v_fn = fn_b[row, t]
                if (v_fn >= 0.0) & (v_fn < _W_CAP * 32.0):
                    fn_id = Int32(v_fn)
                    cute.arch.atomic_or(
                        bits.iterator + ((fn_id >> 5) * n_total + col),
                        Int32(1) << (fn_id & 31),
                    )
                v_noise = noise_b[row, t]
                if (v_noise >= 0.0) & (v_noise < _W_CAP * 32.0):
                    noise_id = Int32(v_noise)
                    cute.arch.atomic_or(
                        bits.iterator + (noise_base + (noise_id >> 5) * n_total + col),
                        Int32(1) << (noise_id & 31),
                    )

    @cute.kernel
    def bitmap_kernel(
        self,
        q_fn_b: cute.Tensor,
        kv_fn_b: cute.Tensor,
        q_noise_b: cute.Tensor,
        kv_noise_b: cute.Tensor,
        q_fn_max: cute.Tensor,
        q_noise_max: cute.Tensor,
        kv_fn_max: cute.Tensor,
        kv_noise_max: cute.Tensor,
        bits: cute.Tensor,
        n_q: Int32,
        n_total: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if bidx < n_q:
            self._build_side(
                q_fn_b,
                q_noise_b,
                q_fn_max,
                q_noise_max,
                bits,
                bidx,
                bidx,
                n_total,
                tidx,
            )
        else:
            self._build_side(
                kv_fn_b,
                kv_noise_b,
                kv_fn_max,
                kv_noise_max,
                bits,
                bidx - n_q,
                bidx,
                n_total,
                tidx,
            )

    @cute.kernel
    def classify_kernel(
        self,
        q_doc_min: cute.Tensor,
        q_doc_max: cute.Tensor,
        kv_doc_min: cute.Tensor,
        kv_doc_max: cute.Tensor,
        q_fn_min: cute.Tensor,
        q_fn_max: cute.Tensor,
        kv_fn_min: cute.Tensor,
        kv_fn_max: cute.Tensor,
        q_noise_min: cute.Tensor,
        q_noise_max: cute.Tensor,
        kv_noise_min: cute.Tensor,
        kv_noise_max: cute.Tensor,
        q_pos_min: cute.Tensor,
        q_pos_max: cute.Tensor,
        kv_pos_min: cute.Tensor,
        kv_pos_max: cute.Tensor,
        bits: cute.Tensor,
        block_type: cute.Tensor,
        n_q: Int32,
        n_kv: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        i = bidy
        j = bidx * self.tile_kv + tidx

        if j < n_kv:
            qd_min = q_doc_min[i]
            qd_max = q_doc_max[i]
            qf_min = q_fn_min[i]
            qf_max = q_fn_max[i]
            qn_min = q_noise_min[i]
            qn_max = q_noise_max[i]
            qp_min = q_pos_min[i]
            qp_max = q_pos_max[i]

            kd_min = kv_doc_min[j]
            kd_max = kv_doc_max[j]
            kf_min = kv_fn_min[j]
            kf_max = kv_fn_max[j]
            kn_min = kv_noise_min[j]
            kn_max = kv_noise_max[j]
            kp_min = kv_pos_min[j]
            kp_max = kv_pos_max[j]

            same_doc = (qd_min == qd_max) & (kd_min == kd_max) & (qd_min == kd_min)
            diff_doc = (qd_max < kd_min) | (kd_max < qd_min)
            all_full_attn = (
                same_doc
                & (qf_min == qf_max)
                & (kf_min == kf_max)
                & (qf_min >= 0.0)
                & (kf_min >= 0.0)
                & (qf_min == kf_min)
            )

            # Words beyond a block's extent hold no ids, so per-pair loops can
            # stop at the kv extents (every test ANDs with a kv word); q words
            # past the q extent select to 0.
            ext_q_fn = _ext_words(qf_max)
            ext_q_noise = _ext_words(qn_max)
            ext_kv_fn = _ext_words(kf_max)
            ext_kv_noise = _ext_words(kn_max)
            w_hi = Int32(select_(ext_kv_fn > ext_kv_noise, ext_kv_fn, ext_kv_noise))

            kv_extra = Int32(0)
            noise_shared = Int32(0)
            fn_shared = Int32(0)
            w = Int32(0)
            while w < w_hi:
                kv_fn_w = bits[0, w, n_q + j]
                kv_noise_w = bits[1, w, n_q + j]
                q_fn_w = Int32(select_(w < ext_q_fn, bits[0, w, i], Int32(0)))
                q_noise_w = Int32(select_(w < ext_q_noise, bits[1, w, i], Int32(0)))
                kv_extra = kv_extra | (kv_noise_w & ~q_noise_w)
                noise_shared = noise_shared | (kv_noise_w & q_noise_w)
                fn_shared = fn_shared | (kv_fn_w & q_fn_w)
                w = w + 1

            kv_subset_of_q = kv_extra == 0
            noise_sets_disjoint = noise_shared == 0
            no_fn_overlap = fn_shared == 0

            remove_noise_none = (kn_min >= 0.0) & noise_sets_disjoint
            q_single_positive_noise = (qn_min == qn_max) & (qn_min >= 0.0)
            remove_noise_all = (kn_max < 0.0) | (
                q_single_positive_noise & kv_subset_of_q
            )

            full_attn_safe = all_full_attn & remove_noise_all
            below_diagonal_full = same_doc & (qp_min > kp_max) & remove_noise_all

            fn_ranges_disjoint = (qf_max < kf_min) | (kf_max < qf_min)
            no_full_noise_range = (qf_max < 0.0) | (kf_max < 0.0) | fn_ranges_disjoint
            no_full_noise_possible = no_full_noise_range | no_fn_overlap
            above_diagonal_empty = (qp_max < kp_min) & (
                no_full_noise_possible | remove_noise_none
            )

            is_empty = diff_doc | remove_noise_none | above_diagonal_empty
            is_full = full_attn_safe | below_diagonal_full

            bt = Int32(select_(is_empty, Int32(BLOCK_EMPTY), Int32(BLOCK_PARTIAL)))
            bt = Int32(select_(is_full, Int32(BLOCK_FULL), bt))
            block_type[i, j] = bt


@functools.cache
def _get_compiled_classify_kernel(block_size: int) -> tuple[Callable, int]:
    kernel_obj = BlockClassifyKernel(block_size)

    sym_nq = cute.sym_int()
    sym_nkv = cute.sym_int()
    sym_total = cute.sym_int()

    def _blocked_fake(sym):
        return cute.runtime.make_fake_compact_tensor(
            Float32, (sym, block_size), stride_order=(1, 0), assumed_align=4
        )

    def _stat_fake(dtype, sym):
        # unbind(0) views land on arbitrary element offsets, so only assume
        # element alignment.
        return cute.runtime.make_fake_compact_tensor(
            dtype, (sym,), stride_order=(0,), assumed_align=dtype.width // 8
        )

    bits_fake = cute.runtime.make_fake_compact_tensor(
        Int32, (2, _W_CAP, sym_total), stride_order=(2, 1, 0), assumed_align=16
    )
    block_type_fake = cute.runtime.make_fake_compact_tensor(
        Int32, (sym_nq, sym_nkv), stride_order=(1, 0), assumed_align=4
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        _blocked_fake(sym_nq),
        _blocked_fake(sym_nkv),
        _blocked_fake(sym_nq),
        _blocked_fake(sym_nkv),
        _stat_fake(Int64, sym_nq),
        _stat_fake(Int64, sym_nq),
        _stat_fake(Int64, sym_nkv),
        _stat_fake(Int64, sym_nkv),
        _stat_fake(Float32, sym_nq),
        _stat_fake(Float32, sym_nq),
        _stat_fake(Float32, sym_nkv),
        _stat_fake(Float32, sym_nkv),
        _stat_fake(Float32, sym_nq),
        _stat_fake(Float32, sym_nq),
        _stat_fake(Float32, sym_nkv),
        _stat_fake(Float32, sym_nkv),
        _stat_fake(Int64, sym_nq),
        _stat_fake(Int64, sym_nq),
        _stat_fake(Int64, sym_nkv),
        _stat_fake(Int64, sym_nkv),
        bits_fake,
        block_type_fake,
        Int32(1),  # Dummy n_q
        Int32(1),  # Dummy n_kv
        Int32(2),  # Dummy n_total
        Int32(1),  # Dummy n_kv_tiles
        stream_fake,
        options="--enable-tvm-ffi",
    )
    return compiled_kernel, kernel_obj.tile_kv


def classify_block_types_cutedsl(
    stats: BlockStats,
    q_fn_b: Tensor,
    kv_fn_b: Tensor,
    q_noise_b: Tensor,
    kv_noise_b: Tensor,
    *,
    max_id: int | None = None,
) -> Tensor:
    """Stats + blocked fn/noise -> ``block_type`` grid, one host call.

    Matches the torch ``classify`` sub-op exactly under the precondition
    that fn/noise values are integral ids (production builds them as split
    indices stored as float32), and never syncs: per-block bitmap extents
    are derived on-device from the fn/noise max stats, so ``max_id`` is
    only used for validation (ids must be < ``_W_CAP * 32`` = 16384;
    unchecked when ``max_id`` is None).

    Raises:
        ValueError: On malformed inputs or an ID space too large to bitmap.
    """
    blocked = (
        ("q_fn_b", q_fn_b),
        ("kv_fn_b", kv_fn_b),
        ("q_noise_b", q_noise_b),
        ("kv_noise_b", kv_noise_b),
    )
    for name, t in blocked:
        if (
            t.dim() != 2
            or t.dtype != torch.float32
            or not t.is_contiguous()
            or not t.is_cuda
        ):
            raise ValueError(
                f"{name} must be a contiguous 2D float32 CUDA tensor, got "
                f"{t.dtype=}, {t.dim()=}, {t.is_contiguous()=}, {t.is_cuda=}."
            )
    n_q_blocks, block = q_fn_b.shape
    n_kv_blocks = kv_fn_b.shape[0]
    device = q_fn_b.device
    if q_noise_b.shape != q_fn_b.shape or kv_noise_b.shape != kv_fn_b.shape:
        raise ValueError("fn and noise blocked tensors must have matching shapes.")
    if kv_fn_b.shape[1] != block or block <= 0 or block & (block - 1):
        raise ValueError(
            f"block width must match on both sides and be a power of two, got "
            f"{q_fn_b.shape=}, {kv_fn_b.shape=}."
        )
    for name in BlockStats.__dataclass_fields__:
        t = getattr(stats, name)
        length = n_q_blocks if name.startswith("q_") else n_kv_blocks
        dtype = torch.float32 if ("fn" in name or "noise" in name) else torch.int64
        if t.dim() != 1 or t.numel() != length or t.dtype != dtype:
            raise ValueError(
                f"stats.{name} must be a 1D {dtype} tensor of length {length}, "
                f"got {t.dtype=}, {t.shape=}."
            )
        if not t.is_contiguous() or t.device != device:
            raise ValueError(f"stats.{name} must be contiguous on {device}.")
    if max_id is not None and (max_id + 32) // 32 > _W_CAP:
        raise ValueError(
            f"fn/noise ID space too large to bitmap: {max_id=} "
            f"(>= {_W_CAP * 32} IDs; production split counts are at most a few "
            "hundred)."
        )

    if n_q_blocks == 0 or n_kv_blocks == 0:
        return torch.empty((n_q_blocks, n_kv_blocks), dtype=torch.int32, device=device)

    n_total = n_q_blocks + n_kv_blocks
    bits = torch.empty((2, _W_CAP, n_total), dtype=torch.int32, device=device)
    block_type = torch.empty(
        (n_q_blocks, n_kv_blocks), dtype=torch.int32, device=device
    )

    kernel_fn, tile_kv = _get_compiled_classify_kernel(block)
    kernel_fn(
        q_fn_b,
        kv_fn_b,
        q_noise_b,
        kv_noise_b,
        stats.q_doc_min,
        stats.q_doc_max,
        stats.kv_doc_min,
        stats.kv_doc_max,
        stats.q_fn_min,
        stats.q_fn_max,
        stats.kv_fn_min,
        stats.kv_fn_max,
        stats.q_noise_min,
        stats.q_noise_max,
        stats.kv_noise_min,
        stats.kv_noise_max,
        stats.q_pos_min,
        stats.q_pos_max,
        stats.kv_pos_min,
        stats.kv_pos_max,
        bits,
        block_type,
        n_q_blocks,
        n_kv_blocks,
        n_total,
        (n_kv_blocks + tile_kv - 1) // tile_kv,
    )
    return block_type


_BCSR_WARPS_PER_CTA = 4


class BcsrKernel:
    def __init__(self):
        self.warps_per_cta = _BCSR_WARPS_PER_CTA
        self.threads = 32 * _BCSR_WARPS_PER_CTA

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
        n_q: Int32,
        n_kv: Int32,
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
            n_q,
            n_kv,
        ).launch(
            grid=[num_ctas, 1, 1],
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.jit
    def _warp_bcsr(
        self,
        block_type: cute.Tensor,
        num_out: cute.Tensor,
        idx_out: cute.Tensor,
        r: Int32,
        length: Int32,
        target: int,
        lane: Int32,
        transpose: bool,
    ):
        # One warp per output row: reproduce the stable descending argsort of
        # the 0/1 membership mask (member columns ascending, then non-members
        # ascending) with ballot prefix sums. Pass 1 counts members, pass 2
        # scatters each column to its position.
        cnt = Int32(0)
        base = Int32(0)
        while base < length:
            j = base + lane
            jj = Int32(select_(j < length, j, Int32(0)))
            if cutlass.const_expr(transpose):
                v = block_type[jj, r]
            else:
                v = block_type[r, jj]
            m = (j < length) & (v == target)
            ballot = cute.arch.vote_ballot_sync(m)
            cnt = cnt + cute.arch.popc(ballot)
            base = base + 32
        if lane == 0:
            num_out[r] = cnt

        lt_mask = (Int32(1) << lane) - 1
        carry = Int32(0)
        base = Int32(0)
        while base < length:
            j = base + lane
            jj = Int32(select_(j < length, j, Int32(0)))
            if cutlass.const_expr(transpose):
                v = block_type[jj, r]
            else:
                v = block_type[r, jj]
            m = (j < length) & (v == target)
            ballot = cute.arch.vote_ballot_sync(m)
            members_before = carry + cute.arch.popc(ballot & lt_mask)
            pos = Int32(select_(m, members_before, cnt + (j - members_before)))
            if j < length:
                idx_out[r, pos] = j
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
        n_q: Int32,
        n_kv: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        wg = bidx * self.warps_per_cta + tidx // 32
        lane = cute.arch.lane_idx()

        if wg < n_q:
            self._warp_bcsr(
                block_type, kv_num, kv_idx, wg, n_kv, BLOCK_PARTIAL, lane, False
            )
        elif wg < 2 * n_q:
            self._warp_bcsr(
                block_type,
                full_kv_num,
                full_kv_idx,
                wg - n_q,
                n_kv,
                BLOCK_FULL,
                lane,
                False,
            )
        elif wg < 2 * n_q + n_kv:
            self._warp_bcsr(
                block_type, q_num, q_idx, wg - 2 * n_q, n_q, BLOCK_PARTIAL, lane, True
            )
        elif wg < 2 * n_q + 2 * n_kv:
            self._warp_bcsr(
                block_type,
                full_q_num,
                full_q_idx,
                wg - (2 * n_q + n_kv),
                n_q,
                BLOCK_FULL,
                lane,
                True,
            )


@functools.cache
def _get_compiled_bcsr_kernel() -> tuple[Callable, int]:
    kernel_obj = BcsrKernel()

    sym_nq = cute.sym_int()
    sym_nkv = cute.sym_int()

    def _i32_fake(*shape):
        return cute.runtime.make_fake_compact_tensor(
            Int32,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=4,
        )

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        _i32_fake(sym_nq, sym_nkv),  # block_type
        _i32_fake(sym_nq),  # kv_num
        _i32_fake(sym_nq, sym_nkv),  # kv_idx
        _i32_fake(sym_nq),  # full_kv_num
        _i32_fake(sym_nq, sym_nkv),  # full_kv_idx
        _i32_fake(sym_nkv),  # q_num
        _i32_fake(sym_nkv, sym_nq),  # q_idx
        _i32_fake(sym_nkv),  # full_q_num
        _i32_fake(sym_nkv, sym_nq),  # full_q_idx
        Int32(1),  # Dummy n_q
        Int32(1),  # Dummy n_kv
        Int32(1),  # Dummy num_ctas
        stream_fake,
        options="--enable-tvm-ffi",
    )
    return compiled_kernel, kernel_obj.warps_per_cta


def bcsr_from_block_type_cutedsl(block_type: Tensor) -> tuple[Tensor, ...]:
    """``block_type`` grid -> the 8 BCSR tensors in a single kernel launch.

    Bit-exact replacement for the ``bcsr`` sub-op of
    ``build_block_sparse_from_cross_qkv_aux`` (including the argsort tail:
    non-member columns follow the members in ascending order). Returns
    (kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices,
    q_num_blocks, q_indices, full_q_num_blocks, full_q_indices).

    Raises:
        ValueError: If block_type is not a contiguous 2D int32 CUDA tensor.
    """
    if (
        block_type.dim() != 2
        or block_type.dtype != torch.int32
        or not block_type.is_contiguous()
        or not block_type.is_cuda
    ):
        raise ValueError(
            f"block_type must be a contiguous 2D int32 CUDA tensor, got "
            f"{block_type.dtype=}, {block_type.dim()=}, "
            f"{block_type.is_contiguous()=}, {block_type.is_cuda=}."
        )
    n_q, n_kv = block_type.shape
    device = block_type.device

    if n_q == 0 or n_kv == 0:
        partial_mask = (block_type == BLOCK_PARTIAL).to(torch.int8)[None, None]
        full_mask = (block_type == BLOCK_FULL).to(torch.int8)[None, None]
        kv_num, kv_idx = _dense_to_ordered(partial_mask)
        full_kv_num, full_kv_idx = _dense_to_ordered(full_mask)
        q_num, q_idx = _transpose_ordered(kv_num, kv_idx, num_cols=n_kv)
        full_q_num, full_q_idx = _transpose_ordered(
            full_kv_num, full_kv_idx, num_cols=n_kv
        )
        return (
            kv_num,
            kv_idx,
            full_kv_num,
            full_kv_idx,
            q_num,
            q_idx,
            full_q_num,
            full_q_idx,
        )

    kv_num = torch.empty((1, 1, n_q), dtype=torch.int32, device=device)
    kv_idx = torch.empty((1, 1, n_q, n_kv), dtype=torch.int32, device=device)
    full_kv_num = torch.empty((1, 1, n_q), dtype=torch.int32, device=device)
    full_kv_idx = torch.empty((1, 1, n_q, n_kv), dtype=torch.int32, device=device)
    q_num = torch.empty((1, 1, n_kv), dtype=torch.int32, device=device)
    q_idx = torch.empty((1, 1, n_kv, n_q), dtype=torch.int32, device=device)
    full_q_num = torch.empty((1, 1, n_kv), dtype=torch.int32, device=device)
    full_q_idx = torch.empty((1, 1, n_kv, n_q), dtype=torch.int32, device=device)

    kernel_fn, warps_per_cta = _get_compiled_bcsr_kernel()
    total_warps = 2 * (n_q + n_kv)
    kernel_fn(
        block_type,
        kv_num.view(n_q),
        kv_idx.view(n_q, n_kv),
        full_kv_num.view(n_q),
        full_kv_idx.view(n_q, n_kv),
        q_num.view(n_kv),
        q_idx.view(n_kv, n_q),
        full_q_num.view(n_kv),
        full_q_idx.view(n_kv, n_q),
        n_q,
        n_kv,
        (total_warps + warps_per_cta - 1) // warps_per_cta,
    )
    return (
        kv_num,
        kv_idx,
        full_kv_num,
        full_kv_idx,
        q_num,
        q_idx,
        full_q_num,
        full_q_idx,
    )


def build_block_sparse_from_cross_qkv_aux_cutedsl(
    aux_tensors: list[Tensor],
    block_size: int = 128,
) -> BlockSparseResult:
    """CuTe DSL version of ``build_block_sparse_from_cross_qkv_aux``.

    Composes the three CuTe DSL sub-ops (stats -> classify -> bcsr, four
    kernel launches over three host calls) into a bit-exact replacement for
    the torch op, under the classify precondition that fn/noise values are
    integral ids (production builds them as split indices). One host sync
    reads the largest fn/noise id: ids of 16384 or more do not fit classify's
    per-block bitmaps, so such layouts run the torch op instead.

    Raises:
        ValueError: If aux_tensors does not contain exactly 8 valid tensors.
    """
    if len(aux_tensors) != 8:
        raise ValueError(
            f"Expected 8 aux tensors for cross q/kv mask, got {len(aux_tensors)=}."
        )
    q_fn, kv_fn, q_noise, kv_noise, q_doc, kv_doc, q_pos, kv_pos = aux_tensors
    # Pad once up front (the stats kernel expects block-aligned lengths) and
    # reuse the padded fn/noise views as classify's blocked inputs.
    q_fn = _pad_to_blocks(q_fn, block_size, fill_value=-1)
    kv_fn = _pad_to_blocks(kv_fn, block_size, fill_value=-1)
    q_noise = _pad_to_blocks(q_noise, block_size, fill_value=-1)
    kv_noise = _pad_to_blocks(kv_noise, block_size, fill_value=-1)
    q_doc = _pad_to_blocks(q_doc, block_size)
    kv_doc = _pad_to_blocks(kv_doc, block_size)
    q_pos = _pad_to_blocks(q_pos, block_size)
    kv_pos = _pad_to_blocks(kv_pos, block_size)
    stats = compute_block_stats_cutedsl(
        [q_fn, kv_fn, q_noise, kv_noise, q_doc, kv_doc, q_pos, kv_pos], block_size
    )
    id_maxes = torch.cat(
        [stats.q_fn_max, stats.kv_fn_max, stats.q_noise_max, stats.kv_noise_max]
    )
    max_id = int(id_maxes.max()) if id_maxes.numel() else -1
    if max_id >= _W_CAP * 32:
        # Ids past the bitmaps would drop out and misclassify blocks.
        return build_block_sparse_from_cross_qkv_aux(aux_tensors, block_size)
    block_type = classify_block_types_cutedsl(
        stats,
        q_fn.reshape(-1, block_size),
        kv_fn.reshape(-1, block_size),
        q_noise.reshape(-1, block_size),
        kv_noise.reshape(-1, block_size),
        max_id=max_id,
    )
    (
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        q_num_blocks,
        q_indices,
        full_q_num_blocks,
        full_q_indices,
    ) = bcsr_from_block_type_cutedsl(block_type)
    return BlockSparseResult(
        n_q_blocks=block_type.shape[0],
        n_kv_blocks=block_type.shape[1],
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
