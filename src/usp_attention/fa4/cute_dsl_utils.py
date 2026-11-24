# Copyright (c) 2025, Tri Dao. (warp_reduce)
# SPDX-FileCopyrightText: 2022 the FlashAttention contributors, as shown by third_party_licenses/flash-attention-AUTHORS (warp_reduce)
# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT AND BSD-3-Clause
"""Shared CuTe DSL helpers for attention and block-sparse kernels."""

from __future__ import annotations

import functools
import math

import cutlass
import cutlass.cute as cute
import torch
import torch.version
from cutlass import Float32, Int32, Int64, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@functools.cache
def device_support_pdl(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 9


@functools.cache
def get_num_sm(device: torch.device) -> int:
    # get the compute capability of the device, which would be cached
    return torch.cuda.get_device_properties(device).multi_processor_count


@dsl_user_op
def get_ptr_as_int64(tensor: cute.Tensor, offset: Int32, *, loc=None, ip=None) -> Int64:
    """Get the memory address of tensor[offset] as Int64.

    WARNING: This uses ptrtoint which strips address space information.
    For SMEM tensors, the resulting Int64 is a raw SMEM offset that does NOT
    work with generic-addressing loads (ld.v4.u32). Use only with explicit
    address-space loads (ld.global.*) or for global memory tensors.
    """
    elem_ptr = tensor.iterator + Int32(offset)
    ptr_int = llvm.ptrtoint(T.i64(), elem_ptr.llvm_ptr, loc=loc, ip=ip)
    return Int64(ptr_int)


@dsl_user_op
def ld_global_v4_f32(
    base_ptr: Int64, *, loc=None, ip=None
) -> tuple[Float32, Float32, Float32, Float32]:
    """Load 128 bits (4 x float32) from global memory."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),  # ty: ignore[unresolved-attribute]
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    v0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    v1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    v2 = llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)
    v3 = llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)

    return Float32(v0), Float32(v1), Float32(v2), Float32(v3)


@dsl_user_op
def ld_global_v2_u64(base_ptr: Int64, *, loc=None, ip=None) -> tuple[Uint64, Uint64]:
    """Load 128 bits (2 x uint64) from global memory."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i64(), T.i64()]),  # ty: ignore[unresolved-attribute]
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
        "ld.global.v2.u64 {$0, $1}, [$2];",
        "=l,=l,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    v0 = llvm.extractvalue(T.i64(), result, [0], loc=loc, ip=ip)
    v1 = llvm.extractvalue(T.i64(), result, [1], loc=loc, ip=ip)

    return Uint64(v0), Uint64(v1)


@dsl_user_op
def fmax_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """Compute max of two float32 values using PTX max.f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fmin_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """Compute min of two float32 values using PTX min.f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "min.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def imax_s64(a: Int64, b: Int64, *, loc=None, ip=None):
    """Compute max of two int64 values using PTX max.f64."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [Int64(a).ir_value(loc=loc, ip=ip), Int64(b).ir_value(loc=loc, ip=ip)],
            "max.s64 $0, $1, $2;",
            "=l,l, l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def imin_s64(a: Int64, b: Int64, *, loc=None, ip=None):
    """Compute max of two int64 values using PTX max.f64."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [Int64(a).ir_value(loc=loc, ip=ip), Int64(b).ir_value(loc=loc, ip=ip)],
            "min.s64 $0, $1, $2;",
            "=l, l, l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def max_min_fp32(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
) -> tuple[Float32, Float32]:
    """
    Compute max/min  value across 4 fp32 values.

    Uses tree reduction: 4 -> 2 -> 1 values.
    Returns a fp32 containing the max and min of values.
    """

    max01 = fmax_f32(v0, v1)
    max23 = fmax_f32(v2, v3)
    max0123 = fmax_f32(max01, max23)

    min01 = fmin_f32(v0, v1)
    min23 = fmin_f32(v2, v3)
    min0123 = fmin_f32(min01, min23)

    return max0123, min0123


@cute.jit
def max_min_i64(
    v0: Int64,
    v1: Int64,
    v2: Int64,
    v3: Int64,
) -> tuple[Int64, Int64]:
    """
    Compute max/min  value across 4 fp32 values.

    Uses tree reduction: 4 -> 2 -> 1 values.
    Returns a fp32 containing the max and min of values.
    """

    max01 = imax_s64(v0, v1)
    max23 = imax_s64(v2, v3)
    max0123 = imax_s64(max01, max23)

    min01 = imin_s64(v0, v1)
    min23 = imin_s64(v2, v3)
    min0123 = imin_s64(min01, min23)

    return max0123, min0123


@cute.jit
def warp_reduce(val, op, width: cutlass.Constexpr[int] = 32):  # ty: ignore[invalid-parameter-default]
    """Reduce across threads in a warp using butterfly shuffle.

    Adapted from FlashAttention-4 (``flash_attn/cute/utils.py``, BSD-3-Clause).
    """
    if cutlass.const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_rmem_tensor(val.shape, val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    else:
        for i in cutlass.range_constexpr(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
        return val
