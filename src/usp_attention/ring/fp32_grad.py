# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Accumulate ring-attention gradients in fp32 across ring steps.

FA4's backward accumulates dq (and, with grouped-query attention, dk and dv)
in fp32 buffers and converts them to the input dtype at the end of every
call. Running one call per ring step therefore rounds every step's partial
gradient to bf16 before the ring adds them up, which leaves ring gradients
noticeably less accurate than a single FA4 call over the whole sequence.

:func:`fp32_grad_step` wraps one ring step's ``_flash_attn_bwd`` call and
intercepts the two FA4 helpers that zero and convert those buffers:

- dq: each step's freshly zeroed fp32 accumulator is seeded with the running
  total, and only the last step converts it to the output dtype.
- dk / dv (GQA only, where FA4 accumulates them in fp32): the conversion is
  skipped and the raw fp32 accumulators are handed back, so the ring can add
  and rotate them in fp32 and convert once with :func:`convert_dkv`.

FA4's own kernels and buffer layouts are used unchanged. The softmax scale is
applied in the conversion, which is linear, so converting the sum once equals
summing converted partials up to rounding.

FA4's head_dim-256 backward on Blackwell has no fp32 accumulators: it writes
dq, dk and dv directly in the input dtype, so neither helper sees them. Callers
check :attr:`Fp32GradState.dq_in_accum` and then sum each step's dq themselves.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator
from dataclasses import dataclass, field

import torch
from torch import Tensor

try:
    import flash_attn.cute.interface as _fa
except ImportError:  # pragma: no cover - flash-attn-4 not installed
    _fa = None


@dataclass(kw_only=True, slots=True)
class Fp32GradState:
    """Running fp32 gradient state for one ring backward.

    Attributes:
        dq_accum: FA4's raw fp32 dq accumulator, summed over the steps so far.
        dk_raw: Raw fp32 dk accumulator from the most recent step (GQA only).
        dv_raw: Raw fp32 dv accumulator from the most recent step (GQA only).
        dkv_convert: The captured conversion calls for dk and dv, used by
            :func:`convert_dkv` to turn raw accumulators into gradients.
        dq_in_accum: FA4 accumulated the most recent step's dq in its fp32
            buffer, so ``dq_accum`` holds the running total. False when FA4
            wrote dq directly (head_dim 256 on Blackwell): each step's returned
            dq is then that step's partial, and the caller sums them.
    """

    dq_accum: Tensor | None = None
    dk_raw: Tensor | None = None
    dv_raw: Tensor | None = None
    dkv_convert: list[tuple[tuple, dict]] = field(default_factory=list)
    dq_in_accum: bool = False


@contextlib.contextmanager
def fp32_grad_step(
    state: Fp32GradState, *, is_first: bool, is_last: bool
) -> Iterator[None]:
    """Run one ring step's ``_flash_attn_bwd`` with fp32 gradient accumulation.

    Inside the context, the dq returned by ``_flash_attn_bwd`` is only valid on
    the last step, and with GQA the returned dk / dv are never valid: read
    ``state.dk_raw`` / ``state.dv_raw`` instead. Both hold only when FA4 used
    its fp32 accumulators: after the context, ``state.dq_in_accum`` is False
    (and ``state.dk_raw`` None) when it wrote the gradients directly, and the
    returned dq is this step's partial.

    Raises:
        RuntimeError: If FA4 used the dq accumulator on some steps of one ring
            backward but not on others.
    """
    if _fa is None:
        raise RuntimeError(
            "flash_attn.cute is required for fp32 gradient accumulation."
        )
    orig_pre = _fa._bwd_preprocess
    orig_post = _fa._bwd_postprocess_convert
    step_dq: list[Tensor] = []
    dkv_calls: list[tuple[Tensor, Tensor, tuple, dict]] = []

    def preprocess(out, dout, dpsum, lse, lse_log2, dq_accum, *args, **kwargs):
        orig_pre(out, dout, dpsum, lse, lse_log2, dq_accum, *args, **kwargs)
        if dq_accum is not None:
            # The preprocess kernel zeroed the buffer; continue from the total.
            if not is_first:
                assert state.dq_accum is not None
                dq_accum.copy_(state.dq_accum)
            step_dq.append(dq_accum)

    def postprocess(accum, output, *args, **kwargs):
        if step_dq and accum is step_dq[0]:
            state.dq_accum = accum
            if is_last:
                orig_post(accum, output, *args, **kwargs)
            return
        dkv_calls.append((accum, output, args, kwargs))

    # The originals look up their compile caches through their module-level
    # names, which now resolve to these wrappers; share the cache objects.
    preprocess.compile_cache = orig_pre.compile_cache  # type: ignore[attr-defined]
    postprocess.compile_cache = orig_post.compile_cache  # type: ignore[attr-defined]
    _fa._bwd_preprocess = preprocess
    _fa._bwd_postprocess_convert = postprocess
    try:
        yield
    finally:
        _fa._bwd_preprocess = orig_pre
        _fa._bwd_postprocess_convert = orig_post
    in_accum = bool(step_dq)
    if not is_first and in_accum != state.dq_in_accum:
        raise RuntimeError(
            "FA4 switched between accumulating dq in fp32 and writing it "
            "directly within one ring backward."
        )
    state.dq_in_accum = in_accum
    if dkv_calls:
        (dk_raw, _, dk_args, dk_kwargs), (dv_raw, _, dv_args, dv_kwargs) = dkv_calls
        state.dk_raw, state.dv_raw = dk_raw, dv_raw
        state.dkv_convert = [(dk_args, dk_kwargs), (dv_args, dv_kwargs)]
    else:
        state.dk_raw = state.dv_raw = None


def convert_dkv(
    state: Fp32GradState, dk_raw: Tensor, dv_raw: Tensor, dk: Tensor, dv: Tensor
) -> None:
    """Convert summed raw fp32 dk / dv accumulators into ``dk`` / ``dv`` in place."""
    assert _fa is not None and len(state.dkv_convert) == 2
    (dk_args, dk_kwargs), (dv_args, dv_kwargs) = state.dkv_convert
    _fa._bwd_postprocess_convert(dk_raw, dk, *dk_args, **dk_kwargs)
    _fa._bwd_postprocess_convert(dv_raw, dv, *dv_args, **dv_kwargs)


@contextlib.contextmanager
def fp32_grad_outputs(*, skip_dq: bool = False) -> Iterator[dict[int, Tensor]]:
    """Have ``_flash_attn_bwd`` hand back its fp32-accumulated gradients in fp16.

    FA4's conversion kernel only writes fp16 / bf16, and its fp32 accumulators
    are stored in a tiled layout. Inside the context, every gradient FA4
    produces through that kernel (dq, and dk / dv with grouped-query attention)
    is converted to fp16 instead, scaled by a power of two that puts the
    largest magnitude near 2**14, and handed back with its inverse scale. fp16
    keeps three more mantissa bits than bf16, so adding several calls'
    gradients in fp32 and rounding once to bf16 matches a single FA4 call. The
    yielded dict maps ``id()`` of the tensor FA4 returns to ``(fp16 gradient,
    inverse scale)``; it is filled when the context exits (read it with
    :func:`as_scaled` after the ``with`` block). The tensor FA4 returns for a
    converted gradient is left unwritten. Gradients FA4 writes directly (dk /
    dv without GQA, and every gradient of its head_dim-256 Blackwell backward)
    are not covered. ``skip_dq`` leaves dq (FA4's first conversion) in the
    usual dtype, for callers whose query rows each come from one call.

    The conversions run when the context exits, after one host sync for all of
    the call's scales.
    """
    if _fa is None:
        raise RuntimeError(
            "flash_attn.cute is required for fp32 gradient accumulation."
        )
    orig_post = _fa._bwd_postprocess_convert
    f16 = _fa.torch2cute_dtype_map[torch.float16]
    bf16 = _fa.torch2cute_dtype_map[torch.bfloat16]
    exact: dict[int, Tensor] = {}
    inf = float("inf")
    # (device amax, host multiplier, accum, output, scale, args, kwargs) per
    # deferred conversion.
    pending: list[tuple[Tensor, float, Tensor, Tensor, float, tuple, dict]] = []

    seen = [0]

    def postprocess(accum, output, scale, *args, **kwargs):
        seen[0] += 1
        is_dq = seen[0] == 1
        if skip_dq and is_dq:
            orig_post(accum, output, scale, *args, **kwargs)
            return
        if is_dq:
            # A bf16 conversion first: its rows give the largest magnitude
            # (dq's accumulator padding may hold stale values), and bf16
            # cannot overflow where an fp16 output could.
            probe = torch.empty_like(output, dtype=torch.bfloat16)
            if "dtype" in kwargs:
                orig_post(accum, probe, scale, *args, **{**kwargs, "dtype": bf16})
            else:
                orig_post(accum, probe, scale, *args[:3], bf16, *args[4:], **kwargs)
            amax, mult = torch.linalg.vector_norm(probe, ord=inf), 1.0
        else:
            # dK / dV accumulators are zero-initialized, padding included, so
            # their largest magnitude bounds the converted gradient.
            amax = torch.linalg.vector_norm(accum, ord=inf)
            mult = abs(float(scale))
        # The output dtype is part of the conversion kernel's compile key:
        # (cu_seqlens, seqused, arch, dtype, ...) after the scale.
        if "dtype" in kwargs:
            kwargs = {**kwargs, "dtype": f16}
        else:
            args = (*args[:3], f16, *args[4:])
        pending.append((amax, mult, accum, output, scale, args, kwargs))

    postprocess.compile_cache = orig_post.compile_cache  # type: ignore[attr-defined]
    _fa._bwd_postprocess_convert = postprocess
    try:
        yield exact
    finally:
        _fa._bwd_postprocess_convert = orig_post
    if not pending:
        return
    # One host sync for every gradient of the call.
    amaxes = torch.stack([p[0].float() for p in pending]).tolist()
    for peak, (_, mult, accum, output, scale, args, kwargs) in zip(
        amaxes, pending, strict=True
    ):
        amax = peak * mult
        shift = math.floor(math.log2(2.0**14 / amax)) if amax > 0 else 0
        factor = 2.0 ** max(min(shift, 60), -60)
        out16 = torch.empty_like(output, dtype=torch.float16)
        orig_post(accum, out16, float(scale) * factor, *args, **kwargs)
        exact[id(output)] = (out16, 1.0 / factor)


def as_scaled(exact: dict, grad: Tensor) -> tuple[Tensor, float]:
    """``(tensor, scale)`` with ``tensor * scale`` the gradient FA4 returned as ``grad``.

    The scaled fp16 tensor for gradients :func:`fp32_grad_outputs` converted,
    else ``grad`` itself (already in the input dtype) with scale 1.
    """
    return exact.get(id(grad), (grad, 1.0))
