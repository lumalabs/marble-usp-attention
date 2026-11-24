# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Regression tests for the ring path: merger -inf rows, ring op LSE shape
and differentiability, dq accumulation when FA4 writes dq directly, deferred
fp16 gradient conversions, and the batched fp16 scaling of the dense paths.

CPU unless marked ``gpu``; the FA4 helpers are replaced by a small fake.
"""

from __future__ import annotations

import math
import os
import socket

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from usp_attention.ring import fp32_grad
from usp_attention.ring.dense_gather import _f16_factor, _scaled_f16, _to_f16
from usp_attention.ring.flash4_ring_attn import (
    HAS_FLASH4_CUTE,
    Flash4RingStepMeta,
    NativeRingMask,
    _add_step_dq,
    _ring_lse_shape,
)
from usp_attention.ring.generation_ring import _ring_factors
from usp_attention.ring.ring_combine import _LocalRingResultMerger

NEG_INF = float("-inf")

# ---------------------------------------------------------------------------
# _LocalRingResultMerger with -inf LSE rows.


def _merge(steps: list[tuple[torch.Tensor, torch.Tensor]]):
    merger = _LocalRingResultMerger()
    for out, lse in steps:
        merger.step(out, lse)
    return merger.result()


def _reference(steps: list[tuple[torch.Tensor, torch.Tensor]]):
    """fp64 softmax-weighted combination; rows without keys give zeros."""
    outs = torch.stack([o for o, _ in steps]).double()
    lses = torch.stack([lse for _, lse in steps]).double()
    lse = torch.logsumexp(lses, dim=0)
    weights = torch.exp(lses - lse).nan_to_num(0.0)
    return (weights.unsqueeze(-1) * outs).sum(0), lse


def _steps_with_empty_rows() -> list[tuple[torch.Tensor, torch.Tensor]]:
    g = torch.Generator().manual_seed(0)
    # Rows: [-inf, f, f], [-inf, -inf, f], all -inf, [f, -inf, f], all
    # finite, [f, -inf, -inf]. FA4 writes zeros for rows without keys.
    finite = torch.tensor(
        [
            [False, True, True],
            [False, False, True],
            [False, False, False],
            [True, False, True],
            [True, True, True],
            [True, False, False],
        ]
    )
    steps = []
    for s in range(3):
        rows = finite[:, s]
        lse = torch.where(rows, torch.randn(1, 6, generator=g) * 4, NEG_INF)
        out = torch.randn(1, 6, 8, generator=g) * rows[None, :, None]
        steps.append((out, lse))
    return steps


def test_merger_handles_rows_without_keys() -> None:
    steps = _steps_with_empty_rows()
    out, lse = _merge(steps)
    want_out, want_lse = _reference(steps)
    assert not out.isnan().any() and not lse.isnan().any()
    torch.testing.assert_close(out.double(), want_out, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(lse.double(), want_lse, atol=1e-6, rtol=1e-6)
    assert lse[0, 2] == NEG_INF


def test_merger_handles_rows_without_keys_bshd() -> None:
    """The ring's layout: out (B, S, H, D), lse (B, S, H)."""
    g = torch.Generator().manual_seed(1)
    steps = []
    for s in range(4):
        lse = torch.randn(2, 16, 3, generator=g)
        lse[:, : 12 - 4 * s] = NEG_INF  # rows gain keys at later steps
        lse[:, 15] = NEG_INF  # a row without keys at every step
        out = torch.randn(2, 16, 3, 8, generator=g) * lse.isfinite().unsqueeze(-1)
        steps.append((out, lse))
    out, lse = _merge(steps)
    want_out, want_lse = _reference(steps)
    assert lse.shape == (2, 16, 3)
    torch.testing.assert_close(out.double(), want_out, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(lse.double(), want_lse, atol=1e-6, rtol=1e-6)


def test_merger_finite_rows_unchanged() -> None:
    """Finite rows still merge bit for bit as before the -inf guard."""
    g = torch.Generator().manual_seed(2)
    steps = [
        (torch.randn(2, 32, 4, 16, generator=g), torch.randn(2, 32, 4, generator=g))
        for _ in range(3)
    ]
    out, lse = _merge(steps)
    old_out, old_lse = steps[0][0], steps[0][1].unsqueeze(-1)
    for step_out, step_lse in steps[1:]:
        step_lse = step_lse.unsqueeze(-1)
        old_out = old_out - torch.sigmoid(step_lse - old_lse) * (old_out - step_out)
        old_lse = old_lse - F.logsigmoid(old_lse - step_lse)
    assert torch.equal(out, old_out)
    assert torch.equal(lse, old_lse.squeeze(-1))


# ---------------------------------------------------------------------------
# Ring op LSE shape and differentiability.


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("online", (2, 4, 256)),
        ("python", (2, 256, 4)),
        ("fp32", (2, 256, 4)),
        ("flash", (2, 256, 4)),
        ("auto", (2, 256, 4)),
    ],
)
def test_ring_lse_shape(mode: str, expected: tuple[int, int, int]) -> None:
    assert _ring_lse_shape(torch.Size([2, 256, 4, 64]), mode) == expected


_needs_fa4_gpu = pytest.mark.skipif(
    not (HAS_FLASH4_CUTE and torch.cuda.is_available()),
    reason="needs CUDA and flash_attn.cute",
)


def _skip_unsupported_mode(mode: str) -> None:
    major = torch.cuda.get_device_capability()[0]
    if mode == "online" and major != 9:
        pytest.skip("the online combine kernel is SM90-only")
    if mode == "fp32" and major not in (10, 11):
        pytest.skip("combine_mode='fp32' needs FA4's split-KV forward (SM100)")


@pytest.fixture
def one_rank_group():
    created = not dist.is_initialized()
    if created:
        dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    yield dist.group.WORLD
    if created:
        dist.destroy_process_group()


def _native_meta(rank: int = 0, size: int = 1) -> Flash4RingStepMeta:
    return Flash4RingStepMeta(
        native=NativeRingMask(mode="full", ring_rank=rank, ring_size=size)
    )


@pytest.mark.gpu
@_needs_fa4_gpu
@pytest.mark.parametrize("mode", ["python", "flash", "online", "fp32", "auto"])
def test_ring_op_fake_lse_matches_real(one_rank_group, mode: str) -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    from usp_attention.ring.flash4_ring_attn import (
        _deregister_ring_ctx,
        _register_ring_ctx,
        _RingForwardCtx,
    )

    _skip_unsupported_mode(mode)
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (
        torch.randn(1, 1024, 4, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        for _ in range(3)
    )
    cid = _register_ring_ctx(
        _RingForwardCtx(
            step_meta=_native_meta(),
            process_group=one_rank_group,
            rank_plan=None,
            token_permutation=None,
            restore_permutation=None,
            seq_dim=1,
            input_is_sharded=True,
            gather_output=False,
            use_cuda_comm_stream=True,
            combine_mode=mode,  # type: ignore[arg-type]
            deterministic=False,
            use_symm_mem=False,
            use_symm_mem_rotator=False,
            use_symm_mem_direct=False,
            use_nccl4py_ring=False,
        )
    )
    try:
        real_out, real_lse = torch.ops.usp_attention.ring_flash4_attn(q, k, v, cid)
        with FakeTensorMode() as fake_mode:
            fq, fk, fv = (fake_mode.from_tensor(t) for t in (q, k, v))
            fake_out, fake_lse = torch.ops.usp_attention.ring_flash4_attn(
                fq, fk, fv, cid
            )
    finally:
        _deregister_ring_ctx(cid)
    assert tuple(real_lse.shape) == _ring_lse_shape(q.shape, mode)
    for fake, real in ((fake_out, real_out), (fake_lse, real_lse)):
        assert fake.shape == real.shape
        assert fake.stride() == real.stride()
        assert fake.dtype == real.dtype


@pytest.mark.gpu
@_needs_fa4_gpu
@pytest.mark.parametrize("mode", ["python", "online", "fp32"])
def test_ring_lse_is_not_differentiable(one_rank_group, mode: str) -> None:
    from usp_attention.ring.flash4_ring_attn import ring_flash4_attn

    _skip_unsupported_mode(mode)
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (
        torch.randn(
            1, 1024, 4, 128, device="cuda", dtype=torch.bfloat16, generator=g
        ).requires_grad_()
        for _ in range(3)
    )
    out, lse = ring_flash4_attn(
        q,
        k,
        v,
        process_group=one_rank_group,
        step_meta=_native_meta(),
        input_is_sharded=True,
        combine_mode=mode,  # type: ignore[arg-type]
        return_lse=True,
    )
    assert out.requires_grad
    assert not lse.requires_grad
    (out.float().sum() + lse.sum()).backward()
    assert all(t.grad is not None for t in (q, k, v))


# ---------------------------------------------------------------------------
# fp32 gradient hooks against a fake FA4 (and the probe/sync win).


class _FakeFA4:
    """Stand-in for ``flash_attn.cute.interface``'s backward helpers.

    The preprocess zeroes ``dq_accum``; the conversion writes ``accum *
    scale`` in the output tensor's dtype and records that dtype.
    """

    def __init__(self) -> None:
        self.conversions: list[torch.dtype] = []

        def _bwd_preprocess(out, dout, dpsum, lse, lse_log2, dq_accum, *a, **kw):
            if dq_accum is not None:
                dq_accum.zero_()

        def _bwd_postprocess_convert(accum, output, scale, *a, **kw):
            self.conversions.append(output.dtype)
            output.copy_((accum * scale).to(output.dtype))

        _bwd_preprocess.compile_cache = {}  # type: ignore[attr-defined]
        _bwd_postprocess_convert.compile_cache = {}  # type: ignore[attr-defined]
        self._bwd_preprocess = _bwd_preprocess
        self._bwd_postprocess_convert = _bwd_postprocess_convert
        self.torch2cute_dtype_map = {torch.float16: "f16", torch.bfloat16: "bf16"}


def _fake_step_bwd(fa: _FakeFA4, dq_part: torch.Tensor, *, uses_accum: bool):
    """One ``_flash_attn_bwd`` for dq: through FA4's fp32 accumulator, or
    written directly (FA4's head_dim-256 backward on Blackwell)."""
    if not uses_accum:
        fa._bwd_preprocess(None, None, None, None, None, None)
        return dq_part.clone()
    dq_accum = torch.empty_like(dq_part, dtype=torch.float32)
    fa._bwd_preprocess(None, None, None, None, None, dq_accum)
    dq_accum += dq_part  # the kernel accumulates into the seeded buffer
    dq = torch.empty_like(dq_part)
    fa._bwd_postprocess_convert(dq_accum, dq, 1.0, None, None, 100, "bf16")
    return dq


@pytest.mark.parametrize("uses_accum", [True, False])
def test_ring_dq_sums_every_step(monkeypatch, uses_accum: bool) -> None:
    """With fp32 gradients the ring's dq is the sum of every step's dq, also
    when FA4 writes dq directly (it used to keep only the last step's)."""
    fa = _FakeFA4()
    monkeypatch.setattr(fp32_grad, "_fa", fa)
    g = torch.Generator().manual_seed(0)
    parts = [torch.randn(2, 128, 4, 64, generator=g).bfloat16() for _ in range(4)]
    state = fp32_grad.Fp32GradState()
    dq = None
    for step, part in enumerate(parts):
        last = step == len(parts) - 1
        with fp32_grad.fp32_grad_step(state, is_first=step == 0, is_last=last):
            dq_step = _fake_step_bwd(fa, part, uses_accum=uses_accum)
        assert state.dq_in_accum is uses_accum
        dq = _add_step_dq(dq, dq_step, state, is_last=last)
    want = sum(p.double() for p in parts)
    assert dq is not None
    torch.testing.assert_close(dq.double(), want, atol=3e-2, rtol=1e-2)


def test_ring_dq_without_fp32_grads() -> None:
    g = torch.Generator().manual_seed(1)
    parts = [torch.randn(8, 4, generator=g).bfloat16() for _ in range(3)]
    dq = None
    for i, part in enumerate(parts):
        dq = _add_step_dq(dq, part, None, is_last=i == len(parts) - 1)
    assert dq is not None and dq.dtype == torch.float32
    torch.testing.assert_close(dq, sum(p.float() for p in parts))


def test_fp32_grad_step_rejects_mixed_dq_paths(monkeypatch) -> None:
    fa = _FakeFA4()
    monkeypatch.setattr(fp32_grad, "_fa", fa)
    part = torch.ones(4, 8, dtype=torch.bfloat16)
    state = fp32_grad.Fp32GradState()
    with fp32_grad.fp32_grad_step(state, is_first=True, is_last=False):
        _fake_step_bwd(fa, part, uses_accum=True)
    with pytest.raises(RuntimeError, match="switched"):
        with fp32_grad.fp32_grad_step(state, is_first=False, is_last=True):
            _fake_step_bwd(fa, part, uses_accum=False)


def _fake_gqa_bwd(fa: _FakeFA4, accums: list[torch.Tensor], scale: float):
    """FA4's GQA backward conversions: dq, dk (softmax scale), dv (scale 1)."""
    outs = []
    for accum, s in zip(accums, (scale, scale, 1.0), strict=True):
        out = torch.empty(accum.shape, dtype=torch.bfloat16)
        fa._bwd_postprocess_convert(accum, out, s, None, None, 100, "bf16", 128)
        outs.append(out)
    return outs


@pytest.mark.parametrize("skip_dq", [False, True])
def test_fp32_grad_outputs_probes_dq_only_and_converts_at_exit(
    monkeypatch, skip_dq: bool
) -> None:
    fa = _FakeFA4()
    monkeypatch.setattr(fp32_grad, "_fa", fa)
    g = torch.Generator().manual_seed(0)
    accums = [torch.randn(64, 32, generator=g) * m for m in (3.0, 50.0, 0.01)]
    scale = 0.125
    with fp32_grad.fp32_grad_outputs(skip_dq=skip_dq) as exact:
        grads = _fake_gqa_bwd(fa, accums, scale)
        assert not exact  # converted when the context exits
    # dq: FA4's own conversion (skip_dq) or one bf16 probe; dk / dv need no
    # probe, and the fp16 conversions follow one host sync.
    if skip_dq:
        assert fa.conversions == [torch.bfloat16] + [torch.float16] * 2
    else:
        assert fa.conversions == [torch.bfloat16] + [torch.float16] * 3
    for i, (grad, accum, s) in enumerate(
        zip(grads, accums, (scale, scale, 1.0), strict=True)
    ):
        t, inv = fp32_grad.as_scaled(exact, grad)
        want = accum * s
        if skip_dq and i == 0:
            assert inv == 1.0 and t.dtype == torch.bfloat16
            torch.testing.assert_close(t, want.bfloat16())
            continue
        assert t.dtype == torch.float16
        peak = float(t.float().abs().max())
        assert 2.0**12 < peak < 2.0**15  # scaled to near 2**14
        torch.testing.assert_close(t.float() * inv, want, atol=0.0, rtol=1e-3)


# ---------------------------------------------------------------------------
# Dense paths' fp16 scaling: one host sync, no fp32 temporary (perf wins).


def _old_to_f16(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    amax = float(x.abs().amax()) if x.numel() else 0.0
    if amax == 0.0:
        return x.half(), 1.0
    f = 2.0 ** max(min(math.floor(math.log2(2.0**13 / amax)), 60), -60)
    return (x.float() * f).half(), f


def _wide_range(n: int, seed: int) -> torch.Tensor:
    """bf16 values from 1e-30 to 1e4, so some land in fp16's subnormals."""
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, generator=g) * torch.logspace(-30, 4, n)).bfloat16()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="needs CUDA"
            ),
        ),
    ],
)
def test_to_f16_matches_fp32_route(device: str) -> None:
    g = torch.Generator().manual_seed(0)
    xs = [
        (torch.randn(4096, generator=g) * 3).bfloat16(),
        _wide_range(4096, 1),
        torch.zeros(16, dtype=torch.bfloat16),
        (torch.randn(64, generator=g) * 1e5).bfloat16(),  # factor below 1
        torch.empty(0, dtype=torch.bfloat16),
        torch.randn(3, 5, generator=g),  # fp32 input
    ]
    xs = [x.to(device) for x in xs]
    for x, (got, f) in zip(xs, _to_f16(*xs), strict=True):
        want, want_f = _old_to_f16(x)
        assert f == want_f
        assert got.dtype == torch.float16
        assert torch.equal(got.view(torch.int16), want.view(torch.int16))


def test_scaled_f16_identity_factor() -> None:
    x = _wide_range(256, 2)
    assert torch.equal(
        _scaled_f16(x, 1.0).view(torch.int16), x.half().view(torch.int16)
    )
    assert _f16_factor(0.0) == 1.0


def test_ring_factors_one_rank_match_separate_syncs() -> None:
    g = torch.Generator().manual_seed(0)
    q = (torch.randn(64, 4, 32, generator=g) * 7).bfloat16()
    k = (torch.randn(64, 2, 32, generator=g) * 0.02).bfloat16()
    v = torch.zeros(64, 2, 32, dtype=torch.bfloat16)
    fq, fk, fv = _ring_factors(q, k, v, None)
    assert fq == _old_to_f16(q)[1]
    m = float(k.abs().amax())
    assert fk == 2.0 ** max(min(int(np.floor(np.log2(2.0**13 / m))), 60), -60)
    assert fv == 1.0


# ---------------------------------------------------------------------------
# Two-GPU ring backward with fp32 gradients against an fp32 reference. On
# Blackwell at head_dim 256 FA4 writes dq directly; this caught the ring
# keeping only the last step's dq.


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ring_grads_worker(
    rank: int, world: int, port: int, head_dim: int, kv_heads: int
) -> None:
    from usp_attention.ring.flash4_ring_attn import ring_flash4_attn

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        local, heads = 1024, 4
        seq = local * world
        g = torch.Generator(device="cuda").manual_seed(0)
        q, dout = (
            torch.randn(1, seq, heads, head_dim, device="cuda", generator=g)
            for _ in range(2)
        )
        k, v = (
            torch.randn(1, seq, kv_heads, head_dim, device="cuda", generator=g)
            for _ in range(2)
        )
        q, k, v, dout = (t.bfloat16() for t in (q, k, v, dout))
        rows = slice(rank * local, (rank + 1) * local)
        ql, kl, vl = (t[:, rows].clone().requires_grad_() for t in (q, k, v))
        out = ring_flash4_attn(
            ql,
            kl,
            vl,
            process_group=dist.group.WORLD,
            step_meta=_native_meta(rank, world),
            input_is_sharded=True,
            combine_mode="python",
            fp32_grads=True,
        )
        out.backward(dout[:, rows])

        rep = heads // kv_heads
        qr, kr, vr = (t.float().requires_grad_() for t in (q, k, v))
        ref = F.scaled_dot_product_attention(
            qr.transpose(1, 2),
            kr.repeat_interleave(rep, dim=2).transpose(1, 2),
            vr.repeat_interleave(rep, dim=2).transpose(1, 2),
        ).transpose(1, 2)
        ref.backward(dout.float())
        for name, got, want in (
            ("dq", ql.grad, qr.grad[:, rows]),
            ("dk", kl.grad, kr.grad[:, rows]),
            ("dv", vl.grad, vr.grad[:, rows]),
        ):
            err = float((got.float() - want).abs().mean() / want.abs().mean())
            assert err < 2e-2, f"rank {rank} {name}: mean relative error {err:.3g}"
    finally:
        dist.destroy_process_group()


@pytest.mark.gpu
@pytest.mark.skipif(
    not (HAS_FLASH4_CUTE and torch.cuda.is_available())
    or torch.cuda.device_count() < 2,
    reason="needs two CUDA GPUs and flash_attn.cute",
)
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("kv_heads", [4, 2])
def test_ring_fp32_grads_match_reference(head_dim: int, kv_heads: int) -> None:
    mp.spawn(
        _ring_grads_worker,
        args=(2, _free_port(), head_dim, kv_heads),
        nprocs=2,
        join=True,
    )
