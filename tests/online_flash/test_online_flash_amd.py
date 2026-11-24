# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""GQA correctness and benchmark coverage for AMD CDNA3 online Flash Attention.

Run from projects/usp_attention with ROCm PyTorch, hipcc, and a gfx942 GPU:

    uv run --frozen python src/usp_attention/online_flash_amd/setup.py build_ext --inplace
    uv run --frozen pytest tests/online_flash/test_online_flash_amd.py -v

Set OF4AMD_RUN_BENCHMARKS=1 to run the large Omni-shape benchmark matrix.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable

import pytest
import torch

try:
    from usp_attention.online_flash_amd import (
        forward,
        online_flash_attn_chunks,
        online_flash_attn_func,
    )
    from usp_attention.online_flash_amd._ext import _C
except ImportError as exc:
    pytest.skip(
        f"AMD online flash extension is not built: {exc}", allow_module_level=True
    )

HEAD_DIM = 128
ATOL = float(os.environ.get("OF4AMD_OUT_ATOL", "0.004"))
RTOL = float(os.environ.get("OF4AMD_OUT_RTOL", "0.01"))
requires_rocm = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="requires ROCm PyTorch and a GPU",
)


def _rand(shape: tuple[int, ...], seed: int, device: str = "cuda") -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)


def _qkv(
    layout: str,
    query_heads: int,
    kv_heads: int,
    seqlen_q: int,
    seqlen_k: int,
    batch: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if layout == "bshd":
        q_shape = (batch, seqlen_q, query_heads, HEAD_DIM)
        kv_shape = (batch, seqlen_k, kv_heads, HEAD_DIM)
        chunk_dim = 1
    else:
        q_shape = (batch, query_heads, seqlen_q, HEAD_DIM)
        kv_shape = (batch, kv_heads, seqlen_k, HEAD_DIM)
        chunk_dim = 2
    return _rand(q_shape, 1), _rand(kv_shape, 2), _rand(kv_shape, 3), chunk_dim


def _reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layout: str
) -> torch.Tensor:
    """Compute an independent fp32 PyTorch GQA reference."""
    if layout == "bshd":
        query_heads, kv_heads = q.shape[2], k.shape[2]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    else:
        query_heads, kv_heads = q.shape[1], k.shape[1]
    groups = query_heads // kv_heads
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(HEAD_DIM)
    result = torch.matmul(torch.softmax(scores, dim=-1), v.float())
    return result.transpose(1, 2) if layout == "bshd" else result


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual.float(), expected.float(), atol=ATOL, rtol=RTOL)


@requires_rocm
@pytest.mark.parametrize("layout", ["bshd", "bhsd"])
@pytest.mark.parametrize(
    ("query_heads", "kv_heads"),
    [(8, 8), (8, 4), (8, 2), (8, 1), (32, 8)],
    ids=["mha", "gqa2", "gqa4", "mqa8", "omni_gqa4"],
)
def test_forward_matches_independent_reference(
    layout: str, query_heads: int, kv_heads: int
) -> None:
    """Cover MHA, GQA ratios 2/4/8, MQA, and representative Omni heads."""
    q, k, v, _ = _qkv(layout, query_heads, kv_heads, 97, 131)
    actual = forward(q, k, v, layout=layout)
    assert actual.shape == q.shape
    _assert_close(actual, _reference(q, k, v, layout))


@requires_rocm
@pytest.mark.parametrize("layout", ["bshd", "bhsd"])
@pytest.mark.parametrize("num_chunks", [1, 2, 4])
def test_online_chunks_matches_full_reference(layout: str, num_chunks: int) -> None:
    """One and multiple chunks support GQA cross-attention."""
    q, k, v, chunk_dim = _qkv(layout, 8, 2, 83, 137)
    actual = online_flash_attn_chunks(
        q,
        list(k.chunk(num_chunks, dim=chunk_dim)),
        list(v.chunk(num_chunks, dim=chunk_dim)),
        layout=layout,
    )
    _assert_close(actual, _reference(q, k, v, layout))


@requires_rocm
@pytest.mark.parametrize("layout", ["bshd", "bhsd"])
def test_explicit_state_threading_uses_uneven_mqa_chunks(layout: str) -> None:
    """Explicit first/middle/last state threading handles uneven MQA chunks."""
    q, k, v, chunk_dim = _qkv(layout, 8, 1, 71, 113, batch=2)
    k_chunks = torch.split(k, (17, 29, 67), dim=chunk_dim)
    v_chunks = torch.split(v, (17, 29, 67), dim=chunk_dim)
    out = state = None
    for index, (k_chunk, v_chunk) in enumerate(zip(k_chunks, v_chunks)):
        out, state = online_flash_attn_func(
            q,
            k_chunk,
            v_chunk,
            out=out,
            state=state,
            is_online_first=index == 0,
            is_online_last=index == len(k_chunks) - 1,
            layout=layout,
        )
    assert out is not None
    _assert_close(out, _reference(q, k, v, layout))


@requires_rocm
@pytest.mark.parametrize("layout", ["bshd", "bhsd"])
@pytest.mark.parametrize("num_chunks", [1, 4, 7])
def test_private_forward_online_matches_reference(layout: str, num_chunks: int) -> None:
    """The private entry point supports uneven internal GQA splits."""
    q, k, v, _ = _qkv(layout, 8, 2, 65, 100)
    _assert_close(
        _C.forward_online(q, k, v, num_chunks, None, "rtne", layout),
        _reference(q, k, v, layout),
    )


@requires_rocm
@pytest.mark.parametrize("round_mode", ["rtne", "rtna", "rtz"])
def test_all_rounding_modes_support_gqa(round_mode: str) -> None:
    """Every bf16 accumulator packing mode supports GQA."""
    q, k, v, _ = _qkv("bshd", 8, 2, 49, 73)
    _assert_close(forward(q, k, v, round_mode=round_mode), _reference(q, k, v, "bshd"))


@requires_rocm
@pytest.mark.parametrize("layout", ["bshd", "bhsd"])
def test_caller_provided_output_buffer(layout: str) -> None:
    """Full and online GQA write the exact caller-provided tensor."""
    q, k, v, chunk_dim = _qkv(layout, 8, 2, 64, 96)
    expected = _reference(q, k, v, layout)
    out = torch.empty_like(q)
    assert forward(q, k, v, out=out, layout=layout).data_ptr() == out.data_ptr()
    _assert_close(out, expected)
    k_chunks, v_chunks = k.chunk(2, dim=chunk_dim), v.chunk(2, dim=chunk_dim)
    online_out = torch.empty_like(q)
    returned, state = online_flash_attn_func(
        q, k_chunks[0], v_chunks[0], out=online_out, is_online_first=True, layout=layout
    )
    returned, _ = online_flash_attn_func(
        q,
        k_chunks[1],
        v_chunks[1],
        out=returned,
        state=state,
        is_online_last=True,
        layout=layout,
    )
    assert returned.data_ptr() == online_out.data_ptr()
    _assert_close(returned, expected)


@requires_rocm
@pytest.mark.parametrize(("query_heads", "kv_heads"), [(2, 4), (6, 4)])
def test_invalid_gqa_head_relationship(query_heads: int, kv_heads: int) -> None:
    """Reject Hq < Hkv and non-divisible head counts."""
    q, k, v, _ = _qkv("bshd", query_heads, kv_heads, 16, 16)
    with pytest.raises(ValueError, match="GQA head mismatch"):
        forward(q, k, v)


@requires_rocm
@pytest.mark.parametrize("mismatch", ["heads", "sequence"])
def test_k_v_shape_mismatch(mismatch: str) -> None:
    """Reject K/V head-count and sequence-length mismatches separately."""
    q, k, _, _ = _qkv("bshd", 8, 2, 16, 16)
    shape = (1, 16, 1, HEAD_DIM) if mismatch == "heads" else (1, 15, 2, HEAD_DIM)
    with pytest.raises(ValueError, match="v shape mismatch"):
        forward(q, k, _rand(shape, 9))


@requires_rocm
@pytest.mark.parametrize("kind", ["shape", "dtype", "device", "layout"])
def test_invalid_output_buffer(kind: str) -> None:
    """Reject invalid output shape, dtype, device, and contiguity."""
    q, k, v, _ = _qkv("bshd", 8, 2, 16, 16)
    if kind == "shape":
        out = torch.empty((1, 15, 8, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    elif kind == "dtype":
        out = torch.empty(q.shape, dtype=torch.float32, device="cuda")
    elif kind == "device":
        out = torch.empty(q.shape, dtype=torch.bfloat16)
    else:
        out = torch.empty(
            (1, 8, 16, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
        ).transpose(1, 2)
    with pytest.raises((ValueError, RuntimeError)):
        forward(q, k, v, out=out)


@requires_rocm
def test_invalid_and_undersized_online_state() -> None:
    """Reject malformed, wrong-dtype, and undersized online state."""
    q, k, v, _ = _qkv("bshd", 8, 2, 16, 32)
    out, state = online_flash_attn_func(q, k[:, :16], v[:, :16], is_online_first=True)
    bad_states = [
        torch.empty(state.numel() - 1, dtype=torch.uint8, device="cuda"),
        torch.empty(state.shape, dtype=torch.bfloat16, device="cuda"),
    ]
    for bad_state in bad_states:
        with pytest.raises(ValueError, match="invalid online state"):
            online_flash_attn_func(
                q, k[:, 16:], v[:, 16:], out=out, state=bad_state, is_online_last=True
            )


@requires_rocm
@pytest.mark.parametrize(
    "kind", ["batch", "query_length", "query_heads", "layout", "device"]
)
def test_state_reuse_rejects_different_query_metadata(kind: str) -> None:
    """Bind opaque state to batch, query length/heads, layout, and device."""
    q, k, v, _ = _qkv("bshd", 8, 2, 16, 16)
    _, state = online_flash_attn_func(q, k, v, is_online_first=True)
    if kind == "layout":
        next_q, next_k, next_v, _ = _qkv("bhsd", 8, 2, 16, 16)
        layout = "bhsd"
    else:
        values = {
            "batch": (8, 2, 16, 2),
            "query_length": (8, 2, 17, 1),
            "query_heads": (4, 2, 16, 1),
        }
        query_heads, kv_heads, seqlen_q, batch = values.get(kind, (8, 2, 16, 1))
        next_q, next_k, next_v, _ = _qkv(
            "bshd", query_heads, kv_heads, seqlen_q, 16, batch
        )
        layout = "bshd"
    bad_state = state.cpu() if kind == "device" else state
    with pytest.raises(ValueError, match="invalid online state"):
        online_flash_attn_func(
            next_q,
            next_k,
            next_v,
            out=torch.empty_like(next_q),
            state=bad_state,
            is_online_last=True,
            layout=layout,
        )


def _benchmark(call: Callable[[], torch.Tensor], repeats: int = 10) -> float:
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(repeats):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


@requires_rocm
@pytest.mark.skipif(
    os.environ.get("OF4AMD_RUN_BENCHMARKS") != "1",
    reason="set OF4AMD_RUN_BENCHMARKS=1 for gfx942 benchmarks",
)
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("seqlen_q", [1024, 4096])
@pytest.mark.parametrize("seqlen_k", [4096, 8192])
@pytest.mark.parametrize("num_chunks", [1, 2, 4])
def test_benchmark_omni_gqa(
    batch: int, seqlen_q: int, seqlen_k: int, num_chunks: int, record_property
) -> None:
    """Benchmark native GQA, expanded MHA, and AITER when installed."""
    q, k, v, chunk_dim = _qkv("bshd", 32, 8, seqlen_q, seqlen_k, batch)
    k_chunks, v_chunks = (
        k.chunk(num_chunks, dim=chunk_dim),
        v.chunk(num_chunks, dim=chunk_dim),
    )
    native_ms = _benchmark(lambda: online_flash_attn_chunks(q, k_chunks, v_chunks))
    expanded_k, expanded_v = (
        k.repeat_interleave(4, dim=2),
        v.repeat_interleave(4, dim=2),
    )
    expanded_ms = _benchmark(lambda: forward(q, expanded_k, expanded_v))
    record_property("native_gqa_ms", native_ms)
    record_property(
        "native_gqa_tflops",
        4 * batch * 32 * seqlen_q * seqlen_k * HEAD_DIM / (native_ms * 1e9),
    )
    record_property("expanded_mha_ms", expanded_ms)
    try:
        import aiter
    except ImportError:
        return
    record_property("aiter_gqa_ms", _benchmark(lambda: aiter.flash_attn_func(q, k, v)))


@requires_rocm
def test_omni_gqa_matches_native_sdpa_when_supported() -> None:
    """Compare representative Omni heads with PyTorch native enable_gqa."""
    q, k, v, _ = _qkv("bshd", 32, 8, 97, 131)
    try:
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            enable_gqa=True,
        ).transpose(1, 2)
    except (RuntimeError, TypeError) as error:
        pytest.skip(f"installed PyTorch SDPA GQA is unavailable: {error}")
    _assert_close(forward(q, k, v), expected)


@requires_rocm
def test_omni_gqa_matches_aiter_when_installed() -> None:
    """Compare representative Omni heads with AITER GQA."""
    aiter = pytest.importorskip("aiter")
    q, k, v, _ = _qkv("bshd", 32, 8, 97, 131)
    _assert_close(forward(q, k, v), aiter.flash_attn_func(q, k, v))
