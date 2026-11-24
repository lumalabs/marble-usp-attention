# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Correctness tests for online Flash Attention 4 (SM90).

Compares the online chunked-KV path against a single full-sequence FA4 call
for varying numbers of KV splits (2, 4, 8) and sequence lengths up to 131072
tokens. Each (num_chunks, seqlen_k) combination is a separate test case so
failures at one setting do not mask failures at another.

Run (from `projects/usp_attention`):

    source .venv/bin/activate
    python -m pytest tests/online_flash/test_online_flash.py -v

Environment overrides:

    ONLINE_FA4_MAX_SEQLEN_K  — cap the longest seqlen_k swept (default 131072)
    ONLINE_FA4_SEQLEN_Q      — override the fixed seqlen_q (default 1024)
    ONLINE_FA4_BATCH         — batch size (default 1)
    ONLINE_FA4_NUM_HEADS     — number of heads (default 4)
    ONLINE_FA4_HEAD_DIM      — head dim (default 128)
    ONLINE_FA4_OUT_ATOL      — output abs tolerance (default 1.5e-3)
    ONLINE_FA4_OUT_RTOL      — output rel tolerance (default 1e-2)
    ONLINE_FA4_LSE_ATOL      — LSE abs tolerance (default 2e-5)
    ONLINE_FA4_LSE_RTOL      — LSE rel tolerance (default 1e-4)
"""

from __future__ import annotations

import os
import unittest

import pytest
import torch

pytest.importorskip("cutlass")

from usp_attention.online_flash.interface import (
    offline_flash_attn_func,
    online_flash_attn_chunks,
)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


MAX_SEQLEN_K = _env_int("ONLINE_FA4_MAX_SEQLEN_K", 131072)
SEQLEN_Q = _env_int("ONLINE_FA4_SEQLEN_Q", 1024)
BATCH = _env_int("ONLINE_FA4_BATCH", 1)
NUM_HEADS = _env_int("ONLINE_FA4_NUM_HEADS", 4)
HEAD_DIM = _env_int("ONLINE_FA4_HEAD_DIM", 128)
# Tolerances chosen empirically (see _probe_tolerance.py) on H100 bf16 across the
# full sweep with ~2x safety margin. Output is dominated by bf16 quantization;
# LSE is much more accurate because it's only written on the last call.
OUT_ATOL = _env_float("ONLINE_FA4_OUT_ATOL", 1.5e-3)
OUT_RTOL = _env_float("ONLINE_FA4_OUT_RTOL", 1e-2)
LSE_ATOL = _env_float("ONLINE_FA4_LSE_ATOL", 2e-5)
LSE_RTOL = _env_float("ONLINE_FA4_LSE_RTOL", 1e-4)

_BASE_SEQLENS_K = [s for s in (2048, 8192, 32768, 131072) if s <= MAX_SEQLEN_K]
_NUM_CHUNKS_SWEEP = (2, 4, 8)

_SM90_UNAVAILABLE_REASON: str | None
if not torch.cuda.is_available():
    _SM90_UNAVAILABLE_REASON = "CUDA is not available"
else:
    _major, _minor = torch.cuda.get_device_capability()
    if _major != 9:
        _SM90_UNAVAILABLE_REASON = (
            f"Online FA4 forward kernel requires SM 9.x (Hopper); "
            f"current device is SM {_major}.{_minor}"
        )
    else:
        _SM90_UNAVAILABLE_REASON = None


def _make_qkv(
    *,
    batch: int,
    seqlen_q: int,
    seqlen_k: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(
        batch, seqlen_q, num_heads, head_dim, dtype=dtype, device=device, generator=g
    )
    k = torch.randn(
        batch, seqlen_k, num_heads, head_dim, dtype=dtype, device=device, generator=g
    )
    v = torch.randn(
        batch, seqlen_k, num_heads, head_dim, dtype=dtype, device=device, generator=g
    )
    return q, k, v


@unittest.skipIf(_SM90_UNAVAILABLE_REASON is not None, _SM90_UNAVAILABLE_REASON or "")
class OnlineFlashAttention4Test(unittest.TestCase):
    """Sweep num_chunks ∈ {2,4,8} x seqlen_k up to ONLINE_FA4_MAX_SEQLEN_K."""

    dtype = torch.bfloat16
    device = "cuda"

    def _run(self, *, num_chunks: int, seqlen_k: int) -> None:
        q, k, v = _make_qkv(
            batch=BATCH,
            seqlen_q=SEQLEN_Q,
            seqlen_k=seqlen_k,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            dtype=self.dtype,
            device=self.device,
            seed=0,
        )
        scale = HEAD_DIM**-0.5

        # Reference: single full-sequence FA4 call.
        ref_out, ref_lse = offline_flash_attn_func(
            q, k, v, softmax_scale=scale, return_lse=True
        )
        self.assertIsNotNone(
            ref_lse, "offline_flash_attn_func should return an LSE here"
        )
        assert ref_lse is not None  # narrow for type checker

        self.assertEqual(
            seqlen_k % num_chunks, 0, "seqlen_k must divide evenly by num_chunks"
        )
        k_chunks = list(k.chunk(num_chunks, dim=1))
        v_chunks = list(v.chunk(num_chunks, dim=1))

        online_out, online_lse = online_flash_attn_chunks(
            q, k_chunks, v_chunks, softmax_scale=scale, return_lse=True
        )
        self.assertIsNotNone(online_lse)
        assert online_lse is not None

        ctx = f"(num_chunks={num_chunks}, seqlen_k={seqlen_k})"
        torch.testing.assert_close(
            online_out.float(),
            ref_out.float(),
            rtol=OUT_RTOL,
            atol=OUT_ATOL,
            msg=f"online_out vs ref_out {ctx}",
        )
        torch.testing.assert_close(
            online_lse.float(),
            ref_lse.float(),
            rtol=LSE_RTOL,
            atol=LSE_ATOL,
            msg=f"online_lse vs ref_lse {ctx}",
        )


def _generate_test_methods() -> None:
    """Inject one `test_chunks{C}_seqlenk{S}` method per (num_chunks, seqlen_k)."""
    for num_chunks in _NUM_CHUNKS_SWEEP:
        for seqlen_k in _BASE_SEQLENS_K:
            if seqlen_k % num_chunks != 0:
                continue
            method_name = f"test_chunks{num_chunks}_seqlenk{seqlen_k}"

            def _method(
                self: OnlineFlashAttention4Test, *, _nc=num_chunks, _sk=seqlen_k
            ) -> None:
                self._run(num_chunks=_nc, seqlen_k=_sk)

            _method.__name__ = method_name
            setattr(OnlineFlashAttention4Test, method_name, _method)


_generate_test_methods()


if __name__ == "__main__":
    unittest.main(verbosity=2)
