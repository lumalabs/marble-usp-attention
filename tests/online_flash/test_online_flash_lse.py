# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""The online kernel's LSE exists only on the last call.

Earlier calls write the running max / sum, never the LSE, so asking for an LSE
there must fail instead of returning an uninitialized buffer. The rejection
runs before any kernel work, so that test needs no GPU.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cutlass")
pytest.importorskip("flash_attn.cute")

from usp_attention.online_flash.interface import online_flash_attn_func  # noqa: E402


def test_return_lse_before_last_call_is_rejected() -> None:
    q = torch.zeros(1, 128, 2, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="is_online_last"):
        online_flash_attn_func(q, q, q, is_online_first=True, return_lse=True)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="the online kernel targets SM90",
)
def test_lse_only_on_last_call() -> None:
    gen = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (
        torch.randn(1, 1024, 4, 128, device="cuda", dtype=torch.bfloat16, generator=gen)
        for _ in range(3)
    )
    (k0, k1), (v0, v1) = k.chunk(2, dim=1), v.chunk(2, dim=1)
    # A caller-provided LSE buffer is ignored before the last call.
    out, lse, max_, sum_ = online_flash_attn_func(
        q, k0, v0, lse=torch.empty(1, 4, 1024, device="cuda"), is_online_first=True
    )
    assert lse is None
    out, lse, _, _ = online_flash_attn_func(
        q, k1, v1, out=out, max_=max_, sum_=sum_, is_online_last=True
    )
    assert lse is not None and torch.isfinite(lse).all()
