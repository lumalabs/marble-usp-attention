# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""The striped native ring reproduces the causal mask exactly (CPU)."""

from __future__ import annotations

import pytest
import torch

from usp_attention.ring.flash4_ring_attn import NativeRingMask


def _native_step_mask(causal: bool, right: int | None, n: int) -> torch.Tensor:
    i = torch.arange(n)
    if causal:
        return i[None, :] <= i[:, None]
    if right == -1:
        return i[None, :] < i[:, None]
    return torch.ones(n, n, dtype=torch.bool)


@pytest.mark.parametrize("ring_size", [2, 4, 8])
def test_striped_steps_match_causal_mask(ring_size: int) -> None:
    seq_len = 64 * ring_size
    perm = torch.arange(seq_len).view(-1, ring_size).T.reshape(-1)
    stripes = perm.view(ring_size, -1)  # global positions held by each ring rank
    for rank in range(ring_size):
        meta = NativeRingMask(
            mode="causal_striped", ring_rank=rank, ring_size=ring_size
        )
        for step in range(ring_size):
            owner = meta.kv_owner(step)
            q_pos, k_pos = stripes[rank], stripes[owner]
            expected = k_pos[None, :] <= q_pos[:, None]
            causal, right = meta.step_mask(step)
            assert torch.equal(_native_step_mask(causal, right, len(q_pos)), expected)


def test_striped_permutation_keeps_stripe_order() -> None:
    ring_size, seq_len = 4, 32
    perm = torch.arange(seq_len).view(-1, ring_size).T.reshape(-1)
    assert sorted(perm.tolist()) == list(range(seq_len))
    for r, stripe in enumerate(perm.view(ring_size, -1)):
        assert torch.equal(stripe, torch.arange(r, seq_len, ring_size))


def test_full_mode_is_full_every_step() -> None:
    meta = NativeRingMask(mode="full", ring_rank=1, ring_size=4)
    assert all(meta.step_mask(step) == (False, None) for step in range(4))
