# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""CPU tests for USP process-group layout and the Ulysses all-to-all (gloo)."""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from usp_attention.distributed import USPGroups, seq_all_to_all
from usp_attention.usp import pad_packed_layout


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(rank: int, world: int, port: int, fn_name: str, args: tuple) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        globals()[fn_name](rank, world, *args)
        # Nonmembers return early from new_group; keep the rendezvous store
        # alive until every rank has finished creating/checking its groups.
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _spawn(world: int, fn_name: str, *args) -> None:
    mp.spawn(_run, args=(world, _free_port(), fn_name, args), nprocs=world, join=True)


def _check_group_layout(rank: int, world: int, ring: int, ulysses: int) -> None:
    groups = USPGroups.create(ring_size=ring, ulysses_size=ulysses)
    cp = ring * ulysses
    base = (rank // cp) * cp
    assert groups.cp_rank == rank - base
    assert groups.cp_rank == groups.ring_rank * ulysses + groups.ulysses_rank
    if ulysses > 1:
        members = dist.get_process_group_ranks(groups.ulysses_group)
        assert members == [
            base + groups.ring_rank * ulysses + u for u in range(ulysses)
        ]
    else:
        assert groups.ulysses_group is None
    if ring > 1:
        members = dist.get_process_group_ranks(groups.ring_group)
        assert members == [
            base + r * ulysses + groups.ulysses_rank for r in range(ring)
        ]
    else:
        assert groups.ring_group is None


@pytest.mark.parametrize(
    ("world", "ring", "ulysses"),
    [(4, 2, 2), (4, 4, 1), (4, 1, 4), (8, 2, 2)],
)
def test_group_layout(world: int, ring: int, ulysses: int) -> None:
    _spawn(world, "_check_group_layout", ring, ulysses)


def _check_all_to_all(rank: int, world: int) -> None:
    batch, local_seq, heads, dim = 2, 3, 4 * world, 5
    torch.manual_seed(0)
    full = torch.randn(batch, local_seq * world, heads, dim, dtype=torch.float64)
    x = full[:, rank * local_seq : (rank + 1) * local_seq].clone().requires_grad_(True)
    group = dist.new_group(list(range(world)))

    y = seq_all_to_all(x, group, scatter_dim=2, gather_dim=1)
    head_chunk = heads // world
    expected = full[:, :, rank * head_chunk : (rank + 1) * head_chunk]
    torch.testing.assert_close(y, expected)

    back = seq_all_to_all(y, group, scatter_dim=1, gather_dim=2)
    torch.testing.assert_close(back, x)

    # d/dx sum(w * a2a(x)) == a2a^-1(w): check the backward is the inverse exchange.
    weight = torch.randn_like(y)
    (y * weight).sum().backward()
    expected_grad = seq_all_to_all(weight, group, scatter_dim=1, gather_dim=2)
    torch.testing.assert_close(x.grad, expected_grad)


def test_seq_all_to_all_roundtrip_and_grad() -> None:
    _spawn(4, "_check_all_to_all")


def test_pad_packed_layout() -> None:
    docs, splits, modes = pad_packed_layout(
        [100, 60], [40, 60, 60], ["causal", "full", "causal"], 128
    )
    assert sum(docs) == 256 and sum(splits) == 256
    # Its own causal document: isolated, and all-causal layouts stay all-causal.
    assert docs[-1] == 96 and splits[-1] == 96 and modes[-1] == "causal"
    assert pad_packed_layout([128], [128], ["full"], 128) == ([128], [128], ["full"])
