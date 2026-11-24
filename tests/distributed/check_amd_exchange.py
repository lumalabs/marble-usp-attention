# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Check fused AMD redistribution, including batches and multiple local KV heads.

PYTHONPATH=src .venv/bin/python -m torch.distributed.run --standalone \
    --nproc-per-node=8 tests/distributed/check_amd_exchange.py
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from usp_attention.distributed import USPGroups, init_distributed, seq_all_to_all
from usp_attention.online_flash_amd.distributed import _qkv_all_to_all
from usp_attention.online_flash_amd.exchange import gather_qkv


@torch.no_grad()
def main():
    device = init_distributed()
    world = dist.get_world_size()
    torch.manual_seed(123 + dist.get_rank())
    for u in (1, 2, 4):
        if world % u:
            continue
        groups = USPGroups.create(ring_size=world // u, ulysses_size=u)
        q = torch.randn(2, 17, 56, 128, device=device, dtype=torch.bfloat16)[:, :, ::2]
        k = torch.randn(2, 17, 8, 128, device=device, dtype=torch.bfloat16)[:, :, ::2]
        v = torch.randn_like(k)
        expected = tuple(
            seq_all_to_all(x, groups.ulysses_group, scatter_dim=2, gather_dim=1)
            for x in (q, k, v)
        )
        actual = _qkv_all_to_all(q, k, v, groups)
        for x, y in zip(actual, expected):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
        actual = gather_qkv(q, k, v, groups)
        torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
        for x, y in zip(actual[1:], expected[1:]):
            if groups.ring_size > 1:
                y = y.contiguous()
                parts = [torch.empty_like(y) for _ in range(groups.ring_size)]
                dist.all_gather(parts, y, group=groups.ring_group)
                y = torch.cat(parts, dim=1)
            torch.testing.assert_close(x, y, atol=0, rtol=0)
        if dist.get_rank() == 0:
            print(f"passed: ring={groups.ring_size}, ulysses={u}, batch=2", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
