# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Process-group setup for Ulysses x ring context parallelism.

The flat context-parallel (CP) group of ``ring_size * ulysses_size`` ranks is
laid out with Ulysses innermost: CP rank ``c = ring_rank * ulysses_size +
ulysses_rank``. Each Ulysses group is a contiguous block of ranks (typically
one NVLink domain), and each ring group strides across Ulysses groups. Ranks
outside one CP group are data-parallel replicas.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


def init_distributed(*, backend: str = "nccl") -> torch.device:
    """Initialize the default process group from torchrun environment variables.

    Falls back to a single-process group (``RANK=0``, ``WORLD_SIZE=1``) when the
    script is not launched by torchrun. Sets the current CUDA device from
    ``LOCAL_RANK``.

    Returns:
        The CUDA device for this rank.
    """
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, device_id=device)
    return device


@dataclass(kw_only=True, slots=True)
class USPGroups:
    """Process groups and coordinates for one rank of a Ulysses x ring CP mesh.

    Attributes:
        ring_size: Ranks along the ring (KV-rotation) axis.
        ulysses_size: Ranks along the Ulysses (head all-to-all) axis.
        ring_rank: This rank's coordinate on the ring axis.
        ulysses_rank: This rank's coordinate on the Ulysses axis.
        cp_rank: This rank's position in the flat CP group.
        ring_group: Group over the ranks sharing this rank's Ulysses coordinate.
        ulysses_group: Group over the ranks sharing this rank's ring coordinate.
        cp_group: Group over all ``ring_size * ulysses_size`` CP ranks.
    """

    ring_size: int
    ulysses_size: int
    ring_rank: int
    ulysses_rank: int
    cp_rank: int
    ring_group: dist.ProcessGroup | None
    ulysses_group: dist.ProcessGroup | None
    cp_group: dist.ProcessGroup | None

    @property
    def cp_size(self) -> int:
        return self.ring_size * self.ulysses_size

    @classmethod
    def create(
        cls,
        *,
        ring_size: int,
        ulysses_size: int,
        symm_mem_copy_engine: bool = False,
    ) -> USPGroups:
        """Create ring, Ulysses and CP groups over the default process group.

        Every rank in the default group must call this collectively. The world
        size must be a multiple of ``ring_size * ulysses_size``.

        Args:
            ring_size: Ring degree.
            ulysses_size: Ulysses degree.
            symm_mem_copy_engine: Configure the ring groups for symmetric-memory
                KV transport over the NCCL copy engine (zero-CTA policy) and set
                torch's symmetric-memory backend to NCCL.

        Returns:
            The groups and coordinates for the calling rank.

        Raises:
            ValueError: If the world size is not divisible by the CP size.
        """
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        cp_size = ring_size * ulysses_size
        if ring_size < 1 or ulysses_size < 1 or world_size % cp_size != 0:
            raise ValueError(
                f"World size {world_size} is not divisible by "
                f"ring_size * ulysses_size = {ring_size} * {ulysses_size}."
            )

        ring_options = None
        if symm_mem_copy_engine:
            from torch.distributed import _symmetric_memory

            _symmetric_memory.set_backend("NCCL")
            ring_options = dist.ProcessGroupNCCL.Options()
            ring_options.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO

        cp_index, cp_rank = divmod(rank, cp_size)
        ring_rank, ulysses_rank = divmod(cp_rank, ulysses_size)

        my_ring = my_ulysses = my_cp = None
        for base in range(0, world_size, cp_size):
            cp_ranks = list(range(base, base + cp_size))
            group = dist.new_group(cp_ranks) if cp_size > 1 else None
            if base == cp_index * cp_size:
                my_cp = group
            for u in range(ulysses_size):
                ranks = [base + r * ulysses_size + u for r in range(ring_size)]
                group = (
                    dist.new_group(ranks, pg_options=ring_options)
                    if ring_size > 1
                    else None
                )
                if rank in ranks:
                    my_ring = group
            for r in range(ring_size):
                ranks = [base + r * ulysses_size + u for u in range(ulysses_size)]
                group = dist.new_group(ranks) if ulysses_size > 1 else None
                if rank in ranks:
                    my_ulysses = group

        return cls(
            ring_size=ring_size,
            ulysses_size=ulysses_size,
            ring_rank=ring_rank,
            ulysses_rank=ulysses_rank,
            cp_rank=cp_rank,
            ring_group=my_ring,
            ulysses_group=my_ulysses,
            cp_group=my_cp,
        )


class _SeqAllToAll(torch.autograd.Function):
    """Differentiable all-to-all that scatters ``scatter_dim`` and gathers ``gather_dim``.

    Each rank splits its input into ``world`` chunks along ``scatter_dim``,
    sends chunk ``i`` to rank ``i``, and concatenates the received chunks along
    ``gather_dim`` in rank order. The backward pass is the inverse exchange.
    """

    @staticmethod
    def forward(ctx, x, group, scatter_dim: int, gather_dim: int):
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        return _all_to_all(x, group, scatter_dim, gather_dim)

    @staticmethod
    def backward(ctx, grad):
        return (
            _all_to_all(grad, ctx.group, ctx.gather_dim, ctx.scatter_dim),
            None,
            None,
            None,
        )


def _all_to_all(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
) -> torch.Tensor:
    world = dist.get_world_size(group)
    if x.shape[scatter_dim] % world != 0:
        raise ValueError(
            f"Dimension {scatter_dim} of size {x.shape[scatter_dim]} is not "
            f"divisible by the all-to-all group size {world}."
        )
    # Peer chunks along a new leading dim. With batch 1 and the sequence as
    # the scatter (or gather) dim, these reshapes are views, so only the
    # head-dim side of each exchange pays for a copy.
    send = x.unflatten(scatter_dim, (world, -1)).movedim(scatter_dim, 0).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return recv.movedim(0, gather_dim).flatten(gather_dim, gather_dim + 1).contiguous()


def seq_all_to_all(
    x: torch.Tensor,
    group: dist.ProcessGroup | None,
    *,
    scatter_dim: int,
    gather_dim: int,
) -> torch.Tensor:
    """Ulysses all-to-all; identity when ``group`` is ``None``."""
    if group is None or dist.get_world_size(group) == 1:
        return x
    return _SeqAllToAll.apply(x, group, scatter_dim, gather_dim)
