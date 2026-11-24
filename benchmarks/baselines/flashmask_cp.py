# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-FileCopyrightText: 2025 Tri Dao
# SPDX-FileCopyrightText: 2026 The FlashMask Authors
# SPDX-License-Identifier: MIT AND BSD-3-Clause
"""FlashMask context parallelism on our packed layouts (baseline).

FlashMask (https://github.com/PaddlePaddle/FlashMask, ICLR 2025; FlashMask-3,
NeurIPS 2026) on H100 exposes its context parallelism through
``tests/torch/distributed/balanced_flashmask.py``: its load balancer
(``flashmask.cp_balance.balance_flashmask_input``) assigns 2048-token chunks
to ranks, each rank all-gathers K/V, runs one FlashMask kernel pass and
reduce-scatters dK/dV. That autograd function is inlined here so the kernels
come from ``flashmask.cute.ops`` (the module the test imports pulls in the
PaddlePaddle frontend). Masks are converted exactly by
:mod:`flashmask_masks`; documents with vision groups are reordered (text
first, image spans reversed), without which FlashMask cannot express them.

Runs in FlashMask's own environment (it pins another CUTLASS DSL than ours),
with ``usp_attention`` installed there without dependencies, and
``FLASHMASK_DIR`` pointing at a FlashMask checkout.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flashmask_masks import flashmask_ranges  # noqa: E402

_FM = os.environ.get("FLASHMASK_DIR", "/path/to/flashmask")
sys.path.insert(0, _FM)
sys.path.insert(0, os.path.join(_FM, "tests/torch/distributed"))

from cp_utils import all_gather_contiguous, reduce_scatter_any_axis_order  # noqa: E402
from flashmask.cp_balance import balance_flashmask_input  # noqa: E402
from flashmask.cp_balance.cp_balance_cuda_kernels import (  # noqa: E402
    indices_rerank_cuda,
    indices_to_chunks_cuda,
)
from flashmask.cute.ops.bwd import _flash_attn_bwd  # noqa: E402
from flashmask.cute.ops.fwd import _flash_attn_fwd  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from jaxtyping import Float


class _BalancedFlashMask(torch.autograd.Function):
    """FlashMask's BalancedFlashMaskFM4 (all-gather K/V, one pass, reduce-scatter dK/dV)."""

    @staticmethod
    def forward(ctx, query, key, value, local_mask, group):
        k = all_gather_contiguous(key, group=group, axis=1)
        v = all_gather_contiguous(value, group=group, axis=1)
        out, lse = _flash_attn_fwd(
            query, k, v, startend_row_indices=local_mask, causal=False,
            return_lse=True, pack_gqa=False,
        )[:2]  # fmt: skip
        ctx.save_for_backward(query, key, value, local_mask, out, lse)
        ctx.group = group
        return out

    @staticmethod
    def backward(ctx, dout):
        query, key, value, local_mask, out, lse = ctx.saved_tensors
        group = ctx.group
        k = all_gather_contiguous(key, group=group, axis=1)
        v = all_gather_contiguous(value, group=group, axis=1)
        dq, dk, dv = _flash_attn_bwd(
            query, k, v, out, dout.contiguous(), lse, local_mask, causal=False,
            deterministic=torch.are_deterministic_algorithms_enabled(),
        )[:3]  # fmt: skip
        return (
            dq,
            reduce_scatter_any_axis_order(dk, axis=1, group=group),
            reduce_scatter_any_axis_order(dv, axis=1, group=group),
            None,
            None,
        )


def _mask(document_lens, split_lens, attn_modes, window, device):
    perm, lo, hi = flashmask_ranges(document_lens, split_lens, attn_modes, window)
    mask = torch.from_numpy(np.stack([hi, lo], axis=-1).astype(np.int32))[None, None]
    return mask.to(device), perm, int((hi - lo).sum())


def _localize(mask, buckets, rank, chunk):
    """FlashMask's rerank + chunk localization for a given chunk assignment."""
    order = torch.tensor(
        [idx for bucket in buckets for _, idx in bucket],
        dtype=torch.int32,
        device=mask.device,
    )
    rerank = indices_rerank_cuda(mask, order)
    local = torch.tensor(
        [idx for _, idx in buckets[rank]], dtype=torch.int32, device=mask.device
    )
    return indices_to_chunks_cuda(rerank, local, chunk)


@dataclass(kw_only=True, slots=True)
class FlashMaskCP:
    """One FlashMask CP setup: which original tokens this rank holds, and attention."""

    local_tokens: torch.Tensor
    local_masks: dict
    group: object
    visible_pairs: dict
    ulysses_group: object = None
    ulysses: int = 1
    replicate_kv: int = 1

    def attention(
        self, window: int | None = None
    ) -> Callable[
        [
            Float[torch.Tensor, " batch local_tokens num_heads head_dim"],
            Float[torch.Tensor, " batch local_tokens kv_heads head_dim"],
            Float[torch.Tensor, " batch local_tokens kv_heads head_dim"],
        ],
        Float[torch.Tensor, " batch local_tokens num_heads head_dim"],
    ]:
        """Create an attention callable for a prebuilt window mask.

        Args:
            window: Optional sliding-window size, or None for global attention;
                must be one of the windows used to build this plan.

        Returns:
            A callable accepting local q, k and v tensors and returning the
            local attention output, including Ulysses exchanges when
            configured.
        """
        mask = self.local_masks[window]
        if self.ulysses == 1:
            return lambda q, k, v: _BalancedFlashMask.apply(q, k, v, mask, self.group)
        g, u, rep = self.ulysses_group, self.ulysses, self.replicate_kv

        def attn(q, k, v):
            if rep > 1:
                k, v = (t.repeat_interleave(rep, dim=2) for t in (k, v))
            q, k, v = (_seq_to_heads(t, g, u) for t in (q, k, v))
            return _heads_to_seq(
                _BalancedFlashMask.apply(q, k, v, mask, self.group), g, u
            )

        return attn


def _seq_to_heads(
    x: Float[torch.Tensor, " 1 local_tokens num_heads head_dim"], group, u
) -> Float[torch.Tensor, " 1 gathered_tokens local_heads head_dim"]:
    """Ulysses all-to-all: (1, l, H, D) -> (1, u*l, H/u, D) within the Ulysses group."""
    from torch.distributed.nn.functional import all_to_all_single

    _, l, h, d = x.shape
    x = x.reshape(l, u, h // u, d).transpose(0, 1).contiguous()
    out = all_to_all_single(torch.empty_like(x), x, group=group)
    return out.reshape(1, u * l, h // u, d)


def _heads_to_seq(
    x: Float[torch.Tensor, " 1 gathered_tokens local_heads head_dim"], group, u
) -> Float[torch.Tensor, " 1 local_tokens num_heads head_dim"]:
    """Inverse all-to-all: (1, u*l, H/u, D) -> (1, l, H, D)."""
    from torch.distributed.nn.functional import all_to_all_single

    _, length, hu, d = x.shape
    x = x.reshape(u, length // u, hu, d).contiguous()
    out = all_to_all_single(torch.empty_like(x), x, group=group)
    return out.transpose(0, 1).reshape(1, length // u, u * hu, d)


def build_flashmask_cp(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[str],
    *,
    windows: Sequence[int | None] = (None,),
    device: torch.device | str,
    group: dist.ProcessGroup | None = None,
    ulysses_group: dist.ProcessGroup | None = None,
    replicate_kv: int = 1,
    chunk: int = 2048,
    use_ipo: bool = True,
) -> FlashMaskCP:
    """Build a balanced FlashMask context-parallel attention plan.

    With ``ulysses_group`` (FM + Ulysses, the hybrid baseline of the attention
    tables): FlashMask CP runs across ``group``, and the ``U`` ranks of a
    Ulysses group split their CP shard's tokens ``U`` ways, trading them for a
    ``1/U`` head split with an all-to-all around each FlashMask call.
    ``replicate_kv`` repeats KV heads so ``U`` divides them.

    Args:
        document_lens: Lengths of the packed documents, in tokens.
        split_lens: Lengths of the attention spans, in document order.
        attn_modes: Attention mode for each span; must match split_lens.
        windows: Window sizes, with None entries selecting global attention.
            The first mask determines token balance; all other masks reuse that
            order.
        device: Device for the local token indices and masks.
        group: Context-parallel process group; None uses the world group.
        ulysses_group: Optional process group for the hybrid Ulysses head
            exchange.
        replicate_kv: KV head replication factor for the Ulysses exchange.
        chunk: Token granularity of FlashMask load balancing.
        use_ipo: Whether to use FlashMask's IPO mask ordering.

    Returns:
        A FlashMaskCP plan with local token indices, masks and process groups.
    """
    group = group or dist.group.WORLD
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    masks, pairs = {}, {}
    perm = None
    for w in windows:
        masks[w], perm, pairs[w] = _mask(
            document_lens, split_lens, attn_modes, w, device
        )
    local0, buckets = balance_flashmask_input(
        masks[windows[0]], world, rank, balance_chunk_size=chunk, use_ipo=use_ipo
    )
    local_masks = {windows[0]: local0}
    for w in windows[1:]:
        local_masks[w] = _localize(masks[w], buckets, rank, chunk)
    stored = torch.cat(
        [torch.arange(idx * chunk, (idx + 1) * chunk) for _, idx in buckets[rank]]
    )
    u = 1 if ulysses_group is None else dist.get_world_size(ulysses_group)
    if u > 1:
        part = stored.numel() // u
        u_rank = dist.get_rank(ulysses_group)
        stored = stored[u_rank * part : (u_rank + 1) * part]
    original = stored if perm is None else torch.as_tensor(perm)[stored]
    return FlashMaskCP(
        local_tokens=original.to(device),
        local_masks=local_masks,
        group=group,
        visible_pairs=pairs,
        ulysses_group=ulysses_group,
        ulysses=u,
        replicate_kv=replicate_kv,
    )
