# SPDX-FileCopyrightText: 2022 the FlashAttention contributors, as shown by third_party_licenses/flash-attention-AUTHORS
# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: I002
# NOTE: do NOT add `from __future__ import annotations` to this file.
# The CUTE DSL inspects runtime types of `Constexpr`-typed parameters.
"""
Local override of flash_attn.cute.block_sparse_utils.consume_block_sparse_loads.

The upstream version hardcodes ``is_first_n_block=True`` / ``is_first_block=True``
for the very first KV-block processed per Q-tile.  For online (ring) attention,
when this is a non-first online call (``is_first_call=False``), the softmax running
state has already been loaded from the previous step and must *not* be re-initialised.
This file adds the ``is_first_call`` Constexpr parameter and threads it through the
first call site inside the function.

``tile_has_blocks`` lets middle online calls skip the state of a Q tile with no KV
blocks before its loads are issued.
"""

from functools import partial
from typing import Callable

import cutlass
import cutlass.cute as cute
from cutlass import const_expr
from flash_attn.cute.block_sparse_utils import get_curr_blocksparse_tensors
from flash_attn.cute.block_sparsity import BlockSparseTensors


def sparse_tensor_m_block(
    m_block,
    qhead_per_kvhead: cutlass.Constexpr[int],
    q_subtile_factor: cutlass.Constexpr[int],
):
    block = m_block
    if const_expr(qhead_per_kvhead != 1):
        block = block // qhead_per_kvhead
    if const_expr(q_subtile_factor != 1):
        block = block // q_subtile_factor
    return block


@cute.jit
def tile_has_blocks(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    seqlen,
    qhead_per_kvhead: cutlass.Constexpr[int] = 1,
    q_subtile_factor: cutlass.Constexpr[int] = 1,
):
    """Whether this Q tile has any mask or full KV block to process."""
    m_block_sparse = sparse_tensor_m_block(m_block, qhead_per_kvhead, q_subtile_factor)
    curr_mask_block_cnt, _, curr_full_block_cnt, _ = get_curr_blocksparse_tensors(
        batch_idx,
        head_idx,
        m_block_sparse,
        blocksparse_tensors,
        seqlen,
    )
    return curr_mask_block_cnt + curr_full_block_cnt > 0


@cute.jit
def consume_block_sparse_loads(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    seqlen,
    kv_consumer_state,
    mma_pv_fn,
    mma_one_n_block,
    process_first_half_block,
    process_last_half_block,
    mask_fn,
    score_mod_fn,
    O_should_accumulate,
    mask_mod,
    fastdiv_mods,
    intra_wg_overlap: cutlass.Constexpr,
    warp_scheduler_barrier_sync: Callable,
    warp_scheduler_barrier_arrive: Callable,
    qhead_per_kvhead: cutlass.Constexpr[int] = 1,
    q_subtile_factor: cutlass.Constexpr[int] = 1,
    is_first_call: cutlass.Constexpr = True,
):
    """Consume the mask and full block lists for a single tile on the consumer side.

    Mirrors ``flash_attn.cute.block_sparse_utils.consume_block_sparse_loads``
    (flash-attn-4 4.0.0b21) and adds ``is_first_call`` so that online (ring)
    attention non-first steps accumulate the loaded softmax state instead of
    re-initialising it. That is the only intended difference from upstream.

    Args:
        qhead_per_kvhead: Pack-GQA factor.
        is_first_call: False on non-first online/ring steps.  Controls whether
            ``is_first_n_block`` / ``is_first_block`` is ``True`` for the very
            first KV-block processed in this tile.
    """

    m_block_sparse = sparse_tensor_m_block(m_block, qhead_per_kvhead, q_subtile_factor)

    (
        curr_mask_block_cnt,
        curr_mask_block_idx,
        curr_full_block_cnt,
        curr_full_block_idx,
    ) = get_curr_blocksparse_tensors(
        batch_idx,
        head_idx,
        m_block_sparse,
        blocksparse_tensors,
        seqlen,
    )

    processed_any = curr_mask_block_cnt + curr_full_block_cnt > 0

    if const_expr(not intra_wg_overlap):
        if curr_mask_block_cnt > 0:
            mask_n_block = curr_mask_block_idx[curr_mask_block_cnt - 1]
            warp_scheduler_barrier_sync()
            kv_consumer_state = mma_one_n_block(
                kv_consumer_state,
                n_block=mask_n_block,
                mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                mask_fn=partial(
                    mask_fn,
                    mask_mod=mask_mod,
                    mask_seqlen=True,
                    fastdiv_mods=fastdiv_mods
                    if cutlass.const_expr(mask_mod is not None)
                    else None,
                ),
                is_first_n_block=is_first_call,
            )
            O_should_accumulate = True
            for i in cutlass.range(1, curr_mask_block_cnt):
                mask_n_block = curr_mask_block_idx[curr_mask_block_cnt - 1 - i]
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=mask_n_block,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_mod=mask_mod, mask_seqlen=False),
                    is_first_n_block=False,
                )
                O_should_accumulate = True
            if curr_full_block_cnt == 0:
                warp_scheduler_barrier_arrive()

        if curr_full_block_cnt > 0:
            full_n_block = curr_full_block_idx[curr_full_block_cnt - 1]
            if curr_mask_block_cnt == 0:
                warp_scheduler_barrier_sync()
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=full_n_block,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_seqlen=True),
                    is_first_n_block=is_first_call,
                )
                O_should_accumulate = True
                for i in cutlass.range(1, curr_full_block_cnt):
                    full_n_block = curr_full_block_idx[curr_full_block_cnt - 1 - i]
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state,
                        n_block=full_n_block,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                        mask_fn=partial(mask_fn, mask_seqlen=False),
                        is_first_n_block=False,
                    )
                    O_should_accumulate = True
            else:
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=full_n_block,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=True),
                    is_first_n_block=False,
                )
                O_should_accumulate = True
                for i in cutlass.range(1, curr_full_block_cnt):
                    full_n_block = curr_full_block_idx[curr_full_block_cnt - 1 - i]
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state,
                        n_block=full_n_block,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                        mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=False),
                        is_first_n_block=False,
                    )
                    O_should_accumulate = True
            warp_scheduler_barrier_arrive()
    else:
        if curr_mask_block_cnt > 0:
            mask_n_block = curr_mask_block_idx[curr_mask_block_cnt - 1]
            kv_consumer_state = process_first_half_block(
                n_block=mask_n_block,
                seqlen=seqlen,
                kv_consumer_state=kv_consumer_state,
                mask_fn=partial(
                    mask_fn,
                    mask_mod=mask_mod,
                    mask_seqlen=True,
                    fastdiv_mods=fastdiv_mods
                    if cutlass.const_expr(mask_mod is not None)
                    else None,
                ),
                score_mod_fn=score_mod_fn,
                is_first_block=is_first_call,
            )
            for i in cutlass.range(1, curr_mask_block_cnt):
                mask_n_block = curr_mask_block_idx[curr_mask_block_cnt - 1 - i]
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=mask_n_block,
                    seqlen=seqlen,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_mod=mask_mod, mask_seqlen=False),
                )
                O_should_accumulate = True

        if curr_full_block_cnt > 0:
            full_n_block = curr_full_block_idx[curr_full_block_cnt - 1]
            if curr_mask_block_cnt == 0:
                kv_consumer_state = process_first_half_block(
                    n_block=full_n_block,
                    seqlen=seqlen,
                    kv_consumer_state=kv_consumer_state,
                    mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=True),
                    score_mod_fn=score_mod_fn,
                    is_first_block=is_first_call,
                )
            else:
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=full_n_block,
                    seqlen=seqlen,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=True),
                )
                O_should_accumulate = True
            for i in cutlass.range(1, curr_full_block_cnt):
                full_n_block = curr_full_block_idx[curr_full_block_cnt - 1 - i]
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=full_n_block,
                    seqlen=seqlen,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=False),
                )
                O_should_accumulate = True

        if curr_mask_block_cnt + curr_full_block_cnt > 0:
            kv_consumer_state = process_last_half_block(
                kv_consumer_state=kv_consumer_state,
                zero_init=not O_should_accumulate,
            )
            O_should_accumulate = True

    return kv_consumer_state, O_should_accumulate, processed_any
