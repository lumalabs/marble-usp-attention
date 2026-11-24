# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Token order that makes vision-group masks tile-friendly.

A vision group (all image or frame tokens of a document mutually visible, as in
Molmo2) leaves most 128x128 tiles partially masked in the natural token order:
a video interleaves a few causal timestamp tokens with every frame, so almost
every tile mixes group and text tokens. Reordering each document that has
groups as its text splits in order, followed by its group splits in reverse
order, makes every key's visible queries one contiguous range (text sees the
later text and the groups after it; a group sees all groups and the text after
it), and the tiles become almost all full or empty (Molmo2 video at 512k: 4%
partial tiles instead of 49%).

The mask itself is unchanged: the predicate compares original positions, so the
plan carries the original position of every token and only the storage order
changes. Activations are permuted once at the model input, like the ring's
load-balancing permutation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from usp_attention.ring.packed_layout import AttentionMode


def vision_group_order(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[AttentionMode],
) -> Tensor | None:
    """Permutation (new index -> original index), or ``None`` without groups.

    Documents without group splits keep their order. In a document with
    groups, its non-group splits come first in order, then its group splits in
    reverse order.
    """
    if "group" not in attn_modes:
        return None
    lens = np.asarray(split_lens, dtype=np.int64)
    split_start = np.cumsum(lens) - lens
    doc_end = np.cumsum(np.asarray(document_lens, dtype=np.int64))
    split_doc = np.searchsorted(doc_end, split_start, side="right")
    is_group = np.asarray(attn_modes) == "group"
    has_group = np.zeros(len(document_lens), bool)
    has_group[split_doc[is_group]] = True
    idx = np.arange(len(lens))
    # Per document: text splits in order, then group splits in reverse order;
    # documents without groups keep their order.
    grouped = is_group & has_group[split_doc]
    order = np.lexsort((np.where(grouped, -idx, idx), grouped, split_doc))
    lens_o = lens[order]
    within = np.arange(int(lens.sum())) - np.repeat(np.cumsum(lens_o) - lens_o, lens_o)
    return torch.from_numpy(np.repeat(split_start[order], lens_o) + within)
