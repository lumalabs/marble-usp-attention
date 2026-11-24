# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-FileCopyrightText: 2025 Bytedance Ltd. and/or its affiliates (packed-mask encoding adapted from BAGEL create_sparse_mask)
# SPDX-License-Identifier: MIT AND Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
from torch import Tensor

AttentionMode: TypeAlias = Literal["causal", "full", "noise", "group"]


@dataclass(kw_only=True, slots=True)
class PackedLayout:
    """Packed sequence metadata for mixed attention modes.

    Attributes:
        seq_len: Total packed sequence length.
        full_and_noise_seq_id: Segment id for full/noise tokens, -1 for causal.
        noise_seq_id: Segment id for noise tokens, -1 otherwise.
        document_id: Document/sample id per token.
        document_lens: Optional original document lengths.
        split_lens: Optional original split lengths.
        attn_modes: Optional original attention mode per split.
    """

    seq_len: int
    full_and_noise_seq_id: Tensor
    noise_seq_id: Tensor
    document_id: Tensor
    document_lens: list[int] | None = None
    split_lens: list[int] | None = None
    attn_modes: list[AttentionMode] | None = None

    def reordered(self, token_permutation: Tensor) -> PackedLayout:
        """Return a layout with all per-token metadata permuted."""
        return PackedLayout(
            seq_len=self.seq_len,
            full_and_noise_seq_id=self.full_and_noise_seq_id[token_permutation],
            noise_seq_id=self.noise_seq_id[token_permutation],
            document_id=self.document_id[token_permutation],
            document_lens=self.document_lens,
            split_lens=self.split_lens,
            attn_modes=self.attn_modes,
        )


def segment_ids(
    document_lens: list[int], split_lens: list[int], attn_modes: list[AttentionMode]
) -> list[int]:
    """Full/noise segment id per split, -1 for causal splits.

    Every full or noise split is its own segment; all ``"group"`` splits of a
    document share one segment (the id of the document's first group split).
    """
    if "group" not in attn_modes:
        return [i if m in ("full", "noise") else -1 for i, m in enumerate(attn_modes)]
    ids: list[int] = []
    split_iter = iter(enumerate(zip(split_lens, attn_modes)))
    for doc_len in document_lens:
        used, group = 0, None
        while used < doc_len:
            i, (length, mode) = next(split_iter)
            if mode == "group":
                group = i if group is None else group
                ids.append(group)
            else:
                ids.append(i if mode in ("full", "noise") else -1)
            used += length
    return ids


def build_packed_layout(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[AttentionMode],
    *,
    device: str | torch.device = "cpu",
    aux_tensors: list[Tensor] | None = None,
) -> PackedLayout:
    """Build per-token metadata used by mask logic.

    If ``aux_tensors`` is provided (``[fn_seq_id, noise_seq_id, doc_id]``),
    those tensors are reused directly instead of being rebuilt from
    ``split_lens`` / ``document_lens``.

    Raises:
        ValueError: If ``split_lens`` and ``attn_modes`` have mismatched lengths,
            document and split lengths don't sum to the same total, the total is
            non-positive, ``aux_tensors`` doesn't have exactly 3 elements, or any
            individual split length is non-positive.
    """
    if len(split_lens) != len(attn_modes):
        raise ValueError(
            f"split_lens and attn_modes must match, got {len(split_lens)=} and "
            f"{len(attn_modes)=}."
        )
    seq_len = sum(split_lens)
    if sum(document_lens) != seq_len:
        raise ValueError(
            f"Sum mismatch between document and split lengths: "
            f"{sum(document_lens)=}, {seq_len=}."
        )
    if seq_len <= 0:
        raise ValueError(f"Expected positive sequence length, got {seq_len=}.")

    if aux_tensors is not None:
        if len(aux_tensors) != 3:
            raise ValueError(
                f"aux_tensors must have exactly 3 elements, got {len(aux_tensors)}."
            )
        full_and_noise_seq_id, noise_seq_id, document_id = aux_tensors
    else:
        fn_tmp: list[int] = []
        noise_tmp: list[int] = []
        ids = segment_ids(document_lens, split_lens, attn_modes)
        for i, (length, mode) in enumerate(zip(split_lens, attn_modes)):
            if length <= 0:
                raise ValueError(
                    f"Split lengths must be positive, got {length=} at index {i}."
                )
            fn_tmp.extend([ids[i]] * length)
            noise_tmp.extend([i if mode == "noise" else -1] * length)

        full_and_noise_seq_id = torch.tensor(fn_tmp, dtype=torch.float32, device=device)
        noise_seq_id = torch.tensor(noise_tmp, dtype=torch.float32, device=device)
        document_id = torch.cat(
            [
                torch.full((length,), i)
                for i, length in enumerate(document_lens, start=1)
            ]
        ).to(device)

    return PackedLayout(
        seq_len=seq_len,
        full_and_noise_seq_id=full_and_noise_seq_id,
        noise_seq_id=noise_seq_id,
        document_id=document_id,
        document_lens=document_lens,
        split_lens=split_lens,
        attn_modes=attn_modes,
    )
