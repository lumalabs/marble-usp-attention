# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Partially masked 128x128 tiles of vision-group masks, with and without reorder.

Reproduces the paper's claim (seqdim.tex, Sec. "Layout transforms"):

    "By sorting like this, each key's visible queries become one contiguous
    range, and partially marked tiles on Molmo2 video at 512k tokens drop from
    49% to 4%."

For each vision-group layout this classifies every 128x128 (query, key) tile
of the attention mask as empty, full (all 16384 pairs visible) or partial,
once in the layout's own token order and once in
:func:`~usp_attention.ring.group_order.vision_group_order`
(``USPConfig(reorder_vision_groups=True)``, the default), and reports
``partial / (full + partial)``: the share of non-empty tiles on which the
kernel evaluates the mask per element.

The tiles are those of one call over the whole sequence (Ulysses only, which
splits heads, not tokens). Ring plans only permute whole 128-token blocks of
this order (PTRR blocks, or contiguous chunks for the KV gather), so the
per-rank masks of any ring size hold exactly the same tiles and the same share.

Per query tile, each row's visible keys are intervals in original positions
(its document's prefix; the rest of its full / noise span; for a group row,
every later group split of its document; other noise spans removed, as in
:func:`~usp_attention.reference.visible_key_counts`); a difference array gives
per-key coverage, which is permuted to stored order and summed per key tile.
``tests/test_claim_partial_tiles.py`` checks the counts against the dense
reference mask.

    PYTHONPATH=src:benchmarks python benchmarks/claim_partial_tiles.py \\
        --out benchmarks/results/claims/partial_tiles.jsonl
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from provenance import result_json

from usp_attention.layouts import build_layout
from usp_attention.ring.group_order import vision_group_order

PACKS = Path(__file__).resolve().parent / "layouts" / "packs"
TILE = 128


@dataclass(kw_only=True, slots=True)
class TileCounts:
    """Tiles of one mask in one token order.

    Attributes:
        full: Tiles with every (query, key) pair visible.
        partial: Tiles with some but not all pairs visible.
        pairs: Visible (query, key) pairs (a consistency check).
    """

    full: int
    partial: int
    pairs: int

    @property
    def partial_share(self) -> float:
        return self.partial / max(1, self.full + self.partial)


def tile_counts(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[str],
    perm: np.ndarray | None,
    *,
    tile: int = TILE,
) -> TileCounts:
    """Full and partial ``tile`` x ``tile`` tiles of the mask stored in ``perm``
    order (new index -> original index; ``None`` for the layout's order).

    The permutation must keep documents contiguous and in place (as
    :func:`vision_group_order` does).
    """
    seq = int(sum(document_lens))
    assert seq % tile == 0
    perm = np.arange(seq) if perm is None else np.asarray(perm, dtype=np.int64)
    lens = np.asarray(split_lens, dtype=np.int64)
    modes = np.asarray(attn_modes)
    s0 = np.cumsum(lens) - lens
    s1 = s0 + lens
    doc_end = np.cumsum(np.asarray(document_lens, dtype=np.int64))
    doc_start = doc_end - np.asarray(document_lens, dtype=np.int64)
    split_doc = np.searchsorted(doc_end, s0, side="right")
    tok_split = np.repeat(np.arange(len(lens)), lens)
    is_group = modes == "group"
    is_noise = modes == "noise"
    noise_splits = np.flatnonzero(is_noise)
    is_causal = modes == "causal"
    full = partial = pairs = 0
    for t0 in range(0, seq, tile):
        q = perm[t0 : t0 + tile]
        qs = tok_split[q]
        qd = split_doc[qs]
        d0, d1 = int(doc_start[qd.min()]), int(doc_end[qd.max()])
        # Stored positions of these documents: the same range (docs in place).
        k0, k1 = d0 // tile * tile, -(-d1 // tile) * tile
        diff = np.zeros(d1 - d0 + 1, dtype=np.int64)
        hi = np.where(is_causal[qs], q + 1, s1[qs])
        np.add.at(diff, doc_start[qd] - d0, 1)
        np.add.at(diff, hi - d0, -1)
        # Group rows also see every later group split of their document.
        g_rows = qs[is_group[qs]]
        for g in np.flatnonzero(
            is_group & (split_doc >= qd.min()) & (split_doc <= qd.max())
        ):
            n = int(((g_rows < g) & (split_doc[g_rows] == split_doc[g])).sum())
            if n:
                diff[s0[g] - d0] += n
                diff[s1[g] - d0] -= n
        cover = np.cumsum(diff[:-1])
        # A noise span is visible to its own rows (whole) and to no one else.
        for g in noise_splits[(s0[noise_splits] >= d0) & (s1[noise_splits] <= d1)]:
            cover[s0[g] - d0 : s1[g] - d0] = int((qs == g).sum())
        stored = np.zeros(k1 - k0, dtype=np.int64)
        idx = perm[k0:k1]
        inside = (idx >= d0) & (idx < d1)
        stored[inside] = cover[idx[inside] - d0]
        per_tile = stored.reshape(-1, tile).sum(axis=1)
        full += int((per_tile == tile * tile).sum())
        partial += int(((per_tile > 0) & (per_tile < tile * tile)).sum())
        pairs += int(per_tile.sum())
    return TileCounts(full=full, partial=partial, pairs=pairs)


def _layouts(builtin_len: int) -> list[tuple[str, str, int]]:
    """(label, layout spec, tokens) of every vision-group layout."""
    out = [(n, n, builtin_len) for n in ("vlm_group", "video_group")]
    for path in sorted(PACKS.glob("*.json.gz")):
        m = re.fullmatch(r"(.+)_(\d+)k\.json\.gz", path.name)
        name = path.name.removesuffix(".json.gz")
        if m and ("molmo2" in name or "group" in name):
            out.append(
                (f"{name}#0", f"file:{PACKS / name}.json#0", int(m.group(2)) * 1024)
            )
    return out


def main() -> None:
    """Count partial tiles with and without the vision-group order."""
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--out", required=True)
    p.add_argument("--builtin-len", type=int, default=512 * 1024)
    p.add_argument("--max-k", type=int, default=4096)
    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as out:
        for label, spec, seq in _layouts(args.builtin_len):
            if seq > args.max_k * 1024:
                continue
            docs, splits, modes = build_layout(spec, seq)
            if "group" not in modes:
                continue
            perm = vision_group_order(docs, splits, modes)
            before = tile_counts(docs, splits, modes, None)
            after = tile_counts(docs, splits, modes, perm.numpy())
            assert before.pairs == after.pairs
            row = {
                "bench": "claim_partial_tiles",
                "layout": label,
                "seq_len": seq,
                "tile": TILE,
                "natural": {
                    "full": before.full,
                    "partial": before.partial,
                    "partial_share": before.partial_share,
                },
                "group_order": {
                    "full": after.full,
                    "partial": after.partial,
                    "partial_share": after.partial_share,
                },
                "visible_pairs": before.pairs,
            }
            out.write(result_json(row) + "\n")
            out.flush()
            print(
                f"{label:32s} partial tiles {before.partial_share:6.1%} -> "
                f"{after.partial_share:6.1%}  (non-empty {before.full + before.partial} "
                f"-> {after.full + after.partial})",
                flush=True,
            )


if __name__ == "__main__":
    main()
