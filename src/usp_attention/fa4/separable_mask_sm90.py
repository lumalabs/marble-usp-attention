# SPDX-FileCopyrightText: 2025 Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao (fragment coordinate setup)
# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT AND BSD-3-Clause
"""Separable masks on FA4's SM90 partial tiles: per-row and per-key loads once.

FA4 applies a ``mask_mod`` on partially masked tiles once per accumulator
element, so every element reloads the metadata of its query and its key. Our
masks are separable: a few values per query (``row_fn``), a few per key
(``col_fn``) and a cheap test of the two (``combine``). Examples are the
column-range mask of :mod:`~usp_attention.fa4.column_ranges` (``lo[k] <= q <
hi[k]``: no row load, two key loads) and the position-aware ring predicate
(four loads each side). On 8 H100s with Molmo2 512k layouts the per-element
path costs 5-7% of a forward and backward step; loading each value once per
tile recovers it, matching FlashMask, which stores masks the same way.

:func:`install` wraps ``flash_attn.cute.mask.AttentionMask.apply_mask`` (the
SM90 forward and backward share it). For a ``mask_mod`` registered with
:func:`mark`, it keeps the values of the thread's shorter fragment dimension
in registers and loads the other side's once per index; any other mask goes
to FA4's own code. The wrapper is written against the pinned FA4 version and
is not installed for another version, in which case a marked ``mask_mod``
still runs, per element, through FA4's path.
"""

from __future__ import annotations

import importlib.metadata
from typing import Callable, Optional

_FA4_VERSION = "4.0.0b34"
_MARKER = "__usp_separable__"
_FA4_APPLY_MASK = None  # FA4's AttentionMask.apply_mask, once wrapped

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, const_expr
    from flash_attn.cute import mask as fa_mask
    from flash_attn.cute import utils as fa_utils
    from flash_attn.cute.utils import AuxData
    from quack import layout_utils

    HAS_FA4 = True
except ImportError:
    HAS_FA4 = False


def mark(
    mask_mod: Callable,
    row_fn: Callable,
    col_fn: Callable,
    combine: Callable,
    *,
    rows_hoisted_only: bool = False,
) -> Callable:
    """Register the separable form of ``mask_mod``.

    ``row_fn(q, aux_tensors)`` and ``col_fn(kv, aux_tensors)`` take Int32
    indices and return tuples of scalars; ``combine(row, col)`` returns the
    keep bit, equal to ``mask_mod`` at ``(q, kv)``. With
    ``rows_hoisted_only`` the path is used only where a thread holds fewer
    query rows than key columns (the forward tiles); elsewhere FA4's own
    per-element path runs. On 8 H100s that is faster for the position-aware
    predicate, whose four per-row loads make FA4's backward tiles (many rows,
    few columns per thread) slower when streamed. The ``mask_mod`` also gets
    a compile-cache hash of its own, so a kernel compiled without the fast
    path is never reused for it.
    """
    setattr(mask_mod, _MARKER, (row_fn, col_fn, combine, rows_hoisted_only))
    base = fa_utils.hash_callable(mask_mod, set_cute_hash=False)
    parts = [base] + [fa_utils.hash_callable(f) for f in (row_fn, col_fn, combine)]
    parts.append("rows" if rows_hoisted_only else "any")
    mask_mod.__cute_hash__ = ":".join(parts) + ":usp-separable-sm90-v2"
    return mask_mod


def install() -> bool:
    """Install the separable path; ``False`` (and no change) on another FA4."""
    global _FA4_APPLY_MASK
    if _FA4_APPLY_MASK is not None:
        return True
    if not HAS_FA4:
        return False
    try:
        if importlib.metadata.version("flash-attn-4") != _FA4_VERSION:
            return False
    except importlib.metadata.PackageNotFoundError:
        return False
    _FA4_APPLY_MASK = fa_mask.AttentionMask.apply_mask
    fa_mask.AttentionMask.apply_mask = _apply_mask
    return True


if HAS_FA4:

    @cute.jit
    def _apply_separable(
        self,
        acc_S: cute.Tensor,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        mask_seqlen: cutlass.Constexpr[bool],
        aux_data: AuxData,
        fns: cutlass.Constexpr,
    ) -> None:
        row_fn, col_fn, combine, _ = fns
        # Fragment coordinates exactly as FA4's mask_mod branch computes them.
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=self.swap_AB)
        acc_shape = (self.tile_m, self.tile_n)
        cS = cute.make_identity_tensor(
            acc_shape if not self.swap_AB else acc_shape[::-1]
        )
        tScS_mn = layout_utils.reshape_acc_to_mn(
            thr_mma.partition_C(cS), transpose=self.swap_AB
        )
        t0ScS_mn = layout_utils.reshape_acc_to_mn(
            thr_mma.get_slice(0).partition_C(cS), transpose=self.swap_AB
        )
        ROW = 0 if const_expr(not self.swap_AB) else 1
        COL = 1 if const_expr(not self.swap_AB) else 0
        thr_col_offset = tScS_mn[0][COL]
        # FA4 treats a negative n_block (fully masked rows) as block 0.
        kv_base = cutlass.max(n_block, 0) * self.tile_n + thr_col_offset
        q_base = m_block * self.tile_m
        nrow = const_expr(cute.size(tScS_mn.shape[0]))
        ncol = const_expr(cute.size(tScS_mn.shape[1]))
        aux = aux_data.tensors

        qs = []
        for r in cutlass.range_constexpr(nrow):
            q = tScS_mn[r, 0][ROW] + q_base
            if const_expr(self.qhead_per_kvhead_packgqa != 1):
                q = q // self.qhead_per_kvhead_packgqa
            qs.append(q)
        kvs = []
        for c in cutlass.range_constexpr(ncol):
            kvs.append(kv_base + t0ScS_mn[0, c][COL])

        # Out-of-range indices load the last row / key and are masked below
        # (FA4 wraps them modulo the length and masks them too).
        if const_expr(nrow <= ncol):
            rows = []
            for r in cutlass.range_constexpr(nrow):
                rows.append(row_fn(cutlass.min(qs[r], self.seqlen_q - 1), aux))
            for c in cutlass.range_constexpr(ncol):
                col = col_fn(cutlass.min(kvs[c], self.seqlen_k - 1), aux)
                for r in cutlass.range_constexpr(nrow):
                    keep = combine(rows[r], col)
                    if const_expr(mask_seqlen):
                        keep = keep & (qs[r] < self.seqlen_q) & (kvs[c] < self.seqlen_k)
                    acc_S_mn[r, c] = acc_S_mn[r, c] if keep else -Float32.inf
        else:
            cols = []
            for c in cutlass.range_constexpr(ncol):
                cols.append(col_fn(cutlass.min(kvs[c], self.seqlen_k - 1), aux))
            for r in cutlass.range_constexpr(nrow):
                row = row_fn(cutlass.min(qs[r], self.seqlen_q - 1), aux)
                for c in cutlass.range_constexpr(ncol):
                    keep = combine(row, cols[c])
                    if const_expr(mask_seqlen):
                        keep = keep & (qs[r] < self.seqlen_q) & (kvs[c] < self.seqlen_k)
                    acc_S_mn[r, c] = acc_S_mn[r, c] if keep else -Float32.inf

    def _rows_fit(self, thr_mma) -> bool:
        """Whether a thread holds at most as many query rows as key columns."""
        nrow, ncol = _fragment_shape(self, thr_mma)
        return nrow <= ncol

    def _fragment_shape(self, thr_mma) -> tuple[int, int]:
        """(query rows, key columns) of a thread's accumulator fragment."""
        acc_shape = (self.tile_m, self.tile_n)
        cS = cute.make_identity_tensor(
            acc_shape if not self.swap_AB else acc_shape[::-1]
        )
        tScS_mn = layout_utils.reshape_acc_to_mn(
            thr_mma.partition_C(cS), transpose=self.swap_AB
        )
        return cute.size(tScS_mn.shape[0]), cute.size(tScS_mn.shape[1])

    @cute.jit
    def _apply_mask(
        self,
        acc_S: cute.Tensor,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        mask_local: cutlass.Constexpr[bool] = False,
        mask_mod: cutlass.Constexpr[Optional[Callable]] = None,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
    ) -> None:
        fns = getattr(mask_mod, _MARKER, None) if mask_mod is not None else None
        if const_expr(
            fns is not None
            and not mask_causal
            and not mask_local
            and (not fns[3] or _rows_fit(self, thr_mma))
        ):
            _apply_separable(
                self,
                acc_S,
                m_block,
                n_block,
                thr_mma,
                mask_seqlen,
                aux_data,
                fns,
            )
        else:
            _FA4_APPLY_MASK(
                self,
                acc_S,
                batch_idx,
                head_idx,
                m_block,
                n_block,
                thr_mma,
                mask_seqlen,
                mask_causal,
                mask_local,
                mask_mod,
                aux_data,
                fastdiv_mods,
            )
