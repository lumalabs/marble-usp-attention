# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: I002
# NOTE: do NOT add `from __future__ import annotations` to this file. The CUTE DSL
# inspects the runtime types of `Constexpr`-typed parameters on `__call__` to decide
# compile-time specialization; with stringified annotations those params are treated
# as dynamic and `const_expr(...)` raises DSLRuntimeError.

import operator
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Callable, Literal, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import Float32, Int32, const_expr, pipeline
from cutlass.base_dsl.arch import Arch
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import LayoutEnum
from flash_attn.cute import pipeline as pipeline_custom
from flash_attn.cute import utils
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.block_sparse_utils import (
    produce_block_sparse_loads,
)
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute.flash_fwd_sm90 import FlashAttentionForwardSm90
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.named_barrier import NamedBarrierFwd
from flash_attn.cute.pack_gqa import (
    PackGQA,
    make_packgqa_tiled_tma_atom,
    pack_gqa_layout,
)
from flash_attn.cute.paged_kv import PagedKVManager
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.softmax import Softmax, apply_score_mod_inner
from flash_attn.cute.tile_scheduler import (
    SingleTileLPTScheduler,
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)
from quack import copy_utils, layout_utils, sm90_utils
from quack.cute_dsl_utils import ParamsBase

from usp_attention.online_flash.block_sparse_utils import (
    consume_block_sparse_loads,
    tile_has_blocks,
)


@dataclass  # noqa: MLR-04.01  vendored from flash-attn-4; parent `Softmax` is a positional-init dataclass
class Softmax2TheSequel(Softmax):
    """This is the same online softmax as before, but doesn't overwrite the row_sum"""

    @staticmethod
    def create(
        scale_log2: Float32,
        num_rows: cutlass.Constexpr[int],
        arch: cutlass.Constexpr[int] = 80,
        softmax_scale: Float32 | None = None,
    ):
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return Softmax2TheSequel(
            scale_log2, num_rows, row_max, row_sum, arch, softmax_scale
        )

    @cute.jit
    def load(
        self,
        mMax: cute.Tensor,
        mSum: cute.Tensor,
        seqlen: SeqlenInfoQK,
        tiled_mma: cute.TiledMma,
        tile_m: Int32,
        tile_hdimv: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        tidx: Int32,
        m_block: Int32,
    ):
        cO = cute.make_identity_tensor((tile_m, tile_hdimv))

        mMax_cur = seqlen.offset_batch_Q(mMax, batch_idx, dim=2)[None, head_idx]
        mSum_cur = seqlen.offset_batch_Q(mSum, batch_idx, dim=2)[None, head_idx]

        gMax = cute.local_tile(mMax_cur, (tile_m,), (m_block,))
        gSum = cute.local_tile(mSum_cur, (tile_m,), (m_block,))

        gMax_expanded_layout = cute.append(
            gMax.layout, cute.make_layout((tile_hdimv,), stride=(0,))
        )
        gMax_expanded = cute.make_tensor(gMax.iterator, gMax_expanded_layout)
        gSum_expanded_layout = cute.append(
            gSum.layout, cute.make_layout((tile_hdimv,), stride=(0,))
        )
        gSum_expanded = cute.make_tensor(gSum.iterator, gSum_expanded_layout)
        thr_mma = tiled_mma.get_slice(tidx)

        taccOgMax = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(gMax_expanded))
        taccOgSum = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(gSum_expanded))

        assert cute.size(taccOgMax, mode=[0]) == cute.size(self.row_max)

        taccOcO = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cO))
        t0accOcO = layout_utils.reshape_acc_to_mn(thr_mma.get_slice(0).partition_C(cO))
        is_col0 = taccOcO[0][1] == 0

        for m in cutlass.range(cute.size(taccOgMax.shape[1]), unroll_full=True):
            if t0accOcO[m, 0][0] < seqlen.seqlen_q - m_block * tile_m - taccOcO[0][0]:
                self.row_max[m] = taccOgMax[m, 0]
                self.row_sum[m] = taccOgSum[m, 0] if is_col0 else 0.0
            else:
                self.row_max[m] = -Float32.inf
                self.row_sum[m] = 0.0


class OnlineFlashAttentionForwardSm90(FlashAttentionForwardSm90):
    def __init__(
        self,
        *args,
        intra_wg_overlap: bool = True,
        mma_pv_is_rs: bool = True,
        paged_kv_non_tma: bool = False,
        **kwargs,
    ):
        super().__init__(
            intra_wg_overlap=intra_wg_overlap,
            mma_pv_is_rs=mma_pv_is_rs,
            paged_kv_non_tma=paged_kv_non_tma,
            *args,
            **kwargs,
        )
        assert not self.pack_gqa, "pack_gqa isn't supported yet"
        assert not paged_kv_non_tma, "paged_kv_non_tma isn't supported yet"

    def _get_smem_layout_atom(self):
        sQ_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR, self.dtype, self.tile_hdim
            ),
            self.dtype,
        )
        sK_layout_atom = sQ_layout_atom
        sV_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR, self.dtype, self.tile_hdimv
            ),
            self.dtype,
        )
        sO_layout_atom = sV_layout_atom
        if not self.mma_pv_is_rs:
            sP_layout_atom = warpgroup.make_smem_layout_atom(
                sm90_utils_basic.get_smem_layout_atom(
                    LayoutEnum.ROW_MAJOR, self.dtype, self.tile_n
                ),
                self.dtype,
            )
        else:
            sP_layout_atom = None
        return (
            sQ_layout_atom,
            sK_layout_atom,
            sV_layout_atom,
            sO_layout_atom,
            sP_layout_atom,
        )

    def _check_type(
        self,
        mQ_type,
        mK_type,
        mV_type,
        mO_type,
        mLSE_type,
        mCuSeqlensQ_type,
        mCuSeqlensK_type,
        mSeqUsedQ_type,
        mSeqUsedK_type,
    ):
        if const_expr(not (mQ_type == mK_type == mV_type)):
            raise TypeError("Q, K, V must have the same data type")
        if const_expr(mQ_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported for Q/K/V")
        # mO is fp32 for intermediate online-attention accumulation
        if const_expr(mO_type not in [cutlass.Float16, cutlass.BFloat16, Float32]):
            raise TypeError("O must be Float16, BFloat16, or Float32")
        if const_expr(mLSE_type not in [None, Float32]):
            raise TypeError("LSE tensor must be Float32")
        if const_expr(mCuSeqlensQ_type not in [None, Int32]):
            raise TypeError("cu_seqlens_q tensor must be Int32")
        if const_expr(mCuSeqlensK_type not in [None, Int32]):
            raise TypeError("cu_seqlens_k tensor must be Int32")
        if const_expr(mSeqUsedQ_type not in [None, Int32]):
            raise TypeError("seqused_q tensor must be Int32")
        if const_expr(mSeqUsedK_type not in [None, Int32]):
            raise TypeError("seqused_k tensor must be Int32")
        assert mQ_type == self.dtype

    def _get_tiled_mma(self):
        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, 1, 1),
            tiler_mn=(64, self.tile_n),
        )
        tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(
                self.tile_m // 64,
                1,
                1,
            ),  # Might need (1, 2, 1) for hdim 512
            tiler_mn=(64, self.tile_hdimv),
            a_source=warpgroup.OperandSource.RMEM
            if self.mma_pv_is_rs
            else warpgroup.OperandSource.SMEM,
        )
        return tiled_mma_qk, tiled_mma_pv

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct = [
            cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(layout)],
                self.buffer_align_bytes,
            ]
            for layout in (
                self.sQ_layout,
                self.sK_layout,
                self.sV_layout,
            )
        ]
        cosize_sQV = max(cute.cosize(self.sQ_layout), cute.cosize(self.sV_layout))
        sQV_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cosize_sQV], 1024
        ]
        cosize_sP = (
            cute.cosize(self.sP_layout) if const_expr(self.sP_layout is not None) else 0
        )
        sP_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sP], 1024]

        # sO (fp32) holds the online state between calls and stages the epilogue
        # TMA store, up to head_dim 128. Above it the state bypasses shared
        # memory (copy_O_state) and sO is empty.
        cosize_sO = (
            cute.cosize(self.sO_layout) if const_expr(self.o_state_in_smem) else 0
        )
        sO_struct = cute.struct.Align[
            cute.struct.MemRange[Float32, cosize_sO], self.buffer_align_bytes
        ]

        # 1 stage * 2 for Q pipeline (full + empty), self.num_stages*2 for K/V,
        # 1 stage * 2 for the O reload pipeline used on non-first online calls.
        mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, 1 * 2]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_O_load_struct = cute.struct.MemRange[cutlass.Int64, 1 * 2]

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr_Q: mbar_ptr_Q_struct  # type: ignore
            mbar_ptr_K: mbar_ptr_K_struct  # type: ignore
            mbar_ptr_V: mbar_ptr_V_struct  # type: ignore
            mbar_ptr_O_load: mbar_ptr_O_load_struct  # type: ignore
            sV: sV_struct  # type: ignore
            sQ: sQ_struct  # type: ignore
            sK: sK_struct  # type: ignore
            sP: sP_struct  # type: ignore
            sO: sO_struct  # type: ignore

        @cute.struct
        class SharedStorageSharedQV:
            mbar_ptr_Q: mbar_ptr_Q_struct  # type: ignore
            mbar_ptr_K: mbar_ptr_K_struct  # type: ignore
            mbar_ptr_V: mbar_ptr_V_struct  # type: ignore
            sQ: sQV_struct  # type: ignore
            sK: sK_struct  # type: ignore
            sP: sP_struct  # type: ignore

        return (
            SharedStorageQKV
            if const_expr(not self.Q_in_regs)
            else SharedStorageSharedQV
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        mK: cute.Tensor,  # (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table
        mV: cute.Tensor,  # (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table
        mO: cute.Tensor,  # (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        mLSE: Optional[cute.Tensor],
        mMax: cute.Tensor,
        mSum: cute.Tensor,
        is_first_call: cutlass.Constexpr,
        is_last_call: cutlass.Constexpr,
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,  # (b_k, max_num_pages_per_seq)
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_tensors: Optional[list] = None,
        mO_state: Optional[cute.Tensor] = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        """Configures and launches the flash attention kernel.

        mQ/mK/mV/mO has same data types(supports fp16 and bf16) and same layout:
        (batch_size, seqlen_q, num_head, head_dim):(_, _, _, 1)

        mO holds the fp32 running output between calls. A resumed last call
        may instead read that state from ``mO_state`` (fp32) and write the
        normalized output to ``mO`` in fp16/bf16.
        """

        # TODO: add mMax and mSum
        self._check_type(
            *(
                t.element_type if t is not None else None
                for t in (
                    mQ,
                    mK,
                    mV,
                    mO,
                    mLSE,
                    mCuSeqlensQ,
                    mCuSeqlensK,
                    mSeqUsedQ,
                    mSeqUsedK,
                )
            )
        )

        self.varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None

        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        QO_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        )
        mQ, mO = [layout_utils.select(t, QO_layout_transpose) for t in (mQ, mO)]
        # The fp32 state a resumed call reads: mO itself unless given separately.
        if const_expr(mO_state is not None):
            assert is_last_call and not is_first_call
            mO_state = layout_utils.select(
                assume_tensor_aligned(mO_state), QO_layout_transpose
            )
        else:
            mO_state = mO
        KV_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        )
        mK, mV = [layout_utils.select(t, KV_layout_transpose) for t in (mK, mV)]
        LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mLSE = (
            layout_utils.select(mLSE, LSE_layout_transpose)
            if const_expr(mLSE is not None)
            else None
        )
        mMax = (
            layout_utils.select(mMax, LSE_layout_transpose)
            if const_expr(mMax is not None)
            else None
        )
        mSum = (
            layout_utils.select(mSum, LSE_layout_transpose)
            if const_expr(mSum is not None)
            else None
        )

        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_qk.size
        self.num_threads_per_warp_group = 128
        self.num_wg_mma = self.num_mma_threads // self.num_threads_per_warp_group
        assert self.num_wg_mma in [1, 2, 3]
        self.num_threads = self.num_threads_per_warp_group * (self.num_wg_mma + 1)
        self.num_producer_threads = 32
        self.num_Q_load_threads = self.num_threads_per_warp_group  # If not TMA_Q
        self.num_epilogue_threads = self.num_mma_threads
        self.num_mma_regs, self.num_producer_regs = {
            1: (256, 56),
            2: (240, 24),
            3: (160, 32),
        }[self.num_wg_mma]
        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)
        # A middle call (neither first nor last) leaves the state of a
        # block-sparse tile with no KV blocks as it is in global memory: it is
        # neither loaded nor stored. First calls must still initialize it, and
        # last calls must still normalize it.
        self.skip_empty_tiles = (
            self.use_block_sparsity and not is_first_call and not is_last_call
        )

        self.use_scheduler_barrier = (
            (self.num_wg_mma >= 2 and self.tile_hdim <= 128)
            if const_expr(self.intra_wg_overlap)
            else (self.num_wg_mma == 2)
        )
        self.use_tma_Q = self.arch >= Arch.sm_90 and not (
            self.pack_gqa and self.tile_m % self.qhead_per_kvhead != 0
        )
        self.use_tma_O = self.use_tma_Q
        # Producer needs more registers when doing cp.async Q or KV loads
        if const_expr(
            self.num_wg_mma == 2 and (not self.use_tma_Q or not self.use_tma_KV)
        ):
            self.num_mma_regs, self.num_producer_regs = 224, 40
        self.rescale_O_before_gemm = self.tile_hdimv > 128 and self.intra_wg_overlap
        # Above head_dim 128 an fp32 O tile in shared memory does not fit next to
        # Q / K / V (Hopper allows 227 KiB), so the online state moves directly
        # between global memory and the accumulator registers instead.
        self.o_state_in_smem = self.tile_hdim <= 128 and self.tile_hdimv <= 128
        if const_expr(not self.o_state_in_smem):
            self.use_tma_O = False
        self._setup_attributes()
        # TODO: we prob don't need most of what's in _setup_attributes
        self.sQ_layout, self.sK_layout, self.sV_layout = [
            sm90_utils.make_smem_layout(
                mX.element_type, LayoutEnum.ROW_MAJOR, shape, stage
            )
            for mX, shape, stage in [
                (mQ, (self.tile_m, self.tile_hdim), None),
                (mK, (self.tile_n, self.tile_hdim), self.num_stages),
                (mV, (self.tile_n, self.tile_hdimv), self.num_stages),
            ]
        ]
        # sO is always fp32; sO_store_layout matches mO.element_type (fp32 for
        # non-last calls, fp16/bf16 for a last call) and is used for the epilogue TMA.
        self.o_dtype = mO.element_type
        self.sO_layout = sm90_utils.make_smem_layout(
            Float32, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_hdimv)
        )
        self.sO_store_layout = sm90_utils.make_smem_layout(
            mO.element_type, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_hdimv)
        )
        self.sP_layout = None
        if const_expr(not self.mma_pv_is_rs):
            self.sP_layout = sm90_utils.make_smem_layout(
                mV.element_type, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_n)
            )
            assert is_first_call, (
                "Cannot use OnlineFlashAttention with these dimensions yet (but this is fixable, either we change the tile dims or switch out TMA for cpasync copies)"
            )

        SharedStorage = self._get_shared_storage_cls()

        mQ_og, mO_og = mQ, mO
        if const_expr(self.pack_gqa):
            nheads_kv = mK.shape[2]
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(
                    mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1
                )
            if const_expr(mMax is not None):
                mMax = pack_gqa_layout(
                    mMax, self.qhead_per_kvhead, nheads_kv, head_idx=1
                )
                mSum = pack_gqa_layout(
                    mSum, self.qhead_per_kvhead, nheads_kv, head_idx=1
                )

        # TMA
        gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_KV = cpasync.CopyBulkTensorTileG2SOp()  # Might multicast
        gmem_tiled_copy_O = cpasync.CopyBulkTensorTileS2GOp()
        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
            ]
        }
        if const_expr(not is_first_call and self.o_state_in_smem):
            self.tma_copy_bytes["O_load"] = cute.size_in_bytes(
                Float32, cute.select(self.sO_layout, mode=[0, 1])
            )
        make_tiled_tma_atom_fn = (
            partial(
                make_packgqa_tiled_tma_atom,
                qhead_per_kvhead=self.qhead_per_kvhead,
                head_idx=2,
            )
            if const_expr(self.pack_gqa)
            else cpasync.make_tiled_tma_atom
        )
        tma_atom_Q, tma_tensor_Q = None, None
        if const_expr(self.use_tma_Q):
            tma_atom_Q, tma_tensor_Q = make_tiled_tma_atom_fn(
                gmem_tiled_copy_Q,
                mQ_og if const_expr(self.pack_gqa) else mQ,
                self.sQ_layout,
                (self.tile_m, self.tile_hdim),  # No mcast
            )
        tma_atom_K, tma_tensor_K = None, None
        tma_atom_V, tma_tensor_V = None, None
        if const_expr(self.use_tma_KV):
            tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mK,
                cute.select(self.sK_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdim),
                1,  # No mcast for now
            )
            tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mV,
                cute.select(self.sV_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdimv),
                1,  # No mcast for now
            )

        tma_atom_O, tma_tensor_O = None, None
        tma_atom_O_load, tma_tensor_O_load = None, None
        if const_expr(self.use_tma_O):
            mO_tma = mO_og if const_expr(self.pack_gqa) else mO
            if const_expr(self.varlen_q):
                mO_tma = copy_utils.create_ragged_tensor_for_tma(
                    mO_tma, ragged_dim=0, ptr_shift=True
                )
            tma_atom_O, tma_tensor_O = make_tiled_tma_atom_fn(
                gmem_tiled_copy_O,
                mO_tma,
                self.sO_store_layout,  # fp32 for non-last call, fp16 for last call
                (self.tile_m, self.tile_hdimv),  # No mcast
            )
            if const_expr(not is_first_call):
                gmem_tiled_copy_O_load = cpasync.CopyBulkTensorTileG2SOp()
                mO_load_tma = mO_og if const_expr(self.pack_gqa) else mO_state
                tma_atom_O_load, tma_tensor_O_load = make_tiled_tma_atom_fn(
                    gmem_tiled_copy_O_load,
                    mO_load_tma,
                    self.sO_layout,  # fp32 (the running state is always fp32)
                    (self.tile_m, self.tile_hdimv),
                )
        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = (
                SingleTileScheduler
                if const_expr(not self.is_causal or self.is_local)
                else SingleTileLPTScheduler
            )
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3])
            if const_expr(mCuSeqlensQ is None)
            else cute.size(mCuSeqlensQ.shape[0] - 1),
            1,  # num_splits
            cute.size(mK.shape[0])
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mQ.shape[1],
            mV.shape[1],
            total_q=cute.size(mQ.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.tile_m, self.tile_n),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead
            if const_expr(self.pack_gqa)
            else 1,
            element_size=self.dtype.width // 8,
            is_persistent=False,
            lpt=self.is_causal or self.is_local,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(
            softmax_scale, self.score_mod
        )
        window_size_left = (
            Int32(window_size_left) if window_size_left is not None else None
        )
        window_size_right = (
            Int32(window_size_right) if window_size_right is not None else None
        )
        fastdiv_mods = utils.compute_fastdiv_mods(
            mQ, mK, self.qhead_per_kvhead, self.pack_gqa, aux_tensors, mPageTable
        )

        self.kernel(
            tma_tensor_Q if const_expr(self.use_tma_Q) else mQ,
            tma_tensor_K if const_expr(self.use_tma_KV) else mK,
            tma_tensor_V if const_expr(self.use_tma_KV) else mV,
            tma_tensor_O if const_expr(self.use_tma_O) else mO,
            # TMA state load into sO, or the state itself when it bypasses smem
            tma_tensor_O_load if const_expr(self.o_state_in_smem) else mO_state,
            mMax,
            mSum,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            tma_atom_O_load,
            softmax_scale_log2,
            softmax_scale,
            window_size_left,
            window_size_right,
            is_first_call,
            is_last_call,
            learnable_sink,
            blocksparse_tensors,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.sO_store_layout,
            self.sP_layout,
            self.gmem_tiled_copy_Q,
            self.gmem_tiled_copy_K,
            self.gmem_tiled_copy_V,
            self.gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            aux_tensors,
            fastdiv_mods,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mO_load: Optional[cute.Tensor],
        mMax: cute.Tensor,
        mSum: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        tma_atom_O_load: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        is_first_call: cutlass.Constexpr[bool],
        is_last_call: cutlass.Constexpr[bool],
        learnable_sink: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sO_store_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        aux_tensors=Optional[list[cute.Tensor]],
        fastdiv_mods=None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # Prefetch tma descriptor
        if warp_idx == 0:
            for tma_atom in (
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_O,
                tma_atom_O_load,
            ):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Mbarrier / pipeline init
        mbar_ptr_Q = storage.mbar_ptr_Q.data_ptr()

        ThreadCooperativeGroup = partial(
            pipeline.CooperativeGroup, pipeline.Agent.Thread
        )
        tma_warp = ThreadCooperativeGroup(1)
        load_threads = ThreadCooperativeGroup(self.num_threads_per_warp_group)
        mma_warps = ThreadCooperativeGroup(self.num_mma_threads // cute.arch.WARP_SIZE)
        if const_expr(self.use_tma_Q):
            pipeline_q = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=mbar_ptr_Q,
                num_stages=1,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["Q"],
                defer_sync=True,
            )
        else:
            pipeline_q = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=mbar_ptr_Q,
                num_stages=1,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )

        if const_expr(self.use_tma_KV):
            pipeline_k = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_K.data_ptr(),
                num_stages=self.num_stages,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["K"],
                defer_sync=True,
            )
            pipeline_v = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_V.data_ptr(),
                num_stages=self.num_stages,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["V"],
                defer_sync=True,
            )
        else:
            pipeline_k = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=storage.mbar_ptr_K.data_ptr(),
                num_stages=self.num_stages,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )
            pipeline_v = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=storage.mbar_ptr_V.data_ptr(),
                num_stages=self.num_stages,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )

        pipeline_o_load = None
        if const_expr(not is_first_call and self.o_state_in_smem):
            pipeline_o_load = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_O_load.data_ptr(),
                num_stages=1,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["O_load"],
                defer_sync=True,
            )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        if const_expr(not self.Q_in_regs):
            sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        else:
            sV = storage.sQ.get_tensor(
                sV_layout.outer, swizzle=sV_layout.inner, dtype=mV.element_type
            )
        # Transpose view of V to tensor with layout (head_dim_v, tile_n) for tiled mma
        sVt = layout_utils.transpose_view(sV)
        sP = None
        if const_expr(sP_layout is not None):
            sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        # fp32 accumulation buffer, when the state is staged through smem
        sO, sO_store = None, None
        if const_expr(self.o_state_in_smem):
            sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
            # epilogue store view: fp32 for non-last calls, fp16/bf16 for a last call
            sO_store = storage.sO.get_tensor(
                sO_store_layout.outer, swizzle=sO_store_layout.inner, dtype=self.o_dtype
            )

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            False,  # is_split_kv
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead
            if const_expr(self.pack_gqa)
            else 1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0]
            if const_expr(not self.pack_gqa)
            else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0]
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            # Don't need to pass in tile_mn because we won't access offset_padded
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead
            if const_expr(self.pack_gqa)
            else 1,
        )
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        # Cluster wait before starting
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        if warp_idx < 4:  # Producer
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            self.load(
                mQ,
                mK,
                mV,
                mO_load,
                sQ,
                sK,
                sV,
                sO if const_expr(not is_first_call and self.o_state_in_smem) else None,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_O_load,
                pipeline_k,
                pipeline_v,
                pipeline_q,
                pipeline_o_load,
                gmem_tiled_copy_Q,
                mPageTable,
                blocksparse_tensors,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
            )

        else:  # Consumer
            cute.arch.setmaxregister_increase(self.num_mma_regs)
            # ///////////////////////////////////////////////////////////////////////////////
            # Tile MMA compute thread partitions and allocate accumulators
            # ///////////////////////////////////////////////////////////////////////////////
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                mMax,
                mSum,
                is_first_call,
                is_last_call,
                mO,
                mO_load,
                mLSE,
                sQ,
                sK,
                sVt,
                sP,
                sO,
                sO_store,
                learnable_sink,
                pipeline_k,
                pipeline_v,
                pipeline_q,
                pipeline_o_load,
                gmem_tiled_copy_O,
                tma_atom_O,
                tidx,
                softmax_scale_log2,
                softmax_scale,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                blocksparse_tensors,
                aux_tensors,
                fastdiv_mods,
            )

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO_load: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sO_load: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O_load: Optional[cute.CopyAtom],
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_o_load: Optional[pipeline.PipelineAsync],
        gmem_tiled_copy_Q: cute.TiledCopy,
        mPageTable: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tidx, _, _ = cute.arch.thread_idx()

        # TMA: only warp 0 loads. cp_async: all warps load.
        # When not use_tma_Q, all 128 producer threads participate in Q loading.
        is_load_warp = warp_idx_in_wg == 0 or const_expr(
            not self.use_tma_KV or not self.use_tma_Q
        )
        # KV loading restricted to warp 0 for TMA, all warps for non-TMA KV
        is_kv_load_warp = warp_idx_in_wg == 0 or const_expr(not self.use_tma_KV)

        if is_load_warp:
            q_producer_phase = Int32(1)
            o_load_producer_phase = Int32(1)
            kv_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_stages
            )
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                # if work_tile.is_valid_tile:
                m_block, head_idx, batch_idx, _ = work_tile.tile_idx
                seqlen = SeqlenInfoCls(batch_idx)
                mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[
                    None, None, head_idx
                ]
                head_idx_kv = (
                    head_idx // self.qhead_per_kvhead
                    if const_expr(not self.pack_gqa)
                    else head_idx
                )

                load_Q = None
                if const_expr(self.use_tma_Q):
                    gQ = cute.local_tile(
                        mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0)
                    )
                    load_Q, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True
                    )

                load_O = None
                if const_expr(sO_load is not None):
                    mO_load_cur = seqlen.offset_batch_Q(mO_load, batch_idx, dim=3)[
                        None, None, head_idx
                    ]
                    gO_load = cute.local_tile(
                        mO_load_cur, (self.tile_m, self.tile_hdimv), (m_block, 0)
                    )
                    load_O, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_O_load,
                        0,
                        cute.make_layout(1),
                        gO_load,
                        sO_load,
                        single_stage=True,
                    )

                paged_kv_manager = None
                tma_load_K_fn = None
                tma_load_V_fn = None
                if const_expr(self.use_tma_KV):
                    # === TMA path (non-paged and paged with page_size == n_block_size) ===
                    if const_expr(mPageTable is not None):
                        # Paged TMA: keep page dimension indexable
                        mK_cur = mK[None, None, head_idx_kv, None]
                        mV_cur = mV[None, None, head_idx_kv, None]
                        gK = cute.local_tile(
                            mK_cur, (self.tile_n, self.tile_hdim), (0, 0, None)
                        )
                        gV = cute.local_tile(
                            mV_cur, (self.tile_n, self.tile_hdimv), (0, 0, None)
                        )
                    else:
                        # Non-paged TMA
                        mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[
                            None, None, head_idx_kv
                        ]
                        mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[
                            None, None, head_idx_kv
                        ]
                        gK = cute.local_tile(
                            mK_cur, (self.tile_n, self.tile_hdim), (None, 0)
                        )
                        gV = cute.local_tile(
                            mV_cur, (self.tile_n, self.tile_hdimv), (None, 0)
                        )
                    # TODO: mcast
                    tma_load_K_fn, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_K, 0, cute.make_layout(1), gK, sK
                    )
                    tma_load_K_fn = copy_utils.tma_producer_copy_fn(
                        tma_load_K_fn, pipeline_k
                    )
                    tma_load_V_fn, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_V, 0, cute.make_layout(1), gV, sV
                    )
                    tma_load_V_fn = copy_utils.tma_producer_copy_fn(
                        tma_load_V_fn, pipeline_v
                    )
                else:
                    # === cp_async path (paged KV with page_size != n_block_size) ===
                    paged_kv_manager = PagedKVManager.create(
                        mPageTable,
                        mK,
                        mV,
                        FastDivmodDivisor(mK.shape[0]),
                        batch_idx,
                        head_idx_kv,
                        tidx,
                        seqlen.seqlen_k,
                        0,  # leftpad_k
                        self.tile_n,
                        self.tile_hdim,
                        self.tile_hdimv,
                        self.num_threads_per_warp_group,
                        mK.element_type,
                        arch=self.arch.major * 10 + self.arch.minor,
                    )

                load_K = partial(
                    self.load_KV,
                    tma_load_K_fn,
                    paged_kv_manager,
                    sK,
                    pipeline_kv=pipeline_k,
                    K_or_V="K",
                )
                load_V = partial(
                    self.load_KV,
                    tma_load_V_fn,
                    paged_kv_manager,
                    sV,
                    pipeline_kv=pipeline_v,
                    K_or_V="V",
                )

                pack_gqa = None
                if const_expr(not self.use_tma_Q):
                    pack_gqa = PackGQA(
                        self.tile_m,
                        self.tile_hdim,
                        self.check_hdim_oob,
                        self.qhead_per_kvhead,
                    )

                if const_expr(not self.use_block_sparsity):
                    n_block_min, n_block_max = block_info.get_n_block_min_max(
                        seqlen, m_block
                    )
                    # if cute.arch.thread_idx()[0] == 0:
                    #     cute.printf("m_block = %d, n_block_min: %d, n_block_max: %d", m_block, n_block_min, n_block_max)
                    # Clamp n_block to 0 when n_block_max == 0 (can happen with causal
                    # + pack_gqa when seqlen_k < tile_n). TMA handles n_block=-1
                    # gracefully (fills zeros), but cp.async would crash on
                    # out-of-bounds page table access.
                    n_block = (
                        n_block_max - 1
                        if const_expr(self.use_tma_KV)
                        else cutlass.max(n_block_max - 1, 0)
                    )
                    page_idx = (
                        mPageTable[batch_idx, n_block]
                        if const_expr(mPageTable is not None and self.use_tma_KV)
                        else None
                    )

                    # First iteration: load K on pipeline_k, Q on pipeline_q
                    if is_kv_load_warp:
                        pipeline_k.producer_acquire(kv_producer_state)
                        if const_expr(not self.use_tma_KV):
                            paged_kv_manager.load_page_table(n_block)
                        load_K(
                            block=n_block,
                            producer_state=kv_producer_state,
                            page_idx=page_idx,
                        )
                    if const_expr(self.use_tma_Q):
                        if warp_idx_in_wg == 0:
                            pipeline_q.producer_acquire_w_index_phase(
                                0, q_producer_phase
                            )
                            load_Q(
                                tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0)
                            )
                            q_producer_phase ^= 1
                            if const_expr(sO_load is not None):
                                pipeline_o_load.producer_acquire_w_index_phase(
                                    0, o_load_producer_phase
                                )
                                load_O(
                                    tma_bar_ptr=pipeline_o_load.sync_object_full.get_barrier(
                                        0
                                    )
                                )
                                o_load_producer_phase ^= 1
                    else:
                        pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                        pack_gqa.load_Q(
                            mQ_cur,
                            sQ,
                            gmem_tiled_copy_Q,
                            tidx,
                            m_block,
                            seqlen.seqlen_q,
                        )
                        cute.arch.cp_async_commit_group()
                        pipeline_q.producer_commit_w_index(0)
                        q_producer_phase ^= 1

                    if is_kv_load_warp:
                        if const_expr(not self.intra_wg_overlap or not self.use_tma_KV):
                            pipeline_v.producer_acquire(kv_producer_state)
                            load_V(
                                block=n_block,
                                producer_state=kv_producer_state,
                                page_idx=page_idx,
                            )
                            kv_producer_state.advance()
                            for i in cutlass.range(
                                n_block_max - 1 - n_block_min, unroll=1
                            ):
                                n_block = n_block_max - 1 - i - 1
                                page_idx = (
                                    mPageTable[batch_idx, n_block]
                                    if const_expr(
                                        mPageTable is not None and self.use_tma_KV
                                    )
                                    else None
                                )
                                if const_expr(not self.use_tma_KV):
                                    paged_kv_manager.load_page_table(n_block)
                                pipeline_k.producer_acquire(kv_producer_state)
                                load_K(
                                    block=n_block,
                                    producer_state=kv_producer_state,
                                    page_idx=page_idx,
                                )
                                pipeline_v.producer_acquire(kv_producer_state)
                                load_V(
                                    block=n_block,
                                    producer_state=kv_producer_state,
                                    page_idx=page_idx,
                                )
                                kv_producer_state.advance()
                        else:
                            for i in cutlass.range(
                                n_block_max - 1 - n_block_min, unroll=1
                            ):
                                n_block_prev = n_block_max - i - 1
                                n_block = n_block_prev - 1
                                page_idx = (
                                    mPageTable[batch_idx, n_block]
                                    if const_expr(mPageTable is not None)
                                    else None
                                )
                                page_idx_prev = (
                                    mPageTable[batch_idx, n_block_prev]
                                    if const_expr(mPageTable is not None)
                                    else None
                                )
                                kv_producer_state_prev = kv_producer_state.clone()
                                kv_producer_state.advance()
                                pipeline_k.producer_acquire(kv_producer_state)
                                load_K(
                                    block=n_block,
                                    producer_state=kv_producer_state,
                                    page_idx=page_idx,
                                )
                                pipeline_v.producer_acquire(kv_producer_state_prev)
                                load_V(
                                    block=n_block_prev,
                                    producer_state=kv_producer_state_prev,
                                    page_idx=page_idx_prev,
                                )
                            n_block = n_block_min
                            page_idx = (
                                mPageTable[batch_idx, n_block]
                                if const_expr(mPageTable is not None)
                                else None
                            )
                            pipeline_v.producer_acquire(kv_producer_state)
                            load_V(
                                block=n_block,
                                producer_state=kv_producer_state,
                                page_idx=page_idx,
                            )
                            kv_producer_state.advance()
                else:
                    # Block sparsity: use TMA closures directly (not paged)
                    # Load Q on pipeline_q, separate from K/V pipeline
                    if const_expr(self.use_tma_Q):
                        if warp_idx_in_wg == 0:
                            pipeline_q.producer_acquire_w_index_phase(
                                0, q_producer_phase
                            )
                            load_Q(
                                tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0)
                            )
                            q_producer_phase ^= 1
                            if const_expr(sO_load is not None):
                                load_O_state = partial(
                                    self.load_O_state_tma, pipeline_o_load, load_O
                                )
                                if const_expr(not self.skip_empty_tiles):
                                    o_load_producer_phase = load_O_state(
                                        o_load_producer_phase
                                    )
                                else:
                                    # A skipped tile (see skip_empty_tiles) loads no state.
                                    if self._tile_has_blocks(
                                        blocksparse_tensors,
                                        batch_idx,
                                        head_idx,
                                        m_block,
                                        seqlen,
                                    ):
                                        o_load_producer_phase = load_O_state(
                                            o_load_producer_phase
                                        )
                    else:
                        pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                        pack_gqa.load_Q(
                            mQ_cur,
                            sQ,
                            gmem_tiled_copy_Q,
                            tidx,
                            m_block,
                            seqlen.seqlen_q,
                        )
                        cute.arch.cp_async_commit_group()
                        pipeline_q.producer_commit_w_index(0)
                        q_producer_phase ^= 1
                    if is_kv_load_warp:
                        kv_producer_state = produce_block_sparse_loads(
                            blocksparse_tensors,
                            batch_idx,
                            head_idx,
                            m_block,
                            seqlen,
                            kv_producer_state,
                            tma_load_K_fn,
                            tma_load_V_fn,
                            pipeline_k,
                            pipeline_v,
                            self.intra_wg_overlap,
                            self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                            self.q_subtile_factor
                            if self.q_subtile_factor is not None
                            else 1,
                        )

                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
                # End of persistent scheduler loop

            # Producer tail is only useful for cluster to avoid early exit of blocks.
            # We only need producer_tail on V since that's the last that's loaded, we don't
            # need it for Q (no cluster) and K.
            if is_kv_load_warp:
                pipeline_v.producer_tail(kv_producer_state)

    @cute.jit
    def load_O_state_tma(
        self,
        pipeline_o_load: pipeline.PipelineAsync,
        load_O: Callable,
        o_load_producer_phase: Int32,
    ) -> Int32:
        """TMA-load the fp32 O state tile into sO; returns the next producer phase."""
        pipeline_o_load.producer_acquire_w_index_phase(0, o_load_producer_phase)
        load_O(tma_bar_ptr=pipeline_o_load.sync_object_full.get_barrier(0))
        return o_load_producer_phase ^ 1

    @cute.jit
    def _tile_has_blocks(
        self,
        blocksparse_tensors: BlockSparseTensors,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen: SeqlenInfoQK,
    ):
        return tile_has_blocks(
            blocksparse_tensors,
            batch_idx,
            head_idx,
            m_block,
            seqlen,
            self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            self.q_subtile_factor if self.q_subtile_factor is not None else 1,
        )

    @cute.jit
    def load_KV(
        self,
        tma_load_fn: Optional[Callable],
        paged_kv_manager: Optional[PagedKVManager],
        sX: cute.Tensor,
        block: Int32,
        pipeline_kv: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        K_or_V: Literal["K", "V"],
        page_idx: Optional[Int32] = None,
    ):
        if const_expr(self.use_tma_KV):
            src_idx = block if const_expr(page_idx is None) else page_idx
            tma_load_fn(src_idx=src_idx, producer_state=producer_state)
        else:
            paged_kv_manager.load_KV(
                block, sX[None, None, producer_state.index], K_or_V
            )
            cute.arch.cp_async_commit_group()
        pipeline_kv.producer_commit(producer_state)

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        mMax: cute.Tensor,
        mSum: cute.Tensor,
        is_first_call: cutlass.Constexpr,
        is_last_call: cutlass.Constexpr,
        mO: cute.Tensor,
        mO_load: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sVt: cute.Tensor,
        sP: Optional[cute.Tensor],
        sO: cute.Tensor,
        sO_store: cute.Tensor,
        learnable_sink: Optional[cute.Tensor],
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_o_load: Optional[pipeline.PipelineAsync],
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        tidx: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
        aux_tensors: Optional[list],
        fastdiv_mods=None,
    ):
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        warp_group_thread_layout = cute.make_layout(
            self.num_wg_mma, stride=self.num_threads_per_warp_group
        )
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx))
        _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
            wg_mma_qk, (self.tile_m, self.tile_n, self.tile_hdim), sQ, sK
        )
        mma_qk_fn = partial(
            sm90_utils.gemm_zero_init,
            tiled_mma_qk,
            (self.tile_m, self.tile_n),
            tSrQ,
            tSrK,
        )
        acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(
            wg_mma_pv, (self.tile_m, self.tile_hdimv, self.tile_n), sP, sVt
        )
        mma_pv_fn = partial(sm90_utils.gemm_w_idx, tiled_mma_pv, acc_O, tOrP, tOrVt)

        # ///////////////////////////////////////////////////////////////////////////////
        # Smem copy atom tiling
        # ///////////////////////////////////////////////////////////////////////////////
        smem_copy_atom_P = utils.get_smem_store_atom(
            self.arch.major * 10 + self.arch.minor, self.dtype
        )
        smem_thr_copy_P = cute.make_tiled_copy_C(
            smem_copy_atom_P, tiled_mma_qk
        ).get_slice(tidx)
        tPsP = smem_thr_copy_P.partition_D(sP) if const_expr(sP is not None) else None
        smem_copy_params = SimpleNamespace(smem_thr_copy_P=smem_thr_copy_P, tPsP=tPsP)

        self.mma_init()

        q_consumer_phase = Int32(0)
        o_load_consumer_phase = Int32(0)
        kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_stages
        )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        softmax = Softmax2TheSequel.create(
            softmax_scale_log2,
            num_rows=acc_O.shape[0][0] * acc_O.shape[1],
            arch=90,
            softmax_scale=softmax_scale,
        )

        # For RescaleOBeforeGemm: persistent scores_scale across iterations
        scores_scale = None
        if const_expr(self.rescale_O_before_gemm):
            scores_scale = cute.make_rmem_tensor_like(softmax.row_max, Float32)

        mma_one_n_block_all = partial(
            self.mma_one_n_block_intrawg_overlap
            if const_expr(self.intra_wg_overlap)
            else self.mma_one_n_block,
            mma_qk_fn=mma_qk_fn,
            pipeline_k=pipeline_k,
            pipeline_v=pipeline_v,
            acc_O=acc_O,
            tOrP=tOrP,
            smem_copy_params=smem_copy_params,
            check_inf=True,
            scores_scale=scores_scale,
        )

        process_first_half_block = partial(
            self.first_half_block_overlap,
            mma_qk_fn=mma_qk_fn,
            pipeline_k=pipeline_k,
            tOrP=tOrP,
            smem_copy_params=smem_copy_params,
            scores_scale=scores_scale,
            softmax=softmax,
            acc_O=acc_O,
            is_first_block=is_first_call,
        )
        process_last_half_block = partial(
            self.last_half_block_overlap,
            pipeline_v=pipeline_v,
            mma_pv_fn=mma_pv_fn,
            scores_scale=scores_scale,
            softmax=softmax,
            acc_O=acc_O,
        )
        while work_tile.is_valid_tile:
            # if work_tile.is_valid_tile:

            # shape: (atom_v_m * rest_m)
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            # Recompute fastdiv_mods if necessary for varlen with aux_tensors
            recompute_fastdiv_mods_q = cutlass.const_expr(
                aux_tensors is not None
                and (seqlen.has_cu_seqlens_q or seqlen.has_seqused_q)
            )
            recompute_fastdiv_mods_k = cutlass.const_expr(
                aux_tensors is not None
                and (seqlen.has_cu_seqlens_k or seqlen.has_seqused_k)
            )
            if cutlass.const_expr(fastdiv_mods is not None):
                seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                fastdiv_mods = (
                    seqlen_q_divmod
                    if not recompute_fastdiv_mods_q
                    else FastDivmodDivisor(seqlen.seqlen_q),
                    seqlen_k_divmod
                    if not recompute_fastdiv_mods_k
                    else FastDivmodDivisor(seqlen.seqlen_k),
                )

            mask = AttentionMaskCls(seqlen)
            mask_fn = partial(
                mask.apply_mask,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=m_block,
                thr_mma=thr_mma_qk,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                # Upstream AttentionMask.apply_mask (b21+) takes an AuxData
                # struct, not a raw tensor list.
                aux_data=utils.AuxData(tensors=aux_tensors),
                fastdiv_mods=fastdiv_mods,
            )
            score_mod_fn = None
            if const_expr(self.score_mod is not None):
                score_mod_fn = partial(
                    self.apply_score_mod,
                    thr_mma_qk,
                    batch_idx,
                    head_idx,
                    m_block,
                    softmax_scale=softmax_scale,
                    aux_tensors=aux_tensors,
                    fastdiv_mods=fastdiv_mods,
                )
            mma_one_n_block = partial(
                mma_one_n_block_all,
                seqlen=seqlen,
                softmax=softmax,
                score_mod_fn=score_mod_fn,
            )
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
            pipeline_q.consumer_wait_w_index_phase(0, q_consumer_phase)

            # A skipped tile (see skip_empty_tiles) neither loads nor stores state.
            tile_has_work = None
            if const_expr(self.skip_empty_tiles):
                tile_has_work = self._tile_has_blocks(
                    blocksparse_tensors, batch_idx, head_idx, m_block, seqlen
                )

            # For performance reason, we separate out two kinds of iterations:
            # those that need masking on S, and those that don't.
            # We need masking on S for the very last block when K and V has length not multiple of tile_n.
            # We also need masking on S if it's causal, for the last several blocks.
            # softmax.reset()  # Don't need reset as we explicitly call softmax w is_first=True
            O_should_accumulate = not is_first_call

            # On the dense intra-warpgroup-overlap path with the state in smem,
            # process_first_half_block waits for the TMA-loaded O state only
            # once acc_O is first needed, after the first QK GEMM and softmax.
            defer_O_load = (
                self.intra_wg_overlap
                and self.o_state_in_smem
                and not self.use_block_sparsity
            )
            load_O = None
            if const_expr(not is_first_call):
                load_O = partial(
                    self.load_O_state,
                    acc_O, mO_load, sO, pipeline_o_load, o_load_consumer_phase,
                    tiled_mma_pv, seqlen, tidx, m_block, head_idx, batch_idx,
                )  # fmt: skip
                if const_expr(defer_O_load):
                    pass  # loaded by process_first_half_block
                elif const_expr(tile_has_work is None):
                    load_O()
                else:
                    if tile_has_work:
                        load_O()
                # Row max/sum stay outside the branch: loaded inside it, values
                # built there get reused after it and IR verification fails.
                softmax.load(
                    mMax,
                    mSum,
                    seqlen,
                    tiled_mma_pv,
                    self.tile_m,
                    self.tile_hdimv,
                    batch_idx,
                    head_idx,
                    tidx,
                    m_block,
                )
                # acc_O was stored as unnormalized fp32 — no un-normalization needed.

            # ==========================================
            # MAINLOOP
            # ==========================================
            if const_expr(not self.use_block_sparsity):
                # ==========================================
                # No block-sparsity (original path)
                # ==========================================
                # First iteration with seqlen masking
                if const_expr(self.intra_wg_overlap):
                    kv_consumer_state = process_first_half_block(
                        n_block=n_block_max - 1,
                        seqlen=seqlen,
                        kv_consumer_state=kv_consumer_state,
                        mask_fn=partial(mask_fn, mask_mod=self.mask_mod),
                        score_mod_fn=score_mod_fn,
                        is_first_block=is_first_call,
                        load_O_fn=load_O if const_expr(defer_O_load) else None,
                    )
                else:
                    self.warp_scheduler_barrier_sync()
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state,
                        n_block=n_block_max - 1,
                        seqlen=seqlen,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                        is_first_n_block=is_first_call,
                        mask_fn=partial(
                            mask_fn, mask_mod=self.mask_mod, mask_seqlen=True
                        ),
                    )
                    O_should_accumulate = True
                # if cute.arch.thread_idx()[0] == 128: cute.printf("m_block = {}, n_block_max = {}, n_block_min = {}", m_block, n_block_max, n_block_min)
                n_block_max -= 1
                # Next couple of iterations with causal masking
                if const_expr(self.is_causal or self.is_local):
                    n_block_min_causal_local_mask = (
                        block_info.get_n_block_min_causal_local_mask(
                            seqlen, m_block, n_block_min
                        )
                    )
                    # if cute.arch.thread_idx()[0] == 128: cute.printf("n_block_min_causal_local_mask = {}", n_block_min_causal_local_mask)
                    for n_tile in cutlass.range(
                        n_block_max - n_block_min_causal_local_mask, unroll=1
                    ):
                        kv_consumer_state = mma_one_n_block(
                            kv_consumer_state,
                            n_block=n_block_max - 1 - n_tile,
                            seqlen=seqlen,
                            mma_pv_fn=partial(
                                mma_pv_fn, zero_init=not O_should_accumulate
                            ),
                            mask_fn=partial(
                                mask_fn, mask_mod=self.mask_mod, mask_seqlen=False
                            ),
                        )
                        O_should_accumulate = True
                    n_block_max = cutlass.min(
                        n_block_max, n_block_min_causal_local_mask
                    )
                # Interior blocks need no causal/local masking, but custom mask_mod still applies.
                n_block_min_before_local_mask = (
                    block_info.get_n_block_min_before_local_mask(
                        seqlen, m_block, n_block_min
                    )
                )
                # if cute.arch.thread_idx()[0] == 128: cute.printf("n_block_min_before_local_mask = {}, n_block_min = {}", n_block_min_before_local_mask, n_block_min)
                for n_tile in cutlass.range(
                    n_block_max - n_block_min_before_local_mask, unroll=1
                ):
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state,
                        n_block=n_block_max - 1 - n_tile,
                        seqlen=seqlen,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                        mask_fn=partial(
                            mask_fn, mask_mod=self.mask_mod, mask_seqlen=False
                        )
                        if const_expr(self.mask_mod is not None)
                        else None,
                    )
                    O_should_accumulate = True
                # Separate iterations with local masking on the left
                if const_expr(
                    self.is_local and block_info.window_size_left is not None
                ):
                    n_block_max = cutlass.min(
                        n_block_max, n_block_min_before_local_mask
                    )
                    for n_tile in cutlass.range(n_block_max - n_block_min, unroll=1):
                        kv_consumer_state = mma_one_n_block(
                            kv_consumer_state,
                            n_block=n_block_max - 1 - n_tile,
                            seqlen=seqlen,
                            mma_pv_fn=partial(
                                mma_pv_fn, zero_init=not O_should_accumulate
                            ),
                            mask_fn=partial(
                                mask_fn, mask_mod=self.mask_mod, mask_seqlen=False
                            ),
                        )
                        O_should_accumulate = True
                # Release Q pipeline so the producer can load the next tile's Q
                pipeline_q.consumer_release_w_index(0)
                if const_expr(pipeline_o_load is not None):
                    pipeline_o_load.consumer_release_w_index(0)
                # Last "half" iteration
                if const_expr(self.intra_wg_overlap):
                    kv_consumer_state = process_last_half_block(
                        kv_consumer_state=kv_consumer_state,
                        zero_init=not O_should_accumulate,
                    )
                    O_should_accumulate = True
                else:
                    self.warp_scheduler_barrier_arrive()

            else:
                # ==========================================
                # Block sparsity
                # ==========================================
                kv_consumer_state, O_should_accumulate, processed_any = (
                    consume_block_sparse_loads(
                        blocksparse_tensors,
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
                        self.mask_mod,
                        fastdiv_mods,
                        self.intra_wg_overlap,
                        self.warp_scheduler_barrier_sync,
                        self.warp_scheduler_barrier_arrive,
                        self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                        self.q_subtile_factor
                        if self.q_subtile_factor is not None
                        else 1,
                        is_first_call,
                    )
                )

                # Release Q pipeline so the producer can load the next tile's Q
                pipeline_q.consumer_release_w_index(0)
                if const_expr(pipeline_o_load is not None):
                    if const_expr(tile_has_work is None):
                        pipeline_o_load.consumer_release_w_index(0)
                    else:
                        if tile_has_work:
                            pipeline_o_load.consumer_release_w_index(0)

                # Handle empty case (when no blocks to process) (and this is the first call)
                if const_expr(is_first_call):
                    if not processed_any:
                        softmax.reset()
                        acc_O.fill(0.0)

            q_consumer_phase ^= 1
            if const_expr(pipeline_o_load is not None):
                if const_expr(tile_has_work is None):
                    o_load_consumer_phase ^= 1
                else:
                    if tile_has_work:
                        o_load_consumer_phase ^= 1

            sink_val = None
            if const_expr(learnable_sink is not None):
                if const_expr(not self.pack_gqa):
                    sink_val = Float32(learnable_sink[head_idx])
                else:  # Each thread might have a different sink value due to different q_head
                    sink_val = cute.make_rmem_tensor_like(softmax.row_max, Float32)
                    cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
                    tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma_qk.partition_C(cS))
                    for r in cutlass.range(cute.size(sink_val), unroll_full=True):
                        row = m_block * self.tile_m + tScS_mn[r][0]
                        q_head_idx = (
                            row % self.qhead_per_kvhead
                            + head_idx * self.qhead_per_kvhead
                        )
                        sink_val[r] = Float32(learnable_sink[q_head_idx])

            # normalize acc_O by row_sum and calculate the lse
            if const_expr(is_last_call):  # Only normalise if this is the last call
                row_scale = softmax.finalize(sink_val=sink_val)
                softmax.rescale_O(acc_O, row_scale)

            # ///////////////////////////////////////////////////////////////////////////////
            # Epilogue
            # ///////////////////////////////////////////////////////////////////////////////
            epilogue = partial(
                self.epilogue,
                acc_O,
                softmax.row_max,
                softmax.row_sum,
                mMax,
                mSum,
                mO,
                mLSE,
                sO_store,
                seqlen,
                gmem_tiled_copy_O,
                tma_atom_O,
                tiled_mma_pv,
                tidx,
                m_block,
                head_idx,
                batch_idx,
                is_last_call,
            )
            if const_expr(tile_has_work is None):
                epilogue()
            else:
                if tile_has_work:
                    epilogue()

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def first_half_block_overlap(
        self,
        n_block: Int32,
        mma_qk_fn: Callable,
        kv_consumer_state,
        pipeline_k,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax2TheSequel,
        seqlen: SeqlenInfoQK,
        scores_scale: Optional[cute.Tensor] = None,
        acc_O: Optional[cute.Tensor] = None,
        mask_fn: Callable = None,
        score_mod_fn: Optional[Callable] = None,
        is_first_block: bool = False,
        load_O_fn: Optional[Callable] = None,
    ):
        """Processes the first half block when using intra-warpgroup-overlap

        ``load_O_fn`` (resumed online calls) loads the O state into acc_O just
        before acc_O is first used, so the first QK GEMM and softmax overlap
        its TMA load.
        """

        pipeline_k.consumer_wait(
            kv_consumer_state, pipeline_k.consumer_try_wait(kv_consumer_state)
        )
        acc_S = mma_qk_fn(B_idx=kv_consumer_state.index, wg_wait=0)
        pipeline_k.consumer_release(kv_consumer_state)

        # Apply score modification if present
        if const_expr(score_mod_fn is not None):
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)

        # Apply mask; mask_seqlen always True for first block
        # Caveat: if full block further right than mask block, seqlen masking is redundant;
        # however, masking is being applied anyway, so essentially no perf hit
        mask_fn(acc_S, n_block=n_block, mask_seqlen=True)

        row_scale = softmax.online_softmax(acc_S, is_first=is_first_block)

        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
        )
        tOrP_cur.store(tOrP_acc.load().to(self.dtype))

        if const_expr(not self.mma_pv_is_rs):
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
            # Fence and barrier to make smem store visible to WGMMA
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()

        if const_expr(load_O_fn is not None):
            load_O_fn()

        # For RescaleOBeforeGemm the rescale is deferred to just before the next
        # PV GEMM. On a resumed (non-first) online call acc_O holds the loaded
        # state, so it is neither zeroed nor rescaled here.
        if const_expr(self.rescale_O_before_gemm):
            if const_expr(is_first_block):
                acc_O.fill(0.0)
            scores_scale.store(row_scale.load())
        elif const_expr(not is_first_block):
            softmax.rescale_O(acc_O, row_scale)

        return kv_consumer_state

    @cute.jit
    def last_half_block_overlap(
        self,
        kv_consumer_state,
        pipeline_v,
        mma_pv_fn: Callable,
        zero_init: bool,
        scores_scale: Optional[cute.Tensor] = None,
        softmax: Optional[Softmax2TheSequel] = None,
        acc_O: Optional[cute.Tensor] = None,
    ):
        """Processes the final PV GEMM when using intra-warpgroup-overlap"""

        # For RescaleOBeforeGemm: rescale O before the final PV GEMM
        if const_expr(self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, scores_scale)

        pipeline_v.consumer_wait(
            kv_consumer_state, pipeline_v.consumer_try_wait(kv_consumer_state)
        )
        mma_pv_fn(B_idx=kv_consumer_state.index, zero_init=zero_init, wg_wait=0)
        pipeline_v.consumer_release(kv_consumer_state)
        kv_consumer_state.advance()
        return kv_consumer_state

    @cute.jit
    def mma_one_n_block(
        self,
        smem_pipe_read: pipeline.PipelineState | pipeline_custom.PipelineStateSimple,
        n_block: Int32,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax2TheSequel,
        seqlen: SeqlenInfoQK,
        scores_scale: Optional[cute.Tensor] = None,  # not used
        score_mod_fn: Optional[Callable] = None,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
    ):
        pipeline_k.consumer_wait(
            smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read)
        )
        # S = Q @ K.T
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(0)
        pipeline_k.consumer_release(smem_pipe_read)

        # handle score mods and masking
        if const_expr(score_mod_fn is not None):
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
        if const_expr(mask_fn is not None):
            mask_fn(acc_S=acc_S, n_block=n_block)

        row_scale = softmax.online_softmax(
            acc_S, is_first=is_first_n_block, check_inf=check_inf
        )
        # if cute.arch.thread_idx()[0] == 0: cute.print_tensor(layout_utils.reshape_acc_to_mn(acc_S))
        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
        )
        # tOrP.store(tOrP_acc.load().to(self.dtype))
        # the "to(self.dtype)" conversion fails to vectorize for block sizes other
        # than 128 x 128, i.e. it calls convert on 1 fp32 element at a time instead of
        # 2 elements. So we just call ptx directly.
        utils.cvt_f16(tOrP_acc, tOrP_cur)
        if const_expr(not self.mma_pv_is_rs):
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
        softmax.rescale_O(acc_O, row_scale)
        if const_expr(not self.mma_pv_is_rs):
            # Fence and barrier to make sure smem store is visible to WGMMA
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()  # Only need syncwarp since each warp is using its own P values for MmaPV
        pipeline_v.consumer_wait(
            smem_pipe_read, pipeline_v.consumer_try_wait(smem_pipe_read)
        )
        self.warp_scheduler_barrier_sync()
        # O += P @ V
        mma_pv_fn(B_idx=smem_pipe_read.index, wg_wait=0)
        pipeline_v.consumer_release(smem_pipe_read)
        smem_pipe_read.advance()
        return smem_pipe_read

    @cute.jit
    def mma_one_n_block_intrawg_overlap(
        self,
        smem_pipe_read: pipeline.PipelineState | pipeline_custom.PipelineStateSimple,
        n_block: Int32,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax2TheSequel,
        seqlen: SeqlenInfoQK,
        scores_scale: Optional[cute.Tensor] = None,
        score_mod_fn: Optional[Callable] = None,
        mask_fn: Optional[Callable] = None,
        check_inf: cutlass.Constexpr = True,
    ):
        smem_pipe_read_v = smem_pipe_read.clone()
        smem_pipe_read.advance()
        pipeline_k.consumer_wait(
            smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read)
        )
        self.warp_scheduler_barrier_sync()
        # S = Q @ K.T
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        # RescaleOBeforeGemm: rescale O while QK GEMM is in flight, before PV GEMM
        if const_expr(self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, scores_scale)
        pipeline_v.consumer_wait(
            smem_pipe_read_v, pipeline_v.consumer_try_wait(smem_pipe_read_v)
        )
        # O += P @ V
        mma_pv_fn(B_idx=smem_pipe_read_v.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(1)
        pipeline_k.consumer_release(smem_pipe_read)

        # handle score mods and masking
        if const_expr(score_mod_fn is not None):
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
        if const_expr(mask_fn is not None):
            mask_fn(acc_S=acc_S, n_block=n_block)
        # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(layout_utils.reshape_acc_to_mn(acc_S))

        row_scale = softmax.online_softmax(acc_S, check_inf=check_inf)
        warpgroup.wait_group(0)
        pipeline_v.consumer_release(smem_pipe_read_v)
        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
        )
        # tOrP_cur.store(tOrP_acc.load().to(self.dtype))
        # the "to(self.dtype)" conversion fails to vectorize for block sizes other
        # than 128 x 128, i.e. it calls convert on 1 fp32 element at a time instead of
        # 2 elements. So we just call ptx directly.
        utils.cvt_f16(tOrP_acc, tOrP_cur)
        if const_expr(not self.mma_pv_is_rs):
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
        if const_expr(not self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, row_scale)
        if const_expr(self.rescale_O_before_gemm):
            scores_scale.store(row_scale.load())
        if const_expr(not self.mma_pv_is_rs):
            # Fence and barrier to make sure smem store is visible to WGMMA
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()  # Only need syncwarp since each warp is using its own P values for MmaPV
        return smem_pipe_read

    @cute.jit
    def mma_init(self):
        warp_group_idx = utils.canonical_warp_group_idx(sync=False)
        if const_expr(self.use_scheduler_barrier):
            if warp_group_idx == 1:
                cute.arch.barrier_arrive(
                    barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1),
                    number_of_threads=2 * self.num_threads_per_warp_group,
                )

    @cute.jit
    def apply_score_mod(
        self,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        acc_S,
        n_block,
        softmax_scale,
        seqlen,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=None,
    ):
        # Prepare index tensor
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        cS = cute.domain_offset((m_block * self.tile_m, n_block * self.tile_n), cS)
        tScS = thr_mma_qk.partition_C(cS)

        apply_score_mod_inner(
            acc_S,
            tScS,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax_scale,
            self.vec_size,
            self.qk_acc_dtype,
            # Upstream apply_score_mod_inner (b21+) takes an AuxData struct,
            # not a raw tensor list.
            utils.AuxData(tensors=aux_tensors),
            fastdiv_mods,
            seqlen_info=seqlen,
            constant_q_idx=None,
            qhead_per_kvhead=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )

    def warp_scheduler_barrier_sync(self):
        if const_expr(self.use_scheduler_barrier):
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1)
                - 1
                + utils.canonical_warp_group_idx(sync=False),
                number_of_threads=2 * self.num_threads_per_warp_group,
            )

    def warp_scheduler_barrier_arrive(self):
        if const_expr(self.use_scheduler_barrier):
            assert self.num_wg_mma in [2, 3]
            cur_wg = utils.canonical_warp_group_idx(sync=False) - 1
            if const_expr(self.num_wg_mma == 2):
                next_wg = 1 - cur_wg
            else:
                t = cur_wg + 1
                next_wg = t % self.num_wg_mma
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + next_wg,
                number_of_threads=2 * self.num_threads_per_warp_group,
            )

    @cute.jit
    def epilogue(
        self,
        acc_O: cute.Tensor,
        row_max: cute.Tensor,
        row_sum_or_lse: cute.Tensor,  # This is the lse in the last call, but row_sum in every other call
        mMax: cute.Tensor,
        mSum: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sO_store: cute.Tensor,
        seqlen: SeqlenInfoQK,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        is_last_call: cutlass.Constexpr[bool],
    ):
        # Non-last calls store the fp32 accumulation. A last call stores acc_O
        # (already normalized) in mO's dtype, which may be fp16/bf16.
        rO = acc_O
        if const_expr(self.o_dtype != Float32):
            rO = cute.make_fragment_like(acc_O, self.o_dtype)
            rO.store(acc_O.load().to(self.o_dtype))
        smem_copy_atom_O = utils.get_smem_store_atom(
            self.arch.major * 10 + self.arch.minor, self.o_dtype
        )
        # Also warp-reduce row_sum for write_lse on non-last call (was previously
        # done as part of normalization; keep it for the max/sum persistence path).
        if const_expr(not is_last_call):
            row_sum_or_lse.store(
                utils.warp_reduce(row_sum_or_lse.load(), operator.add, width=4)
            )
        if const_expr(self.o_state_in_smem):
            # Make sure all threads have finished reading V
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_epilogue_threads,
            )
            smem_thr_copy_O = cute.make_tiled_copy_C(
                smem_copy_atom_O, tiled_mma
            ).get_slice(tidx)
            taccOrO = smem_thr_copy_O.retile(rO)
            taccOsO = smem_thr_copy_O.partition_D(sO_store)
            cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        pack_gqa = PackGQA(
            self.tile_m, self.tile_hdimv, self.check_hdim_v_oob, self.qhead_per_kvhead
        )

        if const_expr(
            mLSE is not None and is_last_call
        ):  # Only save LSE on the last call
            self.write_lse(
                row_sum_or_lse,
                cO,
                mLSE,
                seqlen,
                tiled_mma,
                pack_gqa,
                tidx,
                m_block,
                head_idx,
                batch_idx,
            )

        if const_expr(
            not is_last_call
        ):  # Only write row_max/row_sum if we're not in the last call
            # Note: row_sum_or_lse was already warp-reduced above (for normalization)
            self.write_lse(
                row_max,
                cO,
                mMax,
                seqlen,
                tiled_mma,
                pack_gqa,
                tidx,
                m_block,
                head_idx,
                batch_idx,
            )
            self.write_lse(
                row_sum_or_lse,
                cO,
                mSum,
                seqlen,
                tiled_mma,
                pack_gqa,
                tidx,
                m_block,
                head_idx,
                batch_idx,
            )

        ragged = self.use_tma_O and (seqlen.has_cu_seqlens_q or seqlen.has_seqused_q)
        mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3, ragged=ragged)[
            None, None, head_idx
        ]
        # thr_mma = tiled_mma.get_slice(tidx)
        # taccOgO = thr_mma.partition_C(gO)
        # cute.autovec_copy(rO, taccOgO)
        # sync to make sure all smem stores are done
        if const_expr(not self.o_state_in_smem):
            self.copy_O_state(
                mO, rO, tiled_mma.get_slice(tidx), seqlen, m_block, head_idx,
                batch_idx, to_gmem=True,
            )  # fmt: skip
        elif const_expr(self.use_tma_O):
            # ensure smem writes are visible to TMA
            cute.arch.fence_view_async_shared()
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE,
            )
            gO = cute.local_tile(mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
            store_O, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_O, 0, cute.make_layout(1), sO_store, gO, single_stage=True
            )
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            if warp_idx == 4:
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwd.Epilogue),
                    number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE,
                )
                store_O()
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
        else:
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_epilogue_threads,
            )
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            tOsO = gmem_thr_copy_O.partition_S(sO_store)
            tOrO = cute.make_fragment_like(tOsO, sO_store.element_type)
            # load acc O from smem to rmem for wider vectorization
            cute.autovec_copy(tOsO, tOrO)
            if const_expr(not self.pack_gqa):
                gO = cute.local_tile(
                    mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0)
                )
                tOgO = gmem_thr_copy_O.partition_D(gO)
                tOcO = gmem_thr_copy_O.partition_S(cO)
                t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
                tOpO = utils.predicate_k(tOcO, limit=mO.shape[1])
                # copy acc O from rmem to gmem
                for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                    if (
                        t0OcO[0, rest_m, 0][0]
                        < seqlen.seqlen_q - m_block * self.tile_m - tOcO[0][0]
                    ):
                        cute.copy(
                            gmem_tiled_copy_O,
                            tOrO[None, rest_m, None],
                            tOgO[None, rest_m, None],
                            pred=tOpO[None, rest_m, None]
                            if const_expr(self.check_hdim_v_oob)
                            else None,
                        )
            else:
                pack_gqa.store_O(
                    mO_cur, tOrO, gmem_tiled_copy_O, tidx, m_block, seqlen.seqlen_q
                )

    @cute.jit
    def load_O_state(
        self,
        acc_O: cute.Tensor,
        mO: cute.Tensor,
        sO: Optional[cute.Tensor],
        pipeline_o_load: Optional[pipeline.PipelineAsync],
        o_load_consumer_phase: Int32,
        tiled_mma_pv: cute.TiledMma,
        seqlen: SeqlenInfoQK,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
    ):
        """Load a resumed call's fp32 O tile into acc_O (no conversion needed).

        ``mO`` is the fp32 state itself; it is only read when the state bypasses
        shared memory (otherwise the producer TMA-loads it into ``sO``).
        """
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        if const_expr(pipeline_o_load is not None):
            pipeline_o_load.consumer_wait_w_index_phase(0, o_load_consumer_phase)
            tOsO = thr_mma_pv.partition_C(sO)
            cute.autovec_copy(tOsO, acc_O)
        else:
            self.copy_O_state(
                mO, acc_O, thr_mma_pv, seqlen, m_block, head_idx, batch_idx,
                to_gmem=False,
            )  # fmt: skip

    @cute.jit
    def copy_O_state(
        self,
        mO: cute.Tensor,
        acc_O: cute.Tensor,
        thr_mma,
        seqlen: SeqlenInfoQK,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        to_gmem: cutlass.Constexpr[bool],
    ):
        """Move the O tile between global memory and the accumulator.

        Used when the online state does not fit in shared memory (head_dim >
        128). Loads read the fp32 state; stores write ``acc_O``'s dtype (fp32
        state, or the fp16/bf16 output of a last call). Each thread copies its
        own accumulator fragment; rows past ``seqlen_q`` (and columns past
        head_dim_v) are skipped on store and read as 0 on load, as the TMA path
        does.
        """
        mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]
        gO = cute.local_tile(mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
        tOgO = thr_mma.partition_C(gO)
        rows_left = seqlen.seqlen_q - m_block * self.tile_m
        tOcO = thr_mma.partition_C(
            cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        )
        if const_expr(not self.check_hdim_v_oob):
            if rows_left >= self.tile_m:
                # Full tile: one vectorized copy per fragment.
                if const_expr(to_gmem):
                    cute.autovec_copy(acc_O, tOgO)
                else:
                    cute.autovec_copy(tOgO, acc_O)
            else:
                self._copy_O_state_predicated(
                    tOgO, acc_O, tOcO, rows_left, mO.shape[1], to_gmem
                )
        else:
            self._copy_O_state_predicated(
                tOgO, acc_O, tOcO, rows_left, mO.shape[1], to_gmem
            )

    @cute.jit
    def _copy_O_state_predicated(
        self,
        tOgO: cute.Tensor,
        acc_O: cute.Tensor,
        tOcO: cute.Tensor,
        rows_left: Int32,
        cols: Int32,
        to_gmem: cutlass.Constexpr[bool],
    ):
        for i in cutlass.range_constexpr(cute.size(acc_O)):
            if const_expr(not to_gmem):
                acc_O[i] = Float32(0.0)
            if tOcO[i][0] < rows_left:
                if tOcO[i][1] < cols:
                    if const_expr(to_gmem):
                        tOgO[i] = acc_O[i]
                    else:
                        acc_O[i] = tOgO[i]

    @cute.jit
    def write_lse(
        self,
        lse: cute.Tensor,
        cO: cute.Tensor,
        mLSE: cute.Tensor,
        seqlen: SeqlenInfoQK,
        tiled_mma: cute.TiledMma,
        pack_gqa: PackGQA,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
    ):
        # Write LSE from rmem -> gmem
        mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2)[None, head_idx]
        if const_expr(not self.pack_gqa):
            gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
            gLSE_expanded_layout = cute.append(
                gLSE.layout, cute.make_layout((self.tile_hdimv,), stride=(0,))
            )
            gLSE_expanded = cute.make_tensor(gLSE.iterator, gLSE_expanded_layout)
            thr_mma = tiled_mma.get_slice(tidx)
            taccOgLSE = layout_utils.reshape_acc_to_mn(
                thr_mma.partition_C(gLSE_expanded)
            )
            assert cute.size(taccOgLSE, mode=[0]) == cute.size(lse)
            taccOcO = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cO))
            t0accOcO = layout_utils.reshape_acc_to_mn(
                thr_mma.get_slice(0).partition_C(cO)
            )

            # Only the thread corresponding to column 0 writes out the lse to gmem
            if taccOcO[0][1] == 0:
                for m in cutlass.range(cute.size(taccOgLSE.shape[1]), unroll_full=True):
                    if (
                        t0accOcO[m, 0][0]
                        < seqlen.seqlen_q - m_block * self.tile_m - taccOcO[0][0]
                    ):
                        taccOgLSE[m, 0] = lse[m]
        else:
            pack_gqa.store_LSE(mLSE_cur, lse, tiled_mma, tidx, m_block, seqlen.seqlen_q)
