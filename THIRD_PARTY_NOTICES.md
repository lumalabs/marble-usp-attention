# Third-party notices

usp-attention is MIT-licensed (`LICENSE`). It contains code adapted from the
projects below, which keeps its original license. Every source file states
its license and copyright holders in SPDX headers (or in `REUSE.toml`); the
license texts are in `LICENSES/`, and the upstream license and authors files
are kept verbatim in `third_party_licenses/`. `reuse lint` checks the tree.

## FlashAttention (FlashAttention-4, CuTe DSL)

- Source: https://github.com/Dao-AILab/flash-attention (`flash_attn/cute`)
- License: BSD 3-Clause (`third_party_licenses/flash-attention-LICENSE`; the
  authors file is upstream's `flash_attn/cute/AUTHORS`)
- Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
  Pradeep Ramani, Tri Dao (as in the adapted files' headers); Copyright (c)
  2025, Tri Dao (`flash_attn/cute/utils.py`).
- Adapted files:
  - `src/usp_attention/online_flash/online_flash_fwd_sm90.py`: the SM90
    forward kernel, extended to start and stop across ring steps (fp32 output,
    row max and row sum kept in HBM between calls).
  - `src/usp_attention/online_flash/block_sparse_utils.py`: the block-sparse
    load helpers, changed only to not reinitialize the softmax state on
    continuation calls.
  - `src/usp_attention/online_flash/interface.py`: SM90 tile sizing and
    block-sparse normalization mirror the upstream forward interface.
  - `src/usp_attention/fa4/cute_dsl_utils.py`: `warp_reduce`, from
    `flash_attn/cute/utils.py`.
  - `src/usp_attention/fa4/separable_mask_sm90.py`: the accumulator fragment
    coordinate setup, as in `apply_mask` of `flash_attn/cute/mask.py`.
- The package also calls into flash-attn-4 at runtime (an install
  dependency, not vendored).

## PyTorch

- Source: https://github.com/pytorch/pytorch
  (`torch/distributed/tensor/experimental/_context_parallel/_load_balancer.py`)
- License: BSD 3-Clause (`third_party_licenses/pytorch-LICENSE`)
- Copyright (c) 2016- Facebook, Inc and the PyTorch contributors (see the
  license file).
- Adapted code: `_ptrr_scheduling` in `src/usp_attention/ring/ring_planning.py`,
  the processing-time round-robin (PTRR) schedule of `_PTRRLoadBalancer`,
  applied to a plain per-block cost vector.

## YunChang (long-context-attention)

- Source: https://github.com/feifeibear/long-context-attention
  (`yunchang/kernels/attention.py`)
- License: Apache License 2.0 (`third_party_licenses/yunchang-LICENSE`)
- Copyright: Jiarui Fang and the YunChang contributors.
- Adapted code: the two FA3 wrappers in `_patch_fa3_causal`
  (`benchmarks/baselines/yunchang_usp.py`, benchmark-only), changed only to
  pass the `causal` flag through to the kernel.

## BAGEL

- Source: https://github.com/ByteDance-Seed/Bagel (`data/data_utils.py`)
- License: Apache License 2.0 (`third_party_licenses/bagel-LICENSE`)
- Copyright 2025 Bytedance Ltd. and/or its affiliates.
- Adapted code: the packed-sequence mask encoding of `create_sparse_mask`
  (per-token full/noise segment ids, noise ids and document ids, and the
  predicate `(causal | same segment) & remove_noise & same document`), in
  `_build_common_tensors` and the FlexAttention reference mask of
  `src/usp_attention/fa4/flash4_attn.py`, `build_packed_layout` in
  `src/usp_attention/ring/packed_layout.py`, `make_packed_layout_mask` in
  `src/usp_attention/ring/ring_planning.py`, and the FlexAttention masks of
  `benchmarks/baselines/torch_cp.py` and `benchmarks/bench_model.py`. The
  CuTe DSL mask_mods, block classifiers, position track, group mode and
  sliding window are new.

## MoonMath amd-kernels

- Source: https://github.com/moonmath-ai/amd-kernels (public)
- License: MIT
- Copyright (c) 2026 The MoonMath.ai
- Adapted files:
  - `src/usp_attention/online_flash_amd/csrc/attention_kernel.hip`: the CDNA3
    forward kernel, extended with the online (ring) path.
  - `src/usp_attention/online_flash_amd/csrc/attention_rt{ne,na,z}.hip`:
    upstream's `csrc/gfx942/attention_rt*.hip` round-mode shims, which compile
    the kernel with other bf16 rounding modes, adapted to rename the exported
    symbols.
  - `src/usp_attention/online_flash_amd/csrc/attention_api.cpp`: the pybind11
    bindings, extended with the online entry points.
  - `src/usp_attention/online_flash_amd/setup.py`: the extension build script.

## OPUS

- Source: AMD's AITER,
  https://github.com/ROCm/aiter/blob/a86fb0fab139ce78bce40e2ac51659a27ca161dd/csrc/include/opus/opus.hpp
  (unmodified; also shipped in MoonMath amd-kernels as `csrc/gfx942/opus.hpp`)
- License: MIT
- File: `src/usp_attention/online_flash_amd/csrc/opus.hpp` (copyright and
  license notice retained in the file header).

## FlashMask distributed baseline adapters

`benchmarks/baselines/flashmask_cp.py` and `flashmask_cp_bench.py` adapt the
balanced and overlapped autograd functions from FlashMask at
`6a150fcf771bdbbaded07a427b53513ad1f5fb7b`. Copyright 2025 Tri Dao and 2026 The
FlashMask Authors; BSD-3-Clause. The original license is retained verbatim in
[`third_party_licenses/flashmask-LICENSE`](third_party_licenses/flashmask-LICENSE).
