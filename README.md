# MARBLE: usp-attention

Code, layouts and benchmark results for *MARBLE: Exact, Load-Balanced Sequence
Parallelism for Packed Multimodal Attention*.

Context parallelism for long, packed, irregularly masked attention on
FlashAttention-4. A cost model reads each layer's packed mask and picks an exact
strategy (`usp_attention.strategy`, `usp_attention.auto.AutoStrategy`): a sparse
all-to-all that fetches only the key/value blocks each rank's queries can see
(with a window-only path for sliding-window layers), a chunked ring for long
dense spans, or a balanced block-sparse ring. All of them combine Ulysses (head)
and ring (sequence) parallelism ("USP") with three pieces that matter once the
mask is sparse:

- **Block-sparse planning.** The packed mask is described per token
  (document, segment and attention mode) and classified per 128x128 block,
  never as a dense matrix.
- **PTRR load balancing.** Blocks are reordered before sharding so that every
  ring rank does the same amount of unmasked work, not merely the same number
  of tokens.
- **Online ring combine.** A start/stoppable FA4 forward kernel carries the
  fp32 softmax state (output accumulator, row max, row sum) across ring steps
  and normalizes once, instead of merging a bf16 partial after every step.

KV rotation between ring ranks can use async NCCL P2P, torch symmetric memory
(within one NVLink domain), or one-sided NCCL RMA via NCCL4Py (across nodes).

## Status

- The online ring combine kernel targets Hopper (SM90). By default
  (`USPConfig.combine_mode=None`) rings use it on H100 at every head_dim FA4
  supports there (up to 256), `"fp32"` on Blackwell and `"flash"` elsewhere. An AMD CDNA3 (MI300X/MI325X) online
  forward kernel lives in `usp_attention.online_flash_amd` (build it with
  `python src/usp_attention/online_flash_amd/setup.py build_ext --inplace`);
  the NVIDIA ring dispatcher does not call it (rings on ROCm default to
  `"flash"`); `usp_attention.online_flash_amd.distributed` is a forward-only
  packed sparse USP entry point built on it (see its README).
- The packed sequence length must be a multiple of
  `ring_size * ulysses_size * 128`; use `pad_packed_layout()`, which appends
  the padding as its own causal document (isolated from the real tokens).
- Self-attention only (Q and KV share one packed layout).
- KV transports: `nccl` (default) and `nccl4py` are supported and measured
  equally fast. NCCL4Py needs `NCCL_CUMEM_ENABLE=1`; with cuMem disabled NCCL
  cannot register its windows. The `symm_mem*` transports are experimental:
  all three deadlock reproducibly in the backward at ring 2 x Ulysses 4,
  128k tokens on 8x H100.
- Ring gradients are accumulated in fp32 across ring steps by default
  (`USPConfig.fp32_grads`); this matches single-GPU FA4 gradient accuracy for
  dq, and for dk / dv with grouped-query attention.
- `USPConfig(kv_exchange="gather")` replaces the ring's KV rotation with one
  mask-aware gather: each ring rank receives, in one all-to-all, only the KV
  blocks its queries can see, and attends over them with a single
  block-sparse FA4 call (`ring/halo.py`), or, when the dense ring is enabled
  (`dense_ring`: default on Blackwell, `dense_ring=True` on H100), the
  layout allows it and its full / noise spans are long enough
  (`dense_min_span`), with dense varlen calls over contiguous chunks
  (`ring/dense_gather.py`). dK / dV return the same way and are added in
  fp32. On H100,
  `fa3_gather=True` runs packed causal documents with FlashAttention-3 (about
  20% faster at head_dim 256, but 1.4x the single-GPU dK / dV error). With
  `balance="chunks"` (contiguous chunks per rank) the gathered KV is a small
  fraction of the sequence on packed layouts; gradients match single-GPU FA4
  accuracy. Sliding-window layers always gather their halo when it is small.
- A layout that is one plain causal or full document runs with FA4's native
  masks instead of block sparsity (`USPConfig.native_plain_masks`, on by
  default); same result, up to 7% faster. With a ring, a causal document is
  striped over the ring ranks (token `i` on ring rank `i % ring_size`) so
  every ring step is one causal or strictly causal FA4 call; this replaces
  PTRR for such layouts.
- Packed and mixed layouts. Without a ring, packed causal documents run as
  one varlen causal call, and multimodal layouts whose noise spans end their
  documents as two dense varlen calls merged by log-sum-exp
  (`interval_masks`, default on Blackwell); other layouts, and these on H100
  by default, run one block-sparse FA4 call. With a ring, multimodal layouts
  can run dense varlen calls over contiguous chunks (`dense_ring`, default on
  Blackwell): the dense ring when their noise spans end their documents and
  the chunks balance, or the exact generation ring when the full / noise
  spans are long (`dense_min_span`) and noise spans sit inside documents or
  `exact_dense_ring` is set. Other layouts, packed causal documents, and every
  packed layout on H100 by default use PTRR + block sparsity. Vision-group layouts
  are first reordered so their spans fill whole tiles
  (`reorder_vision_groups`) and always run block-sparse.
- On H100, FA4 has no 128-token block sparsity above head_dim 128 when
  training (its key tiles there are narrower than 128), so `usp_attention`
  raises a `ValueError` for block-sparse plans at those head dims. Build the
  plan for dense varlen calls where the layout allows: `dense_ring=True` with
  a ring (with `kv_exchange="gather"` also `dense_min_span=0` when the full /
  noise spans are shorter than 512 tokens), `interval_masks=True` without one.
- On H100, a native causal document without a ring runs with FlashAttention-3
  when `flash-attn-3` is installed (`USPConfig.fa3_on_hopper`, on by default);
  FA3's causal kernel is 1-3% faster than FA4's there (FA4 stays faster on
  full masks). Install it with
  `uv pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu129`.
- Blackwell (B200 / B300): block sparsity uses 256-row query blocks (FA4's
  SM100 tile), and the block-sparse forward launches one CTA per tile instead
  of FA4's static persistent scheduler, which balances packed masks better
  (11-12% faster forward). The online combine kernel is SM90-only; rings
  default to `combine_mode="fp32"` here, which keeps single-GPU ring accuracy
  (FA4 split-KV partials kept in fp32); `"flash"` / `"python"` also work. FA4's Blackwell backward needs
  `nvidia-cutlass-dsl` >= 4.7 (4.6.x deadlocks).
- Pinned to flash-attn-4 4.0.0b34; also tested with 4.0.0b21.

## Install

```bash
uv sync                      # torch 2.11 (cu129), flash-attn-4 4.0.0b34, CuTe DSL 4.7
uv sync --extra nccl4py      # optional: one-sided RMA ring transport
```

## Usage

```python
import torch
from usp_attention import USPConfig, USPGroups, build_usp_plan, init_distributed, usp_attention

device = init_distributed()                          # torchrun env
groups = USPGroups.create(ring_size=2, ulysses_size=4)

# Two packed samples: [caption (causal) | image (full)] and [prompt (causal) | image (noise)].
plan = build_usp_plan(
    document_lens=[65536, 65536],
    split_lens=[1024, 64512, 2048, 63488],
    attn_modes=["causal", "full", "causal", "noise"],
    groups=groups,
    config=USPConfig(load_balance=True, transport="nccl"),  # combine: online on H100
    device=device,
)

# Shard full-sequence activations once at the model input (applies the PTRR
# permutation), keep them sharded through the network, gather at the end.
x_local = plan.shard(x)                              # (B, S / 8, ...)
q, k, v = ...                                        # (B, S / 8, heads, head_dim)
out = usp_attention(q, k, v, plan)                   # (B, S / 8, heads, head_dim)
y = plan.gather(out_after_model)                     # original token order
```

The plan depends only on the layout, so build it once per batch and reuse it
for every layer.

## Layout

| Module | Contents |
|---|---|
| `usp.py` | `USPConfig`, `build_usp_plan`, `usp_attention` |
| `distributed.py` | Ring / Ulysses / CP process groups, differentiable all-to-all |
| `ring/` | Packed layout, PTRR planner, block-sparse classifiers, ring FA4 forward/backward, combine paths, NCCL / symmetric-memory / NCCL4Py transports |
| `online_flash/` | Online FA4 forward kernel (SM90) |
| `online_flash_amd/` | Online forward kernel for AMD CDNA3 (HIP) |
| `fa4/` | Single-GPU FA4 with the packed mask, block classification |
| `reference.py` | Chunked dense fp32/fp64 reference attention for accuracy checks |
| `layouts.py` | Synthetic packed layouts used by tests and benchmarks |

## Tests

```bash
uv run pytest                                        # CPU tests; GPU tests skip without CUDA
uv run pytest -m gpu                                 # single-GPU kernel tests
torchrun --nproc-per-node 8 tests/distributed/check_usp.py --ring 2 --ulysses 4 --layout mixed
```

`check_usp.py` compares output and Q/K/V gradients for every combine mode
against one FA4 call on the full sequence and against the dense fp32
reference, and prints one `RESULT {json}` line per comparison.

`check_model_parity.py` does the same at model level: a few layers of the
7B multimodal MoT decoder from `benchmarks/bench_model.py`, comparing the output, the
input gradient and every parameter gradient of USP against single-GPU FA4,
a flex_attention BlockMask of the same mask, and an fp32 model:

```bash
torchrun --nproc-per-node 8 tests/distributed/check_model_parity.py --ring 2 --ulysses 4 --layout mixed
```

`check_nccl4py_race.py` stress-tests the NCCL4Py one-sided transports with a
slow consumer at every ring step and fails if any step reads another owner's
or call's K/V (forward and backward KV rings) or a misrouted dK/dV sum (grad
ring). It needs NCCL >= 2.29.7 and `NCCL_CUMEM_ENABLE=1`:

```bash
NCCL_CUMEM_ENABLE=1 torchrun --nproc-per-node 8 tests/distributed/check_nccl4py_race.py --calls 100
```

Every baseline in `benchmarks/baselines/` has a `--check` mode that shards
with the library's own layout, runs forward and backward, and compares
against an fp32 dense reference; baselines are benchmarked only after their
check passes.

## Citation

If you use this code, please cite:

```bibtex
@article{redden2026marble,
  title   = {{MARBLE}: Exact, Load-Balanced Sequence Parallelism for Packed Multimodal Attention},
  author  = {Redden, Alex and Tripathi, Akshat and Meng, Delong and Pan, Sam and Tu, Zhucheng and Sundaram, Narayanan and Neff, Thomas},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

MIT (`LICENSE`), except for code adapted from other projects, which keeps its
license: FlashAttention-4, PyTorch and the FlashMask baseline adapters (BSD
3-Clause), MoonMath amd-kernels and AMD's OPUS header (MIT), BAGEL's packed-mask
encoding (Apache 2.0), and yunchang (Apache 2.0, benchmark baseline only).
`THIRD_PARTY_NOTICES.md` lists every adapted file. Every file
carries SPDX license and copyright tags, and the tree passes `reuse lint`.
Installing the package pulls in NVIDIA's CUDA libraries and CuTe DSL, which come
under NVIDIA's own license terms; they are dependencies and are not redistributed
here.

## Acknowledgements

The online kernel extends the Hopper forward kernel of
[FlashAttention-4](https://github.com/Dao-AILab/flash-attention)
(BSD-3-Clause) and uses [QuACK](https://github.com/Dao-AILab/quack).

## Reproduction

See [the reproduction guide](repro/README.md) for which result rows back the
paper, one command per paper table, pinned baseline setup recipes and the
numerical correctness gates. New benchmark rows record their source revision and
environment; rows measured before stamping existed are left unstamped rather
than relabeled with a current environment. The four `USP_DENSE_RING_*` environment switches have moved
to plan-scoped `USPConfig` fields; see the migration table in that guide.
