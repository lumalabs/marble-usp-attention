# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-FileCopyrightText: 2025 Bytedance Ltd. and/or its affiliates (packed-mask encoding adapted from BAGEL create_sparse_mask)
# SPDX-License-Identifier: MIT AND Apache-2.0
"""Model-level benchmark: a 7B multimodal MoT decoder stack under context parallelism.

The decoder is a mixture-of-transformers (MoT) language model with two
Qwen2.5-7B-sized experts, trained on packed multimodal sequences: every token
is routed to the understanding or the generation expert, each with its own
RMSNorms, QKV/O projections, QK-norm and SwiGLU MLP, and all tokens share one
packed attention under the mixed causal / full / noise mask. Weights are random; this measures
speed and memory, not quality.

``--model qwen3_8b`` instead runs a plain dense decoder with Qwen3-8B's
dimensions (one expert, 36 layers), the language model of Molmo2-8B, on a
vision-language layout such as ``--layout vlm_group``.

Parameters are sharded with FSDP2 over all ranks; each layer is activation-
checkpointed. Attention runs with one of:

- ``usp``: :func:`usp_attention` with the sequence sharded over all ranks
  (``--ring x --ulysses == world``);
- ``flex``: ``flex_attention`` with a packed BlockMask built from the same mask on
  the full sequence, one sequence per GPU (no context parallelism);
- ``fa4``: FA4 with the same mask on the full sequence, one sequence per GPU.

Run with torchrun, e.g.::

    torchrun --nproc-per-node 8 benchmarks/bench_model.py --attn usp \
        --ring 2 --ulysses 4 --seq-len 262144
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn.functional as F
from provenance import result_json
from torch import Tensor, nn
from torch.distributed.fsdp import fully_shard
from torch.utils.checkpoint import checkpoint

from usp_attention import (
    USPConfig,
    USPGroups,
    build_sliding_window_plan,
    build_usp_plan,
    init_distributed,
    usp_attention,
)
from usp_attention.fa4.flash4_attn import build_fa4_sparse_mask, flash4_attn
from usp_attention.layouts import LAYOUT_NAMES, build_layout
from usp_attention.reference import visible_key_counts
from usp_attention.ring.packed_layout import build_packed_layout

if TYPE_CHECKING:
    from collections.abc import Callable

    from jaxtyping import Float, Int

    AttentionFn = Callable[
        [
            Float[Tensor, " batch local_tokens num_heads head_dim"],
            Float[Tensor, " batch local_tokens kv_heads head_dim"],
            Float[Tensor, " batch local_tokens kv_heads head_dim"],
        ],
        Float[Tensor, " batch local_tokens num_heads head_dim"],
    ]

H100_BF16_PEAK_TFLOPS = 989.0


@dataclass(kw_only=True, slots=True)
class MoTConfig:
    """7B multimodal MoT decoder dimensions (Qwen2.5-7B per expert)."""

    hidden: int = 3584
    intermediate: int = 18944
    layers: int = 28
    heads: int = 28
    kv_heads: int = 4
    head_dim: int = 128
    rope_theta: float = 1_000_000.0
    eps: float = 1e-6
    # 2: understanding + generation experts (MoT); 1: a plain dense decoder.
    experts: int = 2
    qkv_bias: bool = True
    # Sliding-window layers: every `global_every`-th layer is global, the rest
    # attend to the last `sliding_window` keys (Gemma 3: 1024, 6).
    sliding_window: int | None = None
    global_every: int = 1

    def is_global(self, layer: int) -> bool:
        """Check whether a decoder layer uses global attention.

        Args:
            layer: Zero-based decoder-layer index.

        Returns:
            True when sliding windows are disabled or this is a global layer.
        """
        return self.sliding_window is None or (layer + 1) % self.global_every == 0


# Decoder presets: the 7B multimodal MoT decoder, Molmo2-8B's language model
# (Qwen3-8B dimensions, one expert, no QKV bias; also Qwen3-VL-8B's), and
# Gemma 3 27B's language model with its sliding-window layers.
PRESETS = {
    "mot7b": MoTConfig(),
    "qwen3_8b": MoTConfig(
        hidden=4096,
        intermediate=12288,
        layers=36,
        heads=32,
        kv_heads=8,
        experts=1,
        qkv_bias=False,
    ),
    # Gemma 3 27B's language model: 5 sliding-window layers per global layer.
    "gemma3_27b": MoTConfig(
        hidden=5376,
        intermediate=21504,
        layers=62,
        heads=32,
        kv_heads=16,
        experts=1,
        qkv_bias=False,
        sliding_window=1024,
        global_every=6,
    ),
}


class _Expert(nn.Module):
    """One expert's parameters for a decoder layer."""

    def __init__(self, cfg: MoTConfig) -> None:
        super().__init__()
        q_dim, kv_dim = cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim
        self.ln1 = nn.Parameter(torch.ones(cfg.hidden))
        self.ln2 = nn.Parameter(torch.ones(cfg.hidden))
        self.q = nn.Linear(cfg.hidden, q_dim, bias=cfg.qkv_bias)
        self.k = nn.Linear(cfg.hidden, kv_dim, bias=cfg.qkv_bias)
        self.v = nn.Linear(cfg.hidden, kv_dim, bias=cfg.qkv_bias)
        self.o = nn.Linear(q_dim, cfg.hidden, bias=False)
        self.q_norm = nn.Parameter(torch.ones(cfg.head_dim))
        self.k_norm = nn.Parameter(torch.ones(cfg.head_dim))
        self.gate = nn.Linear(cfg.hidden, cfg.intermediate, bias=False)
        self.up = nn.Linear(cfg.hidden, cfg.intermediate, bias=False)
        self.down = nn.Linear(cfg.intermediate, cfg.hidden, bias=False)


def _rotate_half(
    x: Float[Tensor, " *leading_dims head_dim"],
) -> Float[Tensor, " *leading_dims head_dim"]:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class MoTLayer(nn.Module):
    """MoT decoder layer training forward with a pluggable attention."""

    def __init__(self, cfg: MoTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.und = _Expert(cfg)
        self.gen = _Expert(cfg) if cfg.experts == 2 else None

    def _route(
        self,
        x: Float[Tensor, " local_tokens input_dim"],
        und: Int[Tensor, " understanding_tokens"],
        gen: Int[Tensor, " generation_tokens"],
        fn_und,
        fn_gen,
        out_dim: int,
    ) -> Float[Tensor, " local_tokens output_dim"]:
        if self.gen is None:
            return fn_und(x)
        out = x.new_empty(x.shape[0], out_dim)
        out.index_copy_(0, und, fn_und(x.index_select(0, und)))
        out.index_copy_(0, gen, fn_gen(x.index_select(0, gen)))
        return out

    def forward(
        self,
        x: Float[Tensor, " local_tokens hidden"],
        und: Int[Tensor, " understanding_tokens"],
        gen: Int[Tensor, " generation_tokens"],
        cos: Float[Tensor, " local_tokens 1 head_dim"],
        sin: Float[Tensor, " local_tokens 1 head_dim"],
        attn: AttentionFn,
    ) -> Float[Tensor, " local_tokens hidden"]:
        """Apply routed decoder attention and feed-forward computation.

        Args:
            x: Packed hidden states shaped (local_tokens, hidden).
            und: Local token indices routed to the understanding expert.
            gen: Local token indices routed to the generation expert.
            cos: Rotary-position cosine tensor for the local token positions.
            sin: Rotary-position sine tensor matching cos.
            attn: Callable accepting q, k and v and returning attention output.

        Returns:
            Updated hidden states with the same shape and token order as x.
        """
        cfg = self.cfg
        eps = cfg.eps
        hd = cfg.head_dim
        h = self._route(
            x,
            und,
            gen,
            lambda t: F.rms_norm(t, (cfg.hidden,), self.und.ln1, eps),
            lambda t: F.rms_norm(t, (cfg.hidden,), self.gen.ln1, eps),  # ty: ignore[possibly-missing-attribute]
            cfg.hidden,
        )

        def qkv(e: _Expert):
            def fn(t: Tensor) -> Tensor:
                q = F.rms_norm(
                    e.q(t).unflatten(-1, (cfg.heads, hd)), (hd,), e.q_norm, eps
                )
                k = F.rms_norm(
                    e.k(t).unflatten(-1, (cfg.kv_heads, hd)), (hd,), e.k_norm, eps
                )
                v = e.v(t).unflatten(-1, (cfg.kv_heads, hd))
                return torch.cat([q.flatten(1), k.flatten(1), v.flatten(1)], dim=-1)

            return fn

        q_dim, kv_dim = cfg.heads * hd, cfg.kv_heads * hd
        packed = self._route(
            h, und, gen, qkv(self.und), qkv(self.gen), q_dim + 2 * kv_dim
        )
        q, k, v = packed.split([q_dim, kv_dim, kv_dim], dim=-1)
        q = q.unflatten(-1, (cfg.heads, hd))
        k = k.unflatten(-1, (cfg.kv_heads, hd))
        v = v.unflatten(-1, (cfg.kv_heads, hd))
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        a = attn(q[None], k[None], v[None])[0].flatten(1)
        x = x + self._route(
            a,
            und,
            gen,
            self.und.o,
            lambda t: self.gen.o(t),
            cfg.hidden,  # ty: ignore[possibly-missing-attribute]
        )

        def mlp(e: _Expert, ln: str):
            def fn(t: Tensor) -> Tensor:
                t = F.rms_norm(t, (cfg.hidden,), getattr(e, ln), eps)
                return e.down(F.silu(e.gate(t)) * e.up(t))

            return fn

        return x + self._route(
            x, und, gen, mlp(self.und, "ln2"), mlp(self.gen, "ln2"), cfg.hidden
        )


class MoTDecoder(nn.Module):
    def __init__(self, cfg: MoTConfig, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(MoTLayer(cfg) for _ in range(layers))

    def forward(
        self,
        x: Float[Tensor, " local_tokens hidden"],
        und: Int[Tensor, " understanding_tokens"],
        gen: Int[Tensor, " generation_tokens"],
        cos: Float[Tensor, " local_tokens 1 head_dim"],
        sin: Float[Tensor, " local_tokens 1 head_dim"],
        attn: AttentionFn | list[AttentionFn],
    ) -> Float[Tensor, " local_tokens hidden"]:
        """Apply routed decoder attention and feed-forward computation.

        Args:
            x: Packed hidden states shaped (local_tokens, hidden).
            und: Local token indices routed to the understanding expert.
            gen: Local token indices routed to the generation expert.
            cos: Rotary-position cosine tensor for the local token positions.
            sin: Rotary-position sine tensor matching cos.
            attn: An attention callable, or a list of one callable per layer.

        Returns:
            Updated hidden states with the same shape and token order as x.
        """
        for i, layer in enumerate(self.layers):
            fn = attn[i] if isinstance(attn, list) else attn
            x = checkpoint(layer, x, und, gen, cos, sin, fn, use_reentrant=False)
        return x


def _rope(
    positions: Int[Tensor, " local_tokens"], cfg: MoTConfig
) -> tuple[
    Float[Tensor, " local_tokens 1 head_dim"], Float[Tensor, " local_tokens 1 head_dim"]
]:
    inv = 1.0 / (
        cfg.rope_theta
        ** (
            torch.arange(0, cfg.head_dim, 2, device=positions.device).float()
            / cfg.head_dim
        )
    )
    freqs = positions.float()[:, None] * inv[None]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[:, None].bfloat16(), emb.sin()[:, None].bfloat16()


def _flex_attention_fn(docs, splits, modes, device, sliding_window=None):
    from torch.nn.attention.flex_attention import (
        and_masks,
        create_block_mask,
        flex_attention,
        or_masks,
    )

    layout = build_packed_layout(docs, splits, modes, device=device)
    fn, noise, doc = (
        layout.full_and_noise_seq_id,
        layout.noise_seq_id,
        layout.document_id,
    )

    # The packed mixed-mode predicate as a flex_attention mask_mod.
    def causal(b, h, q, kv):
        if sliding_window:
            return (q >= kv) & (q - kv < sliding_window)
        return q >= kv

    def full_and_noise(b, h, q, kv):
        return (fn[q] == fn[kv]) & (fn[q] >= 0)

    def remove_noise(b, h, q, kv):
        return ~((noise[kv] >= 0) & (noise[q] != noise[kv]))

    def sample(b, h, q, kv):
        return doc[q] == doc[kv]

    mask = and_masks(or_masks(causal, full_and_noise), remove_noise, sample)
    block_mask = create_block_mask(
        mask, None, None, layout.seq_len, layout.seq_len, device=device, _compile=True
    )
    compiled = torch.compile(flex_attention)

    def attn(q, k, v):
        out = compiled(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            block_mask=block_mask,
            enable_gqa=True,
        )
        return out.transpose(1, 2)

    return attn


def _auto_planner(args, cfg, docs, splits, modes, world, rank, device):
    """``next()`` for ``--attn auto``: (batch, attention setup, record callback).

    :class:`PlanPrefetcher` worker processes run :meth:`AutoStrategy.plan` on CPU
    ahead of the steps; the step only attaches the plans.
    """
    from usp_attention.auto import AutoStrategy, PlanPrefetcher, auto_attention

    n_global = sum(cfg.is_global(i) for i in range(args.layers))
    auto = AutoStrategy(
        world=world,
        heads=cfg.heads,
        kv_heads=cfg.kv_heads,
        head_dim=cfg.head_dim,
        config=USPConfig(
            combine_mode=args.combine,  # type: ignore[arg-type]
            transport=args.transport,  # type: ignore[arg-type]
        ),
        layers=(n_global, args.layers - n_global),
        horizon=4,
    )
    auto.create_groups()
    it = PlanPrefetcher(
        auto,
        [(i, docs, splits, modes) for i in range(args.auto_steps + 1)],
        cp_rank=rank,
        sliding_window=cfg.sliding_window or None,
    )
    positions = torch.arange(args.seq_len, device=device)
    # Generated-image tokens ("noise" splits) use the generation expert.
    is_gen = torch.cat(
        [
            torch.full((n,), m == "noise", dtype=torch.bool)
            for n, m in zip(splits, modes)
        ]
    ).to(device)

    def next_batch():
        batch = next(it)
        plan, sliding = auto.attach(batch, device)
        strategy = batch.strategy

        def attn_with(p):
            return lambda q, k, v: auto_attention(q, k, v, p, strategy)

        a = attn_with(plan)
        if sliding is not None:
            local = attn_with(sliding)
            a = [a if cfg.is_global(i) else local for i in range(args.layers)]
        setup = (
            a,
            plan.shard(is_gen[None], seq_dim=1)[0],
            plan.shard(positions[None], seq_dim=1)[0],
            plan.local_len,
        )
        return batch, setup, lambda ms: auto.record(batch, ms)

    return next_batch


def main() -> None:
    """Benchmark model forward/backward from sys.argv and emit result rows.

    Raises:
        SystemExit: Argument parsing requests help or rejects invalid CLI
            options.
    """
    sys.excepthook = sys.__excepthook__  # CuTe DSL replaces it on import
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--attn",
        choices=["usp", "auto", "flex", "fa4", "flashmask"],
        default="usp",
        help="flashmask: FlashMask's balanced all-gather CP across all ranks, or with "
        "--ulysses U the FM + Ulysses hybrid (baselines/flashmask_cp.py; run in "
        "FlashMask's environment). auto: usp_attention.auto picks the strategy "
        "(planned in data-loader worker processes; --ring/--ulysses ignored)",
    )
    p.add_argument(
        "--auto-steps",
        type=int,
        default=24,
        help="--attn auto: steps run in total (trials, lock-in, then --iters timed)",
    )
    p.add_argument("--ring", type=int, default=1)
    p.add_argument("--ulysses", type=int, default=1)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument(
        "--layout",
        default="mixed",
        help=f"one of {LAYOUT_NAMES} or file:<path>[#index] (a packed-layout JSON)",
    )
    p.add_argument("--model", choices=sorted(PRESETS), default="mot7b")
    p.add_argument("--layers", type=int, default=None, help="default: the preset's")
    p.add_argument(
        "--combine",
        default=None,
        choices=["online", "flash", "python", "fp32"],
        help="ring combine mode (default: online on H100, fp32 on Blackwell)",
    )
    p.add_argument(
        "--sliding-window",
        type=int,
        default=None,
        help="Override the preset's sliding window (0 disables it: all layers global).",
    )
    p.add_argument(
        "--replicate-kv",
        type=int,
        default=1,
        help="Repeat each KV head this many times before attention (Ulysses past the KV heads).",
    )
    p.add_argument(
        "--global-ulysses",
        type=int,
        default=0,
        help="Run global layers on (world / N) x N Ulysses x ring instead of --ring x "
        "--ulysses, sharded like the sliding layers' plan.",
    )
    p.add_argument("--no-load-balance", action="store_true")
    p.add_argument(
        "--no-group-reorder",
        action="store_true",
        help="Keep vision-group documents in natural token order.",
    )
    p.add_argument(
        "--balance",
        choices=["ptrr", "chunks"],
        default="ptrr",
        help="Ring load balancing: PTRR 128-token blocks or contiguous chunks.",
    )
    p.add_argument(
        "--kv-exchange",
        choices=["ring", "gather"],
        default="ring",
        help="Ring groups rotate KV (ring) or gather only the visible KV blocks once (gather).",
    )
    p.add_argument(
        "--fa3-gather",
        action="store_true",
        help="FlashAttention-3 for single-call dense gathers (H100; dK / dV round per rank).",
    )
    p.add_argument(
        "--gather-regather",
        action="store_true",
        help="Gather: keep only local KV for the backward and exchange the halo again.",
    )
    p.add_argument(
        "--exact-dense-ring",
        action="store_true",
        help="Run interval layouts on the exact generation ring too.",
    )
    p.add_argument("--gather-max-keys", type=int, default=1 << 21)
    p.add_argument(
        "--transport",
        default="nccl",
        choices=["nccl", "symm_mem", "symm_mem_rotator", "symm_mem_direct", "nccl4py"],
    )
    p.add_argument(
        "--dense-ring",
        choices=["auto", "on", "off"],
        default="auto",
        help="Dense varlen kernels for ring / gather plans (auto: Blackwell).",
    )
    p.add_argument(
        "--replan",
        action="store_true",
        help="Rebuild the plan (FlashMask: balance + masks) inside every timed step, "
        "as training does for a new packed batch.",
    )
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--label", default="")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = init_distributed()
    rank, world = dist.get_rank(), dist.get_world_size()
    cfg = PRESETS[args.model]
    if args.sliding_window is not None:
        cfg = dataclasses.replace(cfg, sliding_window=args.sliding_window or None)
    args.layers = args.layers or cfg.layers
    docs, splits, modes = build_layout(
        args.layout, args.seq_len, ring_size=max(args.ring, 2), seed=args.seed
    )
    # Generated-image tokens ("noise" splits) use the generation expert.
    is_gen = torch.cat(
        [
            torch.full((n,), m == "noise", dtype=torch.bool)
            for n, m in zip(splits, modes)
        ]
    ).to(device)
    positions = torch.arange(args.seq_len, device=device)

    group_cache: dict = {}

    def cached_groups(ring: int, ulysses: int) -> USPGroups:
        """Process groups are created once; only planning repeats per step."""
        if (ring, ulysses) not in group_cache:
            group_cache[ring, ulysses] = USPGroups.create(
                ring_size=ring, ulysses_size=ulysses
            )
        return group_cache[ring, ulysses]

    def setup_attention():
        """Plan (or FlashMask balance + masks) for the layout; per step with --replan."""
        if args.attn == "usp":
            if args.ring * args.ulysses != world:
                raise ValueError(
                    "--ring * --ulysses must equal the world size for --attn usp"
                )
            groups = cached_groups(args.ring, args.ulysses)
            plan = build_usp_plan(
                document_lens=docs,
                split_lens=splits,
                attn_modes=modes,
                groups=groups,
                config=USPConfig(
                    combine_mode=args.combine,  # type: ignore[arg-type]
                    load_balance=not args.no_load_balance,
                    balance=args.balance,
                    kv_exchange=args.kv_exchange,
                    fa3_gather=args.fa3_gather,
                    gather_regather=args.gather_regather,
                    exact_dense_ring=args.exact_dense_ring,
                    dense_ring={"auto": None, "on": True, "off": False}[
                        args.dense_ring
                    ],
                    gather_max_keys=args.gather_max_keys,
                    reorder_vision_groups=not args.no_group_reorder,
                    transport=args.transport,  # type: ignore[arg-type]
                ),
                device=device,
            )
            local_gen = plan.shard(is_gen[None], seq_dim=1)[0]
            local_pos = plan.shard(positions[None], seq_dim=1)[0]

            def with_plan(p, replicate=1):
                if replicate > 1:
                    return lambda q, k, v: usp_attention(
                        q,
                        k.repeat_interleave(replicate, dim=2),
                        v.repeat_interleave(replicate, dim=2),
                        p,
                    )
                return lambda q, k, v: usp_attention(q, k, v, p)

            # With --global-ulysses, --replicate-kv applies to the global layers only.
            attn = with_plan(plan, 1 if args.global_ulysses else args.replicate_kv)
            global_attn = attn
            if args.global_ulysses:
                # Global layers on their own split, sharded exactly like `plan`.
                global_groups = cached_groups(
                    world // args.global_ulysses, args.global_ulysses
                )
                global_plan = build_usp_plan(
                    document_lens=docs,
                    split_lens=splits,
                    attn_modes=modes,
                    groups=global_groups,
                    config=plan.config,
                    device=device,
                    token_permutation=(
                        plan.token_permutation
                        if plan.token_permutation is not None
                        else torch.arange(args.seq_len, device=device)
                    ),
                )
                global_attn = with_plan(global_plan, args.replicate_kv)

            if cfg.sliding_window:
                # Sliding layers shard tokens like the global plan.
                local_plan = build_sliding_window_plan(
                    plan,
                    sliding_window=cfg.sliding_window,
                    document_lens=docs,
                    split_lens=splits,
                    attn_modes=modes,
                    device=device,
                )

                local_attn = with_plan(
                    local_plan, 1 if args.global_ulysses else args.replicate_kv
                )
                attn = [
                    global_attn if cfg.is_global(i) else local_attn
                    for i in range(args.layers)
                ]
            else:
                attn = global_attn

            tokens_per_rank = plan.local_len
        elif args.attn == "flashmask":
            sys.path.insert(0, str(Path(__file__).parent / "baselines"))
            from flashmask_cp import build_flashmask_cp

            windows = (None, cfg.sliding_window) if cfg.sliding_window else (None,)
            # --ulysses U > 1: the FM + Ulysses hybrid (Ulysses over U consecutive
            # ranks, FlashMask CP across the groups), as in the attention tables.
            cp_group = u_group = None
            if args.ulysses > 1:
                u_size = args.ulysses
                for r_ in range(world // u_size):
                    g = dist.new_group(list(range(r_ * u_size, (r_ + 1) * u_size)))
                    if rank // u_size == r_:
                        u_group = g
                for u_ in range(u_size):
                    g = dist.new_group(list(range(u_, world, u_size)))
                    if rank % u_size == u_:
                        cp_group = g
            fm = build_flashmask_cp(
                docs,
                splits,
                modes,
                windows=windows,
                device=device,
                group=cp_group,
                ulysses_group=u_group,
                replicate_kv=args.replicate_kv,
            )
            local_gen = is_gen[fm.local_tokens]
            local_pos = positions[fm.local_tokens]
            tokens_per_rank = int(fm.local_tokens.numel())
            attn = fm.attention(None)
            if cfg.sliding_window:
                local_attn = fm.attention(cfg.sliding_window)
                attn = [
                    attn if cfg.is_global(i) else local_attn for i in range(args.layers)
                ]
        else:
            local_gen, local_pos = is_gen, positions
            tokens_per_rank = args.seq_len

            def single_gpu(window):
                if args.attn == "flex":
                    return _flex_attention_fn(docs, splits, modes, device, window)
                mask = build_fa4_sparse_mask(
                    docs, splits, modes, device=device, sliding_window=window
                )
                return lambda q, k, v: flash4_attn(q, k, v, mask)

            attn = single_gpu(None)
            if cfg.sliding_window:
                local_attn = single_gpu(cfg.sliding_window)
                attn = [
                    attn if cfg.is_global(i) else local_attn for i in range(args.layers)
                ]
        return attn, local_gen, local_pos, tokens_per_rank

    if args.attn == "auto":
        auto_next = _auto_planner(args, cfg, docs, splits, modes, world, rank, device)
        _, first, _ = auto_next()
        # The packed layout fixes the local sequence length for every strategy.
        setup_attention = lambda: first  # noqa: E731

    t0 = time.perf_counter()
    attn, local_gen, local_pos, tokens_per_rank = setup_attention()
    torch.cuda.synchronize()
    setup_ms = (time.perf_counter() - t0) * 1e3

    und = (~local_gen).nonzero().squeeze(1)
    gen = local_gen.nonzero().squeeze(1)
    cos, sin = _rope(local_pos, cfg)

    with torch.device("meta"):
        model = MoTDecoder(cfg, args.layers).to(torch.bfloat16)
    for layer in model.layers:
        fully_shard(layer)
    fully_shard(model)
    model.to_empty(device=device)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(".bias"):
                param.zero_()
            elif param.ndim == 1:
                param.fill_(1.0)
            else:
                param.normal_(0.0, 0.02)

    gen_t = torch.Generator(device=device).manual_seed(args.seed + rank)
    x = torch.randn(
        tokens_per_rank,
        cfg.hidden,
        device=device,
        dtype=torch.bfloat16,
        generator=gen_t,
    )

    def step() -> None:
        if args.replan:
            # A new packed batch every step: plan inside the timed step.
            a, lg, lp, _ = setup_attention()
            u, g = (~lg).nonzero().squeeze(1), lg.nonzero().squeeze(1)
            c, s_ = _rope(lp, cfg)
            out = model(x, u, g, c, s_, a)
        else:
            out = model(x, und, gen, cos, sin, attn)
        out.float().pow(2).mean().backward()
        for param in model.parameters():
            param.grad = None

    if args.attn == "auto":
        times, strategies = [], []
        for i in range(args.auto_steps):
            batch, (a, lg, lp, _), record = auto_next()
            u, g = (~lg).nonzero().squeeze(1), lg.nonzero().squeeze(1)
            c, s_ = _rope(lp, cfg)
            dist.barrier()
            s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            s.record()
            model(x, u, g, c, s_, a).float().pow(2).mean().backward()
            for param in model.parameters():
                param.grad = None
            e.record()
            e.synchronize()
            t = torch.tensor([s.elapsed_time(e)], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            record(t.item())
            if i == args.auto_steps - args.iters:
                torch.cuda.reset_peak_memory_stats()
            times.append(t.item())
            strategies.append(batch.strategy.label)
        if rank == 0:
            print(
                "AUTO " + json.dumps({"strategies": strategies, "step_ms": times}),
                flush=True,
            )
        times = times[-args.iters :]
    for _ in range(0 if args.attn == "auto" else args.warmup):
        step()
    torch.cuda.synchronize()
    if args.attn != "auto":
        torch.cuda.reset_peak_memory_stats()
    times = times if args.attn == "auto" else []
    for _ in range(0 if args.attn == "auto" else args.iters):
        dist.barrier()
        s, e = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        s.record()
        step()
        e.record()
        e.synchronize()
        t = torch.tensor([s.elapsed_time(e)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        times.append(t.item())
    peak = torch.cuda.max_memory_allocated() / 2**30

    if rank == 0:
        median = statistics.median(times)
        seqs_per_step = 1 if args.attn in ("usp", "flashmask") else world
        tokens_per_step = args.seq_len * seqs_per_step
        # Active parameters per token: one expert's attention projections + MLP.
        per_layer = (
            cfg.hidden * (cfg.heads + 2 * cfg.kv_heads) * cfg.head_dim
            + cfg.heads * cfg.head_dim * cfg.hidden
            + 3 * cfg.hidden * cfg.intermediate
        )
        dense_flops = 6 * per_layer * args.layers * tokens_per_step
        pairs = int(visible_key_counts(docs, splits, modes).sum())
        n_global = sum(cfg.is_global(i) for i in range(args.layers))
        layer_pairs = pairs * n_global
        if cfg.sliding_window:
            local_pairs = visible_key_counts(
                docs, splits, modes, sliding_window=cfg.sliding_window
            ).sum()
            layer_pairs += int(local_pairs) * (args.layers - n_global)
        attn_flops = 3.5 * 4 * cfg.heads * cfg.head_dim * layer_pairs * seqs_per_step
        tflops = (dense_flops + attn_flops) / (median * 1e-3) / world / 1e12
        row = {
            "bench": "model",
            "model": args.model,
            "label": args.label,
            "attn": args.attn,
            "layout": args.layout,
            "sliding_window": cfg.sliding_window,
            "global_every": cfg.global_every if cfg.sliding_window else None,
            "replicate_kv": args.replicate_kv,
            "global_ulysses": args.global_ulysses or None,
            "balance": args.balance,
            "kv_exchange": args.kv_exchange,
            "fa3_gather": args.fa3_gather,
            "gather_regather": args.gather_regather,
            "exact_dense_ring": args.exact_dense_ring,
            "gather_max_keys": args.gather_max_keys,
            "replan": args.replan,
            "dense_ring": args.dense_ring,
            "seq_len": args.seq_len,
            "layers": args.layers,
            "world": world,
            "ring": args.ring,
            "ulysses": args.ulysses,
            "combine": USPConfig(combine_mode=args.combine).resolved_combine_mode(
                device, head_dim=cfg.head_dim
            ),  # type: ignore[arg-type]
            "load_balance": not args.no_load_balance,
            "transport": args.transport,
            "step_ms_median": median,
            "step_ms_all": times,
            "tokens_per_s_per_gpu": tokens_per_step / (median * 1e-3) / world,
            "model_tflops_per_gpu": tflops,
            "mfu": tflops / H100_BF16_PEAK_TFLOPS,
            "attn_flop_share": attn_flops / (dense_flops + attn_flops),
            "setup_ms": setup_ms,
            "peak_mem_gib": peak,
            "mask_density": pairs / args.seq_len**2,
        }
        print("RESULT " + result_json(row), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
