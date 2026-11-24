# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Does ring-attention numerics matter for training? A small LM convergence A/B.

Trains the same decoder-only language model on the same packed web-text
stream with different context-parallel attention paths and logs the loss per
step, so curves can be compared against a reference:

- ``ulysses``: Ulysses only (one FA4 call over the full sequence per head
  subset); numerically the single-GPU result, used as the reference.
- ``ring``: ring attention with a chosen combine mode and gradient precision.

All runs share the initialization (fixed seed) and the data order. Sequences
pack whole documents; attention is causal within each document.

Prepare data once (downloads one FineWeb-Edu shard and tokenizes it with the
GPT-2 BPE)::

    python benchmarks/convergence.py --prepare --data-dir /path/to/data

then train, e.g.::

    torchrun --nproc-per-node 8 benchmarks/convergence.py --data-dir /path/to/data \\
        --mode ring --combine online --out results/conv_ring_online.jsonl
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from provenance import result_json
from torch import Tensor, nn

if TYPE_CHECKING:
    from collections.abc import Callable

    from jaxtyping import Float, Int, Integer

    AttentionFn = Callable[
        [
            Float[Tensor, " batch local_tokens num_heads head_dim"],
            Float[Tensor, " batch local_tokens kv_heads head_dim"],
            Float[Tensor, " batch local_tokens kv_heads head_dim"],
        ],
        Float[Tensor, " batch local_tokens num_heads head_dim"],
    ]

EOT = 50256  # GPT-2 end-of-text token


def prepare(data_dir: Path, max_tokens: int) -> None:
    """Download one FineWeb-Edu shard and write ``tokens.bin`` (uint16).

    Args:
        data_dir: Directory to create and populate with tokens.bin.
        max_tokens: Maximum number of token IDs to retain from the downloaded
            shard.
    """
    import pyarrow.parquet as pq
    import tiktoken
    from huggingface_hub import hf_hub_download

    data_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        "HuggingFaceFW/fineweb-edu",
        "sample/10BT/000_00000.parquet",
        repo_type="dataset",
        local_dir=data_dir,
    )
    enc = tiktoken.get_encoding("gpt2")
    out = np.empty(max_tokens, dtype=np.uint16)
    n = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=["text"]):
        docs = enc.encode_ordinary_batch(
            batch.column("text").to_pylist(), num_threads=32
        )
        for ids in docs:
            ids = [*ids, EOT]
            take = min(len(ids), max_tokens - n)
            out[n : n + take] = ids[:take]
            n += take
            if n >= max_tokens:
                break
        if n >= max_tokens:
            break
    out[:n].tofile(data_dir / "tokens.bin")
    print(f"wrote {n} tokens to {data_dir / 'tokens.bin'}")


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.norm1 = nn.RMSNorm(dim)
        self.qkv = nn.Linear(dim, (heads + 2 * kv_heads) * head_dim, bias=False)
        self.o = nn.Linear(heads * head_dim, dim, bias=False)
        self.norm2 = nn.RMSNorm(dim)
        hidden = int(dim * 8 / 3) // 64 * 64
        self.gate_up = nn.Linear(dim, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(
        self,
        x: Float[Tensor, " batch tokens hidden"],
        cos: Float[Tensor, " batch tokens 1 head_dim"],
        sin: Float[Tensor, " batch tokens 1 head_dim"],
        attn: AttentionFn,
    ) -> Float[Tensor, " batch tokens hidden"]:
        """Apply one decoder block using the supplied attention function.

        Args:
            x: Hidden-state tensor shaped (batch, tokens, hidden).
            cos: Rotary-position cosine tensor broadcastable over queries and
                keys.
            sin: Rotary-position sine tensor matching cos.
            attn: Callable accepting q, k and v and returning attention output.

        Returns:
            Updated hidden states with the same shape as x.
        """
        h = self.qkv(self.norm1(x))
        q, k, v = h.split(
            [
                self.heads * self.head_dim,
                self.kv_heads * self.head_dim,
                self.kv_heads * self.head_dim,
            ],
            -1,
        )
        q = q.unflatten(-1, (self.heads, self.head_dim))
        k = k.unflatten(-1, (self.kv_heads, self.head_dim))
        v = v.unflatten(-1, (self.kv_heads, self.head_dim))
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        a = attn(q.bfloat16(), k.bfloat16(), v.bfloat16()).flatten(-2)
        x = x + self.o(a.to(x.dtype))
        g, u = self.gate_up(self.norm2(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(g) * u)


def _rope(
    x: Float[Tensor, " batch tokens num_heads head_dim"],
    cos: Float[Tensor, " batch tokens 1 head_dim"],
    sin: Float[Tensor, " batch tokens 1 head_dim"],
) -> Float[Tensor, " batch tokens num_heads head_dim"]:
    x1, x2 = x.float().chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return (x.float() * cos + rotated * sin).to(x.dtype)


class LM(nn.Module):
    def __init__(
        self, vocab: int, dim: int, layers: int, heads: int, kv_heads: int
    ) -> None:
        super().__init__()
        self.head_dim = dim // heads
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList(
            Block(dim, heads, kv_heads, self.head_dim) for _ in range(layers)
        )
        self.norm = nn.RMSNorm(dim)

    def forward(
        self,
        ids: Int[Tensor, " batch tokens"],
        positions: Int[Tensor, " batch tokens"],
        attn: AttentionFn,
    ) -> Float[Tensor, " batch tokens vocab_size"]:
        """Compute next-token logits for a packed token sequence.

        Args:
            ids: Token IDs shaped (batch, tokens).
            positions: Per-token positions used to construct rotary embeddings.
            attn: Attention callable shared by all decoder blocks.

        Returns:
            Logits shaped (batch, tokens, vocab_size), using the tied
            embedding.
        """
        inv = 1.0 / (
            10000
            ** (
                torch.arange(0, self.head_dim, 2, device=ids.device).float()
                / self.head_dim
            )
        )
        freqs = positions.float()[:, :, None] * inv
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos()[:, :, None], emb.sin()[:, :, None]
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x, cos, sin, attn)
        return self.norm(x) @ self.embed.weight.T


def window(
    tokens: Integer[np.ndarray, " corpus_tokens"], step: int, seq_len: int
) -> tuple[
    Int[Tensor, " 1 tokens"],
    Int[Tensor, " 1 tokens"],
    Int[Tensor, " 1 tokens"],
    list[int],
]:
    """Build a packed next-token-prediction window for one training step.

    Args:
        tokens: One-dimensional NumPy token-ID array with an extra target
            token.
        step: Zero-based window index into tokens.
        seq_len: Number of input tokens in the window.

    Returns:
        A tuple (ids, labels, positions, document_lens). The first three values
        are tensors shaped (1, seq_len); boundary-crossing labels are ignored,
        and positions restart at each document boundary.
    """
    start = step * seq_len
    chunk = torch.from_numpy(tokens[start : start + seq_len + 1].astype(np.int64))
    ids, labels = chunk[:-1].clone(), chunk[1:].clone()
    # Documents end after each EOT; never predict across a document boundary.
    ends = (ids == EOT).nonzero().squeeze(1).tolist()
    bounds = [0, *[e + 1 for e in ends if e + 1 < seq_len], seq_len]
    doc_lens = [b - a for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    labels[ids == EOT] = -100
    positions = torch.cat([torch.arange(n) for n in doc_lens])
    return ids[None], labels[None], positions[None], doc_lens


def main() -> None:
    """Prepare tokens or train the comparison model using CLI options.

    Raises:
        SystemExit: Argument parsing requests help or rejects invalid CLI
            options.
    """
    sys.excepthook = sys.__excepthook__
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--max-tokens", type=int, default=400_000_000)
    p.add_argument("--mode", choices=["ulysses", "ring"], default="ring")
    p.add_argument(
        "--combine",
        choices=["online", "flash", "python", "fp32"],
        default=None,
        help="ring combine mode (default: online on H100, fp32 on Blackwell)",
    )
    p.add_argument("--bf16-grads", action="store_true")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seq-len", type=int, default=65536)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    data_dir = Path(args.data_dir)
    if args.prepare:
        prepare(data_dir, args.max_tokens)
        return

    from usp_attention import (
        USPConfig,
        USPGroups,
        build_usp_plan,
        init_distributed,
        usp_attention,
    )

    device = init_distributed()
    rank, world = dist.get_rank(), dist.get_world_size()
    ring, ulysses = (1, world) if args.mode == "ulysses" else (world, 1)
    groups = USPGroups.create(ring_size=ring, ulysses_size=ulysses)
    config = USPConfig(combine_mode=args.combine, fp32_grads=not args.bf16_grads)  # type: ignore[arg-type]

    tokens = np.memmap(data_dir / "tokens.bin", dtype=np.uint16, mode="r")
    torch.manual_seed(args.seed)
    model = LM(50304, args.dim, args.layers, args.heads, args.kv_heads).to(device)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
    nn.init.normal_(model.embed.weight, std=0.02)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        fused=True,
    )
    params = [p for p in model.parameters()]
    out = open(args.out, "a") if (args.out and rank == 0) else None

    for step in range(args.steps):
        t0 = time.perf_counter()
        lr = args.lr * min(1.0, (step + 1) / args.warmup)
        lr *= 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / args.steps)))
        for group in opt.param_groups:
            group["lr"] = lr
        ids, labels, positions, doc_lens = window(tokens, step, args.seq_len)
        plan = build_usp_plan(
            document_lens=doc_lens,
            split_lens=doc_lens,
            attn_modes=["causal"] * len(doc_lens),
            groups=groups,
            config=config,
            device=device,
        )
        ids, labels, positions = (
            plan.shard(t.to(device)) for t in (ids, labels, positions)
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids, positions, lambda q, k, v: usp_attention(q, k, v, plan))
        loss_sum = F.cross_entropy(
            logits.float().flatten(0, 1),
            labels.flatten(),
            ignore_index=-100,
            reduction="sum",
        )
        count = (labels != -100).sum().float()
        total = torch.stack([loss_sum.detach(), count])
        dist.all_reduce(total)
        (loss_sum / total[1]).backward()

        flat = torch.cat([p.grad.flatten() for p in params])
        dist.all_reduce(flat)
        offset = 0
        for p_ in params:
            n = p_.numel()
            p_.grad.copy_(flat[offset : offset + n].view_as(p_))
            offset += n
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if out is not None:
            row = {
                "step": step,
                "loss": (total[0] / total[1]).item(),
                "grad_norm": grad_norm.item(),
                "lr": lr,
                "step_s": time.perf_counter() - t0,
                "mode": args.mode,
                "combine": config.resolved_combine_mode(
                    device, head_dim=model.head_dim
                ),
                "fp32_grads": config.fp32_grads,
                "seed": args.seed,
            }
            out.write(result_json(row) + "\n")
            if step % 50 == 0:
                out.flush()
                print(
                    f"step {step} loss {row['loss']:.4f} gnorm {row['grad_norm']:.3f} {row['step_s']:.2f}s",
                    flush=True,
                )
    if out is not None:
        out.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
