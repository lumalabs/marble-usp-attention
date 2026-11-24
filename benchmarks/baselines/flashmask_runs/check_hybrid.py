# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Check the FM + Ulysses hybrid of ``flashmask_cp.py`` against plain FlashMask CP.

Both run the same FlashMask kernels on the same global Q/K/V; the hybrid adds a
Ulysses all-to-all around them and splits each CP shard over a Ulysses group.
Each rank scatters its output rows (and dQ/dK/dV) back to original token order,
the ranks sum them, and rank 0 prints the largest differences, which should be
bf16 rounding (both run the same kernel, so this is not a reference check)::

    torchrun --nproc-per-node 8 benchmarks/baselines/flashmask_runs/check_hybrid.py \\
        --layout benchmarks/layouts/packs/bagel_video_32k.json#0 --seq-len 32768 --ulysses 4
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from flashmask_cp import build_flashmask_cp  # noqa: E402


def load_layout(spec: str):
    path, _, index = spec.partition("#")
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path += ".gz"
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        entry = json.load(f)["layouts"][int(index or 0)]
    return entry["document_lens"], entry["split_lens"], entry["attn_modes"]


def run(fm, q, k, v, dout):
    idx = fm.local_tokens
    ql, kl, vl = (t[:, idx].detach().clone().requires_grad_(True) for t in (q, k, v))
    out = fm.attention(None)(ql, kl, vl)
    out.backward(dout[:, idx])
    full = [torch.zeros_like(t, dtype=torch.float32) for t in (q, q, k, v)]
    for dst, src in zip(full, (out, ql.grad, kl.grad, vl.grad)):
        dst[:, idx] = src.float()
        dist.all_reduce(dst)
    return full


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--layout", required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--heads", type=int, default=28)
    p.add_argument("--kv-heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--ulysses", type=int, default=4)
    args = p.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    world, rank, u = dist.get_world_size(), dist.get_rank(), args.ulysses
    u_group = cp_group = None
    for r_ in range(world // u):
        g = dist.new_group(list(range(r_ * u, (r_ + 1) * u)))
        if rank // u == r_:
            u_group = g
    for u_ in range(u):
        g = dist.new_group(list(range(u_, world, u)))
        if rank % u == u_:
            cp_group = g

    docs, splits, modes = load_layout(args.layout)
    assert sum(docs) == args.seq_len
    gen = torch.Generator(device=device).manual_seed(0)
    shape = (1, args.seq_len)
    q = torch.randn(*shape, args.heads, args.head_dim, device=device, generator=gen)
    k = torch.randn(*shape, args.kv_heads, args.head_dim, device=device, generator=gen)
    v = torch.randn_like(k)
    dout = torch.randn_like(q)
    q, k, v, dout = (t.bfloat16() for t in (q, k, v, dout))
    for t in (q, k, v, dout):  # identical inputs on every rank
        dist.broadcast(t, src=0)

    plain = build_flashmask_cp(docs, splits, modes, device=device)
    hybrid = build_flashmask_cp(
        docs, splits, modes, device=device, group=cp_group, ulysses_group=u_group
    )
    a, b = run(plain, q, k, v, dout), run(hybrid, q, k, v, dout)
    if rank == 0:
        for name, x, y in zip(("out", "dq", "dk", "dv"), a, b):
            rel = (x - y).abs().mean() / x.abs().mean()
            print(
                f"{name}: max abs diff {(x - y).abs().max():.3e}, mean rel diff {rel:.3e}"
            )
        # Exact fp32 outputs for sampled query rows.
        from usp_attention.reference import dense_mask_rows
        from usp_attention.ring.packed_layout import build_packed_layout

        rows = torch.randperm(args.seq_len, generator=torch.Generator().manual_seed(1))[
            :256
        ]
        layout = build_packed_layout(docs, splits, modes, device=device)
        mask = dense_mask_rows(layout, 0, args.seq_len)[rows.to(device)]
        rep = args.heads // args.kv_heads
        qs = q[0, rows.to(device)].float()  # (n, H, D)
        kf = k[0].float().repeat_interleave(rep, dim=1)  # (S, H, D)
        vf = v[0].float().repeat_interleave(rep, dim=1)
        s = torch.einsum("nhd,shd->hns", qs, kf) / args.head_dim**0.5
        s = s.masked_fill(~mask[None], float("-inf"))
        ref = torch.einsum("hns,shd->nhd", s.softmax(-1), vf)
        for name, out in (("plain", a[0]), ("hybrid", b[0])):
            got = out[0, rows.to(device)]
            rel = (got - ref).abs().mean() / ref.abs().mean()
            print(f"{name} vs fp32 reference on 256 rows: mean rel err {rel:.3e}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
