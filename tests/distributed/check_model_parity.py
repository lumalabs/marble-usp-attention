# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Model-level forward + backward parity for usp_attention in the 7B multimodal MoT decoder.

Runs a few full-size layers of the decoder from ``benchmarks/bench_model.py``
(same weights and inputs on every rank, no FSDP) and compares, against an fp32
model with exact masked attention on the full sequence:

- ``fa4``: the bf16 model with single-GPU FA4 on the full sequence, the result
  without any parallelism;
- ``flex``: the bf16 model with a ``flex_attention`` BlockMask of the same mask
  on the full sequence;
- ``usp_<combine>``: the bf16 model with every rank running its shard under
  :func:`usp_attention`.

Compared quantities: the decoder output, the gradient of the input hidden
states, and every parameter gradient (USP gradients are summed over ranks).
The loss is a fixed random projection of the output, so every output element
contributes. Each comparison prints a ``RESULT {json}`` line; USP should match
``fa4``'s errors, and ``usp_vs_fa4`` gives the direct bf16 difference::

    torchrun --nproc-per-node 8 tests/distributed/check_model_parity.py \
        --ring 2 --ulysses 4 --layout mixed --seq-len 8192
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from usp_attention import (
    USPConfig,
    USPGroups,
    build_usp_plan,
    init_distributed,
    usp_attention,
)
from usp_attention.fa4.flash4_attn import build_fa4_sparse_mask, flash4_attn
from usp_attention.layouts import LAYOUT_NAMES, build_layout
from usp_attention.reference import reference_attention

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
from bench_model import (  # noqa: E402
    MoTConfig,
    MoTDecoder,
    _flex_attention_fn,
    _rope,
)
from provenance import result_json  # noqa: E402


def _run(model, x, w, is_gen, positions, attn, cfg):
    """Forward + backward; returns (out, dx, {param name: grad})."""
    x = x.detach().clone().requires_grad_(True)
    und = (~is_gen).nonzero().squeeze(1)
    gen = is_gen.nonzero().squeeze(1)
    cos, sin = _rope(positions, cfg)
    if x.dtype == torch.float32:
        cos, sin = cos.float(), sin.float()
    out = model(x, und, gen, cos, sin, attn)
    (out.float() * w).sum().backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    return out.detach(), x.grad.detach(), grads


def _errors(ours, reference) -> dict[str, object]:
    """Mean relative errors of output, input grad and parameter grads."""
    out, dx, grads = ours
    ref_out, ref_dx, ref_grads = reference

    def rel(a, b):
        return ((a.float() - b.float()).abs().sum() / b.float().abs().sum()).item()

    per_param = {n: rel(grads[n], ref_grads[n]) for n in ref_grads}
    worst = max(per_param, key=per_param.get)
    total_diff = sum(
        (grads[n].float() - ref_grads[n].float()).abs().sum() for n in ref_grads
    )
    total_ref = sum(ref_grads[n].float().abs().sum() for n in ref_grads)
    return {
        "out": rel(out, ref_out),
        "dx": rel(dx, ref_dx),
        "params": (total_diff / total_ref).item(),
        "worst_param": worst,
        "worst_param_err": per_param[worst],
    }


def main() -> None:
    """Compare sharded and single-GPU model gradients using CLI options.

    Raises:
        SystemExit: Argument parsing requests help or rejects invalid CLI
            options.
    """
    sys.excepthook = sys.__excepthook__  # CuTe DSL replaces it on import
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--ring", type=int, required=True)
    p.add_argument("--ulysses", type=int, required=True)
    p.add_argument("--layout", choices=LAYOUT_NAMES, default="mixed")
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument(
        "--combine",
        nargs="+",
        default=["online", "flash", "python"],
        choices=["online", "flash", "python"],
    )
    p.add_argument("--no-load-balance", action="store_true")
    p.add_argument("--bf16-grads", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = init_distributed()
    rank = dist.get_rank()
    cfg = MoTConfig()
    docs, splits, modes = build_layout(
        args.layout, args.seq_len, ring_size=max(args.ring, 2), seed=args.seed
    )
    is_gen = torch.cat(
        [
            torch.full((n,), m == "noise", dtype=torch.bool)
            for n, m in zip(splits, modes)
        ]
    ).to(device)
    positions = torch.arange(args.seq_len, device=device)

    # Same weights and inputs on every rank.
    torch.manual_seed(args.seed)
    model = MoTDecoder(cfg, args.layers).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(".bias"):
                param.normal_(0.0, 0.02)
            elif param.ndim == 1:
                param.normal_(1.0, 0.1)
            else:
                param.normal_(0.0, 0.02)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    x = torch.randn(
        args.seq_len, cfg.hidden, device=device, dtype=torch.bfloat16, generator=gen
    )
    w = torch.randn(args.seq_len, cfg.hidden, device=device, generator=gen)

    results = {}
    if rank == 0:
        fp32_model = copy.deepcopy(model).float()

        def exact(q, k, v):
            return reference_attention(
                q, k, v, document_lens=docs, split_lens=splits, attn_modes=modes
            )[0]

        results["fp32"] = _run(fp32_model, x.float(), w, is_gen, positions, exact, cfg)
        del fp32_model
        mask = build_fa4_sparse_mask(docs, splits, modes, device=device)
        results["fa4"] = _run(
            model,
            x,
            w,
            is_gen,
            positions,
            lambda q, k, v: flash4_attn(q, k, v, mask),
            cfg,
        )
        results["flex"] = _run(
            model,
            x,
            w,
            is_gen,
            positions,
            _flex_attention_fn(docs, splits, modes, device),
            cfg,
        )
    dist.barrier()

    groups = USPGroups.create(ring_size=args.ring, ulysses_size=args.ulysses)
    for combine in args.combine:
        plan = build_usp_plan(
            document_lens=docs,
            split_lens=splits,
            attn_modes=modes,
            groups=groups,
            config=USPConfig(
                combine_mode=combine,  # type: ignore[arg-type]
                load_balance=not args.no_load_balance,
                fp32_grads=not args.bf16_grads,
            ),
            device=device,
        )

        def shard(t):
            return plan.shard(t[None], seq_dim=1)[0]

        out, dx, grads = _run(
            model,
            shard(x),
            shard(w),
            shard(is_gen),
            shard(positions),
            lambda q, k, v: usp_attention(q, k, v, plan),
            cfg,
        )
        for g in grads.values():
            dist.all_reduce(g)
        out = plan.gather(out[None], seq_dim=1)[0]
        dx = plan.gather(dx[None], seq_dim=1)[0]
        if rank == 0:
            results[f"usp_{combine}"] = (out, dx, grads)

    if rank == 0:
        base = {
            "layout": args.layout,
            "seq_len": args.seq_len,
            "layers": args.layers,
            "ring": args.ring,
            "ulysses": args.ulysses,
            "load_balance": not args.no_load_balance,
            "fp32_grads": not args.bf16_grads,
        }
        for name, res in results.items():
            if name == "fp32":
                continue
            row = {"check": name, "reference": "fp32", **base}
            row.update(_errors(res, results["fp32"]))
            print("RESULT " + result_json(row), flush=True)
            if name.startswith("usp_"):
                row = {"check": f"{name}_vs_fa4", "reference": "fa4", **base}
                row.update(_errors(res, results["fa4"]))
                print("RESULT " + result_json(row), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
