# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

try:
    # [MLR-02.08] Triton ships with CUDA builds of PyTorch only.
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None

# Elements of the running output one program of _merge_step_kernel handles.
_MERGE_BLOCK = 4096


if triton is not None:

    @triton.jit
    def _merge_step_kernel(
        acc_out, acc_lse, out, lse, new_out, new_lse,
        rows, head_dim,
        BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr,
    ):  # fmt: skip
        r = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
        dd = tl.arange(0, BLOCK_D)
        rmask = r < rows
        old = tl.load(acc_lse + r, mask=rmask, other=float("-inf"))
        part = tl.load(lse + r, mask=rmask, other=float("-inf"))
        top = tl.maximum(old, part)
        # The step's weight sigmoid(part - old), from the LSE difference as in
        # the PyTorch merge: exp(part - new) would carry new's rounding.
        d = tl.where(top == float("-inf"), 0.0, part - old)
        e = libdevice.exp(-tl.abs(d))
        s = tl.math.div_rn(1.0, 1.0 + e)
        w = tl.where(d > 0, s, e * s)[:, None]
        tl.store(new_lse + r, top + libdevice.log1p(e), mask=rmask)
        mask = rmask[:, None] & (dd < head_dim)[None, :]
        offs = r[:, None] * head_dim + dd[None, :]
        acc = tl.load(acc_out + offs, mask=mask, other=0.0)
        o = tl.load(out + offs, mask=mask, other=0.0)
        merged = acc + w * (o - acc)
        # A side without keys (LSE -inf) may hold any output, even NaN: take
        # the other side as is, and 0 where neither side has keys.
        merged = tl.where((part == float("-inf"))[:, None], acc, merged)
        merged = tl.where((old == float("-inf"))[:, None], o, merged)
        merged = tl.where((top == float("-inf"))[:, None], 0.0, merged)
        tl.store(new_out + offs, merged, mask=mask)


def _use_fused_merge(
    out: Tensor, lse: Tensor, step_out: Tensor, step_lse: Tensor
) -> bool:
    """Whether one fp32 merge step can run as :func:`_merge_step_kernel`."""
    tensors = (out, lse, step_out, step_lse)
    return (
        triton is not None
        and step_out.is_cuda
        and all(t.dtype == torch.float32 and t.is_contiguous() for t in tensors)
        and out.shape == step_out.shape
        and lse.shape == step_lse.shape
        and lse.shape[: out.ndim - 1] == out.shape[:-1]
        and lse.numel() * out.shape[-1] == out.numel()
    )


def _merge_step_fused(
    out: Tensor, lse: Tensor, step_out: Tensor, step_lse: Tensor
) -> tuple[Tensor, Tensor]:
    """Merge one fp32 step into the fp32 running ``(out, lse)`` in one pass.

    Returns new tensors. Writing in place would race: the kernel reads each
    LSE in two thread layouts, so one warp could see another's new LSE as old.
    A row whose LSE is ``-inf`` on both sides stays ``-inf`` with output 0.
    """
    new_out, new_lse = torch.empty_like(out), torch.empty_like(lse)
    rows, head_dim = lse.numel(), out.shape[-1]
    block_d = triton.next_power_of_2(head_dim)
    block_r = max(1, _MERGE_BLOCK // block_d)
    _merge_step_kernel[(triton.cdiv(rows, block_r),)](
        out, lse, step_out, step_lse, new_out, new_lse,
        rows, head_dim,
        BLOCK_R=block_r, BLOCK_D=block_d,
    )  # fmt: skip
    return new_out, new_lse


@dataclass(kw_only=True, slots=True)
class _LocalRingResultMerger:
    """Numerically stable merge of per-step attention outputs.

    A row with no visible keys so far has a running LSE of ``-inf``; it takes
    the step's partial as is (as :func:`~usp_attention.ring.dense_merge.merge_into`
    does), since the sigmoid form is ``nan`` there. Rows the step adds no keys
    to (step LSE ``-inf``, finite output: FA4 writes zeros) keep their running
    result.
    """

    out: Tensor | None = None
    lse: Tensor | None = None

    def step(self, step_out: Tensor, step_lse: Tensor) -> None:
        if step_lse.ndim < step_out.ndim:
            step_lse = step_lse.unsqueeze(-1)
        if self.out is None:
            self.out = step_out
            self.lse = step_lse
            return
        assert self.lse is not None
        lse = self.lse
        out = self.out
        if _use_fused_merge(out, lse, step_out, step_lse):
            # fp32 partials (combine_mode="fp32"): one Triton pass per step,
            # which also handles -inf rows on either side.
            self.out, self.lse = _merge_step_fused(out, lse, step_out, step_lse)
            return
        # Guards act on LSE-sized tensors only; finite rows are unchanged.
        empty = lse == float("-inf")
        weight = torch.where(empty, 1.0, torch.sigmoid(step_lse - lse))
        out = out - weight * (out - step_out)
        lse = torch.where(empty, step_lse, lse - F.logsigmoid(lse - step_lse))
        self.out = out
        self.lse = lse

    def result(self) -> tuple[Tensor, Tensor]:
        if self.out is None or self.lse is None:
            raise RuntimeError("No ring steps were merged.")
        lse = self.lse
        if lse.ndim == self.out.ndim and lse.shape[-1] == 1:
            lse = lse.squeeze(-1)
        return self.out, lse


@dataclass(kw_only=True, slots=True)
class _LocalRingResultMergerFlash:
    """Numerically stable merge of per-step attention outputs using flash-attn's dedicated combine kernel."""

    out: Tensor | None = None
    lse: Tensor | None = None

    def step(self, step_out: Tensor, step_lse: Tensor) -> None:
        if self.out is None:
            self.out = step_out
            self.lse = step_lse
            return
        assert self.lse is not None
        lse = self.lse
        out = self.out
        out, lse = _combine_partials_flash([out, step_out], [lse, step_lse])
        self.out = out
        self.lse = lse

    def result(self) -> tuple[Tensor, Tensor]:
        if self.out is None or self.lse is None:
            raise RuntimeError("No ring steps were merged.")
        lse = self.lse
        if self.lse.ndim == self.out.ndim and self.lse.shape[-1] == 1:
            lse = lse.squeeze(-1)
        return self.out, lse


def _normalize_step_lse(step_out: Tensor, step_lse: Tensor) -> Tensor:
    """Normalize LSE shape to (B, S, H) for combine kernels.

    Raises:
        ValueError: If ``step_lse`` shape cannot be aligned with ``step_out``.
    """
    if step_lse.ndim == step_out.ndim and step_lse.shape[-1] == 1:
        step_lse = step_lse.squeeze(-1)
    if step_lse.ndim != step_out.ndim - 1:
        raise ValueError(
            f"Expected step_lse to have one fewer dim than step_out, got "
            f"{step_lse.shape=} and {step_out.shape=}."
        )
    expected = step_out.shape[:-1]
    if step_lse.shape == expected:
        return step_lse

    if step_lse.ndim >= 2 and step_lse.shape[:-2] == expected[:-2]:
        if step_lse.shape[-2:] == (expected[-1], expected[-2]):
            return step_lse.transpose(-1, -2).contiguous()

    raise ValueError(
        "Unable to align step_lse shape for combine kernel: "
        f"{step_lse.shape=} vs expected {expected}."
    )


def _combine_partials_flash(
    out_partials: list[Tensor],
    lse_partials: list[Tensor],
) -> tuple[Tensor, Tensor]:
    """Combine split outputs using flash-attn's dedicated combine kernel.

    Raises:
        RuntimeError: If no partials are provided, ``flash_attn_combine`` is
            unavailable, or the combine kernel returns ``lse=None``.
        ValueError: If ``out_partials`` and ``lse_partials`` have different lengths.
    """
    if len(out_partials) == 0:
        raise RuntimeError("No step partials were collected for flash combine.")
    if len(out_partials) != len(lse_partials):
        raise ValueError(
            f"Mismatched partial count for combine: {len(out_partials)=}, "
            f"{len(lse_partials)=}."
        )

    try:
        from flash_attn.cute.interface import flash_attn_combine
    except ImportError as exc:
        raise RuntimeError(
            "flash_attn_combine is unavailable; install flash_attn.cute to use "
            "combine_mode='flash'."
        ) from exc

    out_partial = torch.stack([x.to(torch.float32) for x in out_partials], dim=0)
    lse_hs = torch.stack(
        [x.transpose(-1, -2).contiguous() for x in lse_partials], dim=0
    ).to(torch.float32)
    lse_partial = lse_hs.transpose(-1, -2)
    out, lse = flash_attn_combine(
        out_partial=out_partial,
        lse_partial=lse_partial,
        return_lse=True,
    )
    if lse is None:
        raise RuntimeError("flash_attn_combine unexpectedly returned lse=None.")
    return out.to(out_partials[0].dtype), lse


def _ensure_lse_bwd_layout(lse: Tensor, out: Tensor) -> Tensor:
    """Ensure LSE is in (B, H, S) layout as required by _flash_attn_bwd.

    The forward combine produces LSE in (B, S, H) layout, but
    _flash_attn_bwd asserts shape == (batch_size, num_head, seqlen_q).

    Raises:
        ValueError: If ``lse`` shape cannot be reshaped to ``(B, H, S)``.
    """
    B, S, H = out.shape[0], out.shape[1], out.shape[2]
    if lse.shape == (B, H, S):
        return lse
    if lse.shape == (B, S, H):
        return lse.transpose(-1, -2).contiguous()
    raise ValueError(
        f"Cannot reshape LSE for _flash_attn_bwd: {lse.shape=}, "
        f"expected ({B}, {H}, {S}) or ({B}, {S}, {H})."
    )


def _split_local_shard(
    x: Tensor, *, dim: int, process_group: dist.ProcessGroup
) -> Tensor:
    world_size = dist.get_world_size(process_group)
    rank = dist.get_rank(process_group)
    if x.shape[dim] % world_size != 0:
        raise ValueError(
            f"Sequence dimension must be divisible by world size for sharding, got "
            f"{x.shape[dim]=} and {world_size=}."
        )
    chunks = x.chunk(world_size, dim=dim)
    return chunks[rank].contiguous()


def _all_gather_cat(
    x_local: Tensor, *, dim: int, process_group: dist.ProcessGroup
) -> Tensor:
    x_local = x_local.contiguous()
    world_size = dist.get_world_size(process_group)
    gathered = [torch.empty_like(x_local) for _ in range(world_size)]
    dist.all_gather(gathered, x_local, group=process_group)
    return torch.cat(gathered, dim=dim)
