# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Exact packed mask and online-state edge cases on CDNA3."""

from __future__ import annotations

import pytest
import torch

from usp_attention.distributed import USPGroups

try:
    from usp_attention.online_flash_amd import online_flash_attn_func
    from usp_attention.online_flash_amd.distributed import (
        build_forward_plan,
        sparse_usp_forward,
    )
    from usp_attention.online_flash_amd.exchange import _pack, _unpack, gather_qkv
    from usp_attention.online_flash_amd.sparse import build_sparse_plan
except ImportError as exc:
    pytest.skip(
        f"AMD online flash extension is not built: {exc}", allow_module_level=True
    )

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.version.hip or not torch.cuda.is_available(),
        reason="requires ROCm and a GPU",
    ),
]


def metadata(n: int, device: str = "cuda") -> torch.Tensor:
    pos = torch.arange(n, device=device, dtype=torch.int32)
    seg = torch.full_like(pos, -1)
    noise = seg.clone()
    # Multiple generation groups, follow-up text, and unaligned document edges.
    seg[31:160] = 1
    seg[161:260] = 2
    noise[161:260] = 2
    seg[277:389] = 3
    noise[277:389] = 3
    return torch.stack([seg, noise, pos // 431, pos])


def reference(q, k, v, qa, ka):
    mask = (qa[3, :, None] >= ka[3, None, :]) | (
        (qa[0, :, None] >= 0) & (qa[0, :, None] == ka[0, None, :])
    )
    mask &= (ka[1, None, :] < 0) | (qa[1, :, None] == ka[1, None, :])
    mask &= qa[2, :, None] == ka[2, None, :]
    groups = q.shape[2] // k.shape[2]
    scores = (
        q.float().transpose(1, 2)
        @ k.float().repeat_interleave(groups, 2).transpose(1, 2).transpose(-1, -2)
        / 128**0.5
    )
    probs = scores.masked_fill(~mask, -torch.inf).softmax(-1).nan_to_num()
    return (probs @ v.float().repeat_interleave(groups, 2).transpose(1, 2)).transpose(
        1, 2
    )


@pytest.mark.parametrize("layout", ["bshd", "bhsd"])
@pytest.mark.parametrize("chunks", [1, 2, 4])
@pytest.mark.parametrize("heads", [(4, 4), (4, 1), (28, 4)])
def test_packed_permuted_stream(layout, chunks, heads):
    torch.manual_seed(23)
    n = 901
    qa = metadata(n)
    perm = torch.randperm(n, device="cuda")
    qa = qa[:, perm].contiguous()
    q = torch.randn(1, n, heads[0], 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, n, heads[1], 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    expected = reference(q, k, v, qa, qa)
    if layout == "bhsd":
        q, k, v = (x.transpose(1, 2).contiguous() for x in (q, k, v))
        expected = expected.transpose(1, 2)
    dim = 1 if layout == "bshd" else 2
    out = state = None
    for i, (kc, vc, ka) in enumerate(
        zip(k.chunk(chunks, dim), v.chunk(chunks, dim), qa.chunk(chunks, 1))
    ):
        out, state = online_flash_attn_func(
            q,
            kc,
            vc,
            state=state,
            out=out,
            is_online_first=i == 0,
            is_online_last=i == chunks - 1,
            layout=layout,
            sparse_plan=build_sparse_plan(qa, ka),
        )
    torch.testing.assert_close(out.float(), expected, atol=0.004, rtol=0.01)


@pytest.mark.parametrize("empty_step", [0, 1, 2, -1])
def test_empty_steps_and_fully_masked_rows(empty_step):
    q = torch.randn(1, 401, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 79, 1, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    qa, ka = metadata(401), metadata(79)
    # Row 0 is always isolated. One step has no legal tiles at all.
    qa[2, 0] = 99
    out = state = None
    keys, values, metas = [], [], []
    for i in range(3):
        km = ka.clone()
        if i == empty_step or empty_step == -1:
            km[2] = 100
        keys.append(k)
        values.append(v)
        metas.append(km)
        out, state = online_flash_attn_func(
            q,
            k,
            v,
            state=state,
            out=out,
            is_online_first=i == 0,
            is_online_last=i == 2,
            sparse_plan=build_sparse_plan(qa, km),
        )
    expected = reference(
        q, torch.cat(keys, 1), torch.cat(values, 1), qa, torch.cat(metas, 1)
    )
    torch.testing.assert_close(out.float(), expected, atol=0.004, rtol=0.01)
    assert torch.count_nonzero(out[:, 0]) == 0


@pytest.mark.parametrize("round_mode", ["rtne", "rtna", "rtz"])
def test_full_partial_empty_tiles(round_mode):
    """Exercise all three tile classes and shared-mask batch broadcasting."""
    torch.manual_seed(91)
    n = 1537
    aux = metadata(n)
    aux[0, 384:768] = 5
    aux[1, 384:768] = -1
    aux[2, :768] = 0
    q = torch.randn(2, n, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, n, 1, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    plan = build_sparse_plan(aux, aux)
    assert set(plan.types.unique().tolist()) == {0, 1, 2}
    out, _ = online_flash_attn_func(
        q,
        k,
        v,
        is_online_first=True,
        is_online_last=True,
        sparse_plan=plan,
        round_mode=round_mode,
    )
    torch.testing.assert_close(
        out.float(), reference(q, k, v, aux, aux), atol=0.004, rtol=0.01
    )


def test_partial_bitmaps_match_predicate_with_padding():
    qa = metadata(901)
    ka = metadata(517)
    qa = qa[:, torch.randperm(901, device="cuda")]
    ka = ka[:, torch.randperm(517, device="cuda")]
    plan = build_sparse_plan(qa, ka)
    legal = (qa[3, :, None] >= ka[3, None, :]) | (
        (qa[0, :, None] >= 0) & (qa[0, :, None] == ka[0, None, :])
    )
    legal &= (ka[1, None, :] < 0) | (qa[1, :, None] == ka[1, None, :])
    legal &= qa[2, :, None] == ka[2, None, :]
    padded = torch.zeros((1152, 576), dtype=torch.bool, device="cuda")
    padded[:901, :517] = legal
    bits = torch.arange(32, device="cuda")
    decoded = ((plan.mask_bits[..., None] >> bits) & 1).bool().reshape(-1, 384, 64)
    for qb, kb in (plan.types == 1).nonzero().tolist():
        offset = plan.mask_offsets[qb, kb]
        torch.testing.assert_close(
            decoded[offset], padded[qb * 384 : (qb + 1) * 384, kb * 64 : (kb + 1) * 64]
        )
    assert (plan.mask_offsets[plan.types != 1] == -1).all()


@pytest.mark.parametrize("heads", [2, 7])
@torch.no_grad()
def test_packed_query_heads(heads):
    """Head folding preserves batch, document, noise and original-position axes."""
    torch.manual_seed(45)
    n = 1152
    aux = metadata(n)
    aux = aux[:, torch.randperm(n, device="cuda")]
    # Include a one-token document.
    aux[2, 0] = 99
    groups = USPGroups(
        ring_size=1,
        ulysses_size=1,
        ring_rank=0,
        ulysses_rank=0,
        cp_rank=0,
        ring_group=None,
        ulysses_group=None,
        cp_group=None,
    )
    plan = build_forward_plan(aux, groups, query_head_factor=heads)
    q = torch.randn(2, n, heads, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, n, 1, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    out = sparse_usp_forward(q, k, v, plan)
    torch.testing.assert_close(
        out.float(), reference(q, k, v, aux, aux), atol=0.004, rtol=0.01
    )


@torch.no_grad()
def test_sparse_schedule_does_not_change_online_state():
    """Each chunk may schedule query blocks differently; state uses logical IDs."""
    torch.manual_seed(46)
    n = 1153
    aux = metadata(n)
    q = torch.randn(2, n, 7, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, n, 1, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    chunks = list(zip(k.chunk(3, 1), v.chunk(3, 1), aux.chunk(3, 1)))
    outputs = []
    for reverse in (False, True):
        out = state = None
        for i, (kc, vc, ka) in enumerate(chunks):
            plan = build_sparse_plan(aux, ka)
            if reverse:
                plan.q_order = plan.q_order.flip(0)
            out, state = online_flash_attn_func(
                q,
                kc,
                vc,
                out=out,
                state=state,
                is_online_first=i == 0,
                is_online_last=i == 2,
                sparse_plan=plan,
            )
        outputs.append(out)
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    torch.testing.assert_close(
        outputs[1].float(), reference(q, k, v, aux, aux), atol=0.004, rtol=0.01
    )


@pytest.mark.parametrize("scale", [0.1, 1.0, 8.0])
@pytest.mark.parametrize("round_mode", ["rtne", "rtna", "rtz"])
@torch.no_grad()
def test_sparse_scaled_online_stream(scale, round_mode):
    """Check uniform/peaked attention through ragged online KV chunks."""
    torch.manual_seed(47)
    n = 1153
    aux = metadata(n)
    q = (torch.randn(1, n, 7, 128, device="cuda") * scale).bfloat16()
    k = (torch.randn(1, n, 1, 128, device="cuda") * scale).bfloat16()
    v = torch.randn_like(k)
    out = state = None
    for i, (kc, vc, ka) in enumerate(
        zip(k.chunk(4, 1), v.chunk(4, 1), aux.chunk(4, 1))
    ):
        out, state = online_flash_attn_func(
            q,
            kc,
            vc,
            out=out,
            state=state,
            round_mode=round_mode,
            is_online_first=i == 0,
            is_online_last=i == 3,
            sparse_plan=build_sparse_plan(aux, ka),
        )
    expected = reference(q, k, v, aux, aux)
    if round_mode == "rtz":
        # RTZ truncates probabilities; use a norm bound for scaled inputs.
        # RTNE/RTNA retain the elementwise reference gate.
        relative_l2 = (out.float() - expected).norm() / expected.norm()
        assert relative_l2 < 0.01
    else:
        torch.testing.assert_close(out.float(), expected, atol=0.004, rtol=0.01)


@pytest.mark.parametrize("dtype,dim", [(torch.float32, 128), (torch.bfloat16, 64)])
def test_gather_rejects_unsupported_storage(dtype, dim):
    q = torch.empty(1, 384, 7, dim, device="cuda", dtype=dtype)
    k = torch.empty(1, 384, 1, dim, device="cuda", dtype=dtype)
    groups = USPGroups(
        ring_size=1,
        ulysses_size=1,
        ring_rank=0,
        ulysses_rank=0,
        cp_rank=0,
        ring_group=None,
        ulysses_group=None,
        cp_group=None,
    )
    with pytest.raises(ValueError):
        gather_qkv(q, k, k, groups)


@pytest.mark.parametrize("u", [1, 2, 4])
def test_exchange_64bit_indexing(u):
    """Exercise the large-buffer address path on small, exactly comparable data."""
    b, s, world = 2, 17, 8
    hq, hk = 28 // u, 4 // u
    q = torch.randn(b, s, 28, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, s, 4, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    splits = [
        b * s * 128 * (2 * hk + (hq if peer // u == 1 else 0)) for peer in range(world)
    ]
    grid = ((b * s * (hq + 2 * hk) * 128 + 4095) // 4096, world)
    results = []
    for wide in (False, True):
        send = torch.empty(sum(splits), device="cuda", dtype=q.dtype)
        _pack[grid](q, k, v, send, b * s, hq, hk, u, 1, 4096, INDEX64=wide)
        qo = torch.empty(b, s * u, hq, 128, device="cuda", dtype=q.dtype)
        ko = torch.empty(b, s * world, hk, 128, device="cuda", dtype=q.dtype)
        vo = torch.empty_like(ko)
        _unpack[grid](send, qo, ko, vo, b, s, hq, hk, u, world, 1, 4096, INDEX64=wide)
        results.append((send, qo, ko, vo))
    for narrow, wide in zip(*results):
        torch.testing.assert_close(narrow, wide, atol=0, rtol=0)
