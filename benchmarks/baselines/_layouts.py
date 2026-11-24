# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Dependency-free copy of ``usp_attention.layouts`` for baseline scripts.

Baselines run in their own environments without usp_attention installed.
``tests/test_baseline_layouts.py`` checks this copy matches the original.

Synthetic packed-sequence layouts for tests and benchmarks.

Each builder returns ``(document_lens, split_lens, attn_modes)`` with a total
length of exactly ``seq_len``. Mode semantics follow the FA4 mask:

- ``"causal"``: attends causally to earlier tokens of the same document.
- ``"full"``: bidirectional within the split, causal to earlier splits.
- ``"noise"``: bidirectional within the split and invisible to every other
  split (a generated image being denoised).
- ``"group"``: bidirectional with every other group split of the document,
  causal to the other earlier splits (all image tokens of a sample as one
  vision group).
"""

from __future__ import annotations

import random

import torch
from torch import Tensor

AttentionMode = str

PackedSpec = tuple[list[int], list[int], list[AttentionMode]]

LAYOUT_NAMES = (
    "causal",
    "full",
    "docs",
    "mixed",
    "tiny_kv",
    "vlm",
    "video",
    "vlm_group",
    "video_group",
)


def _random_partition(
    total: int, parts: int, rng: random.Random, minimum: int
) -> list[int]:
    if parts * minimum > total:
        raise ValueError(
            f"Cannot split {total} into {parts} parts of at least {minimum}."
        )
    cuts = sorted(rng.sample(range(1, total - parts * minimum + parts), parts - 1))
    sizes = [
        b - a for a, b in zip([0, *cuts], [*cuts, total - parts * minimum + parts])
    ]
    return [s - 1 + minimum for s in sizes]


def causal_layout(seq_len: int) -> PackedSpec:
    """One causal document (decoder LM)."""
    return [seq_len], [seq_len], ["causal"]


def full_layout(seq_len: int) -> PackedSpec:
    """One bidirectional document (video DiT)."""
    return [seq_len], [seq_len], ["full"]


def docs_layout(seq_len: int, *, num_docs: int = 8, seed: int = 0) -> PackedSpec:
    """Document-packed causal LM: ``num_docs`` causal documents of random length."""
    docs = _random_partition(seq_len, num_docs, random.Random(seed), minimum=256)
    return docs, list(docs), ["causal"] * num_docs


def mixed_layout(seq_len: int, *, num_docs: int = 4, seed: int = 0) -> PackedSpec:
    """Interleaved multimodal samples, as in unified understanding + generation.

    Each document is ``[text causal][image full][text causal][image noise]``
    with random proportions: an understanding image, its caption, and a
    generated image that only attends to itself and earlier context.
    """
    rng = random.Random(seed)
    docs = _random_partition(seq_len, num_docs, rng, minimum=1024)
    split_lens: list[int] = []
    attn_modes: list[AttentionMode] = []
    for doc in docs:
        parts = _random_partition(doc, 4, rng, minimum=64)
        split_lens.extend(parts)
        attn_modes.extend(["causal", "full", "causal", "noise"])
    return docs, split_lens, attn_modes


def _vlm_document(
    doc_len: int,
    rng: random.Random,
    *,
    tokens_per_image: int,
    max_crops: int,
    text_range: tuple[int, int],
) -> tuple[list[int], list[AttentionMode]]:
    """One interleaved image-text document: causal text spans between images.

    An image is one or more crops of ``tokens_per_image`` tokens. As in
    Gemma 3, every crop is its own bidirectional span, and the end- and
    start-of-image markers between two crops are causal tokens. Text before,
    between and after the images is causal. The document starts and ends with
    text.
    """
    splits: list[int] = []
    modes: list[AttentionMode] = []
    used = 0
    lo, hi = text_range
    while True:
        text = rng.randint(lo, hi)
        crops = rng.randint(1, max_crops)
        image = crops * tokens_per_image + (crops - 1) * 2
        # Close the document with a text span once another image no longer
        # fits ahead of a minimal closing text.
        if used + text + image + lo > doc_len:
            splits.append(doc_len - used)
            modes.append("causal")
            return splits, modes
        splits.append(text)
        modes.append("causal")
        for crop in range(crops):
            if crop:
                splits.append(2)
                modes.append("causal")
            splits.append(tokens_per_image)
            modes.append("full")
        used += text + image


def vlm_layout(
    seq_len: int,
    *,
    num_docs: int = 8,
    seed: int = 0,
    tokens_per_image: int = 256,
    max_crops: int = 4,
) -> PackedSpec:
    """Packed interleaved image-text documents, as in vision-language models.

    VLMs such as Gemma 3 read text causally and each image crop
    bidirectionally: every crop of ``tokens_per_image`` tokens (``1..max_crops``
    per image) is a ``full`` span that sees the text and crops before it, and
    later text sees the whole image. About two thirds of the tokens are image
    tokens.
    """
    rng = random.Random(seed)
    num_docs = max(1, min(num_docs, seq_len // 4096))
    docs = _random_partition(seq_len, num_docs, rng, minimum=min(4096, seq_len))
    split_lens: list[int] = []
    attn_modes: list[AttentionMode] = []
    for doc in docs:
        splits, modes = _vlm_document(
            doc,
            rng,
            tokens_per_image=tokens_per_image,
            max_crops=max_crops,
            text_range=(32, 512),
        )
        split_lens += splits
        attn_modes += modes
    return docs, split_lens, attn_modes


def vlm_group_layout(seq_len: int, *, num_docs: int = 8, seed: int = 0) -> PackedSpec:
    """Interleaved image-text documents with one vision group per document.

    As in Molmo2, all image tokens of a document form one bidirectional group:
    every image crop also sees the later crops and images of its document,
    while text stays causal. Same spans as :func:`vlm_layout`.
    """
    docs, split_lens, attn_modes = vlm_layout(seq_len, num_docs=num_docs, seed=seed)
    return docs, split_lens, ["group" if m == "full" else m for m in attn_modes]


def video_layout(
    seq_len: int,
    *,
    num_docs: int = 2,
    seed: int = 0,
    tokens_per_frame: int = 256,
) -> PackedSpec:
    """Packed long videos for a video VLM.

    Each document is a causal prompt, then frames of ``tokens_per_frame``
    tokens, each a bidirectional ``full`` span preceded by a short causal
    timestamp, then a causal answer.
    """
    rng = random.Random(seed)
    num_docs = max(1, min(num_docs, seq_len // 8192))
    docs = _random_partition(seq_len, num_docs, rng, minimum=min(8192, seq_len))
    split_lens: list[int] = []
    attn_modes: list[AttentionMode] = []
    for doc in docs:
        prompt, stamp = rng.randint(128, 1024), 8
        frames = (doc - prompt - 256) // (stamp + tokens_per_frame)
        answer = doc - prompt - frames * (stamp + tokens_per_frame)
        for frame in range(frames):
            # The first frame's timestamp joins the prompt's causal span.
            split_lens += [prompt + stamp if frame == 0 else stamp, tokens_per_frame]
            attn_modes += ["causal", "full"]
        split_lens.append(answer)
        attn_modes.append("causal")
    return docs, split_lens, attn_modes


def video_group_layout(seq_len: int, *, num_docs: int = 2, seed: int = 0) -> PackedSpec:
    """Long videos with one vision group per video, as in Molmo2.

    Same spans as :func:`video_layout`, but every frame also sees the later
    frames of its video (all frames form one bidirectional group), while the
    prompt, timestamps and answer stay causal.
    """
    docs, split_lens, attn_modes = video_layout(seq_len, num_docs=num_docs, seed=seed)
    return docs, split_lens, ["group" if m == "full" else m for m in attn_modes]


def tiny_kv_layout(seq_len: int, *, ring_size: int, tiny: int = 8) -> PackedSpec:
    """Adversarial layout for ring numerics.

    Every ring shard boundary is crossed by a causal document that has only
    ``tiny`` tokens in the earlier shard. Rows in the later shard then combine
    a ring step with ``tiny`` visible keys with a step that has thousands, the
    case where per-step bf16 partials lose the most precision.
    """
    shard = seq_len // ring_size
    if shard <= 2 * tiny:
        raise ValueError(f"Shard length {shard} too short for tiny={tiny}.")
    docs = [shard - tiny]
    for _ in range(ring_size - 1):
        docs.append(shard)
    docs[-1] += tiny
    return docs, list(docs), ["causal"] * len(docs)


def load_layout_file(spec: str, seq_len: int | None = None):
    """``path[#index]`` from a packed-layout JSON (see ``usp_attention.layouts``).

    ``x.json`` also finds ``x.json.gz``: the committed packs under
    ``benchmarks/layouts/packs`` are gzipped, and ``sweep.py`` names them
    without the suffix (as ``usp_attention.layouts.load_layout_file`` does).
    """
    import gzip
    import json
    import os

    path, _, index = spec.partition("#")
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path += ".gz"
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        entry = json.load(f)["layouts"][int(index or 0)]
    docs, splits, modes = (
        list(entry["document_lens"]),
        list(entry["split_lens"]),
        list(entry["attn_modes"]),
    )
    if sum(docs) != sum(splits) or (seq_len is not None and sum(docs) != seq_len):
        raise ValueError(f"Layout {spec} does not have {seq_len} tokens.")
    return docs, splits, modes


def build_layout(
    name: str, seq_len: int, *, ring_size: int = 1, seed: int = 0
) -> PackedSpec:
    """Build a named synthetic layout.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    if name.startswith("file:"):
        return load_layout_file(name.removeprefix("file:"), seq_len)
    if name == "causal":
        return causal_layout(seq_len)
    if name == "full":
        return full_layout(seq_len)
    if name == "docs":
        return docs_layout(seq_len, seed=seed)
    if name == "mixed":
        return mixed_layout(seq_len, seed=seed)
    if name == "tiny_kv":
        return tiny_kv_layout(seq_len, ring_size=max(ring_size, 2))
    if name == "vlm":
        return vlm_layout(seq_len, seed=seed)
    if name == "video":
        return video_layout(seq_len, seed=seed)
    if name == "vlm_group":
        return vlm_group_layout(seq_len, seed=seed)
    if name == "video_group":
        return video_group_layout(seq_len, seed=seed)
    raise ValueError(f"Unknown layout {name!r}; expected one of {LAYOUT_NAMES}.")


def visible_key_counts(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[AttentionMode],
) -> Tensor:
    """Exact number of visible keys for every query row, in closed form.

    A causal token sees the earlier tokens of its document (itself included);
    a full or noise token sees its whole split plus the earlier tokens of its
    document. Tokens of an earlier noise split are invisible to everyone
    outside that split. A group token also sees every later group split of
    its document. Splits never straddle documents.

    Returns:
        ``int64`` tensor of shape ``(seq_len,)`` on CPU.
    """
    counts = []
    split_iter = iter(zip(split_lens, attn_modes))
    for doc_len in document_lens:
        doc_splits = []
        offset = 0
        while offset < doc_len:
            length, mode = next(split_iter)
            doc_splits.append((length, mode))
            offset += length
        if offset != doc_len:
            raise ValueError("A split straddles a document boundary.")
        group_after = sum(n for n, m in doc_splits if m == "group")
        offset = 0
        hidden_noise = 0
        for length, mode in doc_splits:
            visible_before = offset - hidden_noise
            if mode == "group":
                group_after -= length
            if mode == "causal":
                counts.append(
                    visible_before + torch.arange(1, length + 1, dtype=torch.int64)
                )
            else:
                # A group token also sees the group splits after its own.
                later = group_after if mode == "group" else 0
                counts.append(
                    torch.full(
                        (length,), visible_before + length + later, dtype=torch.int64
                    )
                )
            if mode == "noise":
                hidden_noise += length
            offset += length
    return torch.cat(counts)


def dense_mask(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[AttentionMode],
    device: torch.device | str = "cpu",
    sliding_window: int | None = None,
) -> Tensor:
    """The packed layout's ``(seq_len, seq_len)`` boolean mask, True = visible.

    With ``sliding_window``, causally visible keys must also satisfy
    ``q - k < sliding_window``; a token still sees its whole full, noise or
    group span (as in ``usp_attention.sliding_window``).

    Same rule as :func:`visible_key_counts`: a query sees the non-noise tokens
    of earlier splits in its document, and its own split causally (causal
    splits) or in full (full and noise splits); a group token also sees every
    group token of its document.
    """
    doc_id, split_id, causal, noise, group = [], [], [], [], []
    split_iter = iter(zip(split_lens, attn_modes))
    split_index = 0
    for d, doc_len in enumerate(document_lens):
        offset = 0
        while offset < doc_len:
            length, mode = next(split_iter)
            doc_id += [d] * length
            split_id += [split_index] * length
            causal += [mode == "causal"] * length
            noise += [mode == "noise"] * length
            group += [mode == "group"] * length
            split_index += 1
            offset += length
    doc_t, split_t, causal_t, noise_t, group_t = (
        torch.tensor(x, device=device) for x in (doc_id, split_id, causal, noise, group)
    )
    pos = torch.arange(len(doc_id), device=device)
    same_doc = doc_t[:, None] == doc_t[None, :]
    earlier = (split_t[None, :] < split_t[:, None]) & ~noise_t[None, :]
    own = (split_t[None, :] == split_t[:, None]) & (
        ~causal_t[:, None] | (pos[None, :] <= pos[:, None])
    )
    vision_group = group_t[:, None] & group_t[None, :]
    if sliding_window:
        near = (pos[:, None] - pos[None, :]) < sliding_window
        earlier = earlier & near
        own = own & (~causal_t[:, None] | near)
    return same_doc & (earlier | own | vision_group)


def dense_reference(
    q: Tensor, k: Tensor, v: Tensor, dout: Tensor, mask: Tensor
) -> list[Tensor]:
    """fp32 attention output and dq / dk / dv for full ``(B, S, H, D)`` inputs.

    Returns ``[out, dq, dk, dv]`` in fp32, ``(B, S, H, D)`` layout. Grouped
    query attention is handled by repeating K/V heads.
    """
    qf, kf, vf = (t.detach().float().requires_grad_(True) for t in (q, k, v))
    rep = q.shape[2] // k.shape[2]
    kr, vr = (t.repeat_interleave(rep, dim=2) for t in (kf, vf))
    out = torch.nn.functional.scaled_dot_product_attention(
        qf.transpose(1, 2), kr.transpose(1, 2), vr.transpose(1, 2), attn_mask=mask
    ).transpose(1, 2)
    out.backward(dout.float())
    return [out.detach(), qf.grad, kf.grad, vf.grad]


def mean_rel_errors(ours: list[Tensor], reference: list[Tensor]) -> list[float]:
    """Mean relative error ``mean|a - b| / mean|b|`` of each pair, as floats."""
    return [
        ((a.float() - b).abs().mean() / b.abs().mean()).item()
        for a, b in zip(ours, reference)
    ]
