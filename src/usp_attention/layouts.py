# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Synthetic packed-sequence layouts for tests and benchmarks.

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

from usp_attention.fa4.flash4_attn import AttentionMode

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


def load_layout_file(spec: str, seq_len: int | None = None) -> PackedSpec:
    """Load one packed layout from a JSON file of recorded or packed batches.

    Args:
        spec: ``path`` or ``path#index``; the file (JSON, optionally gzipped)
            holds ``{"layouts": [{"document_lens", "split_lens", "attn_modes"},
            ...]}`` (as written by ``benchmarks/layouts/pack.py``) and ``index``
            picks one (default 0). ``x.json`` also finds ``x.json.gz``.
        seq_len: Expected total length, checked when given.

    Raises:
        ValueError: If the layout is inconsistent or its length differs.
    """
    import gzip
    import json
    import os

    path, _, index = spec.partition("#")
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path += ".gz"  # the committed packs are gzipped
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        entry = json.load(f)["layouts"][int(index or 0)]
    docs, splits, modes = (
        list(entry["document_lens"]),
        list(entry["split_lens"]),
        list(entry["attn_modes"]),
    )
    total = sum(docs)
    if sum(splits) != total or len(splits) != len(modes):
        raise ValueError(f"Inconsistent layout in {spec}.")
    if seq_len is not None and total != seq_len:
        raise ValueError(f"Layout {spec} has {total} tokens, expected {seq_len}.")
    return docs, splits, modes


def build_layout(
    name: str, seq_len: int, *, ring_size: int = 1, seed: int = 0
) -> PackedSpec:
    """Build a named synthetic layout, or load ``file:<path>[#index]``.

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
