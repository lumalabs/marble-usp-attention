"""BAGEL-style packed layouts (unified understanding + generation, MoT on Qwen2.5-7B).

Token counts follow BAGEL's setup: SigLIP2 ViT (14 px patches, NaViT up to 980 px)
for understanding, FLUX VAE (8x) latents in 2x2 patches for generation (512 px ->
1024 tokens, 1024 px -> 4096). A generated image appears as noised VAE tokens
("noise": bidirectional within, invisible to everything else) followed, in
interleaved sequences, by its clean VAE tokens ("full") that later tokens attend.

    bagel_mix:   packed training mix (understanding, T2I, editing, interleaved video)
    bagel_video: long interleaved video generation, one 128k-1M document per sample

Writes layouts/bagel_{mix,video}_{1024k,...,16384k}.json in the file-layout format.
"""

from __future__ import annotations

import json
import random
import sys

VIT = (729, 1369, 2500, 4900)  # 378 / 518 / 700 / 980 px
VAE = (1024, 4096)  # 512 / 1024 px


def und(rng):
    s = [rng.randint(16, 400), rng.choice(VIT), rng.randint(16, 600)]
    return s, ["causal", "full", "causal"]


def t2i(rng):
    return [rng.randint(16, 300), rng.choice(VAE)], ["causal", "noise"]


def edit(rng):
    v = rng.choice(VAE)
    return [rng.randint(16, 200), rng.choice(VIT), v, rng.randint(8, 64), v], [
        "causal",
        "full",
        "full",
        "causal",
        "noise",
    ]


def video(rng, frames, frame=1024):
    s, m = [rng.randint(32, 300)], ["causal"]
    for f in range(frames):
        s += [frame, frame]
        m += ["noise", "full"]
        if rng.random() < 0.3:
            s.append(rng.randint(8, 48))
            m.append("causal")
    s.append(rng.randint(8, 64))
    m.append("causal")
    return s, m


def pack(seq_len, sample, rng):
    docs, splits, modes = [], [], []
    while True:
        s, m = sample(rng)
        if sum(docs) + sum(s) > seq_len:
            break
        docs.append(sum(s))
        splits += s
        modes += m
    # Pad the tail with one causal text document.
    rest = seq_len - sum(docs)
    if rest:
        docs.append(rest)
        splits.append(rest)
        modes.append("causal")
    return {"document_lens": docs, "split_lens": splits, "attn_modes": modes}


def mix_sample(rng):
    r = rng.random()
    if r < 0.3:
        return und(rng)
    if r < 0.65:
        return t2i(rng)
    if r < 0.85:
        return edit(rng)
    return video(rng, rng.randint(8, 32))


def long_video_sample(rng):
    # 64-480 frames of 1024 tokens, noise + clean each: 128k-1M tokens.
    return video(rng, rng.randint(64, 480))


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "layouts"
    for name, sample in (("mix", mix_sample), ("video", long_video_sample)):
        for size, n in (
            ("1024k", 1 << 20),
            ("2048k", 1 << 21),
            ("4096k", 1 << 22),
            ("8192k", 1 << 23),
            ("16384k", 1 << 24),
        ):
            rng = random.Random(f"bagel-{name}-{size}")
            lay = pack(n, sample, rng)
            assert sum(lay["document_lens"]) == n == sum(lay["split_lens"])
            json.dump({"layouts": [lay]}, open(f"{out}/bagel_{name}_{size}.json", "w"))
            docs = lay["document_lens"]
            print(
                name,
                size,
                "docs",
                len(docs),
                "max doc",
                max(docs),
                "splits",
                len(lay["split_lens"]),
            )
