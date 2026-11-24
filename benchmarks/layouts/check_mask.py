"""Brute-force check of pack.py's analytic mask density on tiny (L=4096) packed sequences.

uv run -q --with numpy --with tokenizers python -I check_mask.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pack  # noqa: E402

L = 4096


def brute(lay, window=None):
    doc = np.repeat(np.arange(len(lay["document_lens"])), lay["document_lens"])
    split = np.repeat(np.arange(len(lay["split_lens"])), lay["split_lens"])
    code = {"causal": 0, "full": 1, "group": 2}
    mode = np.repeat(np.array([code[m] for m in lay["attn_modes"]]), lay["split_lens"])
    q = np.arange(L)[:, None]
    k = np.arange(L)[None, :]
    same_doc = doc[:, None] == doc[None, :]
    causal = k <= q
    if window is not None:
        causal &= (q - k) < window
    full = (mode[:, None] == 1) & (split[:, None] == split[None, :])
    group = (mode[:, None] == 2) & (mode[None, :] == 2)
    m = same_doc & (causal | full | group)
    return int(m.sum())


def main():
    mol, obe = pack.load_docs(max_obelics=20000)
    # keep only docs that can fit so the tiny stream does not spin on drops
    pools = {"molmo2_cap": mol, "obelics": obe}
    ok = True
    for model in pack.MODELS:
        lengths = {
            s: np.array([pack.doc_len(model, d) for d in pools[s]]) for s in pools
        }
        for mix in pack.MIXES:
            for seed in range(2):
                st = pack.Stream(mix, pools, lengths, L, seed)
                ((seq, pad),) = pack.pack_sequences(st, L, 1)
                lay, _ = pack.build_layout(model, pools, seq, pad)
                windows = [None, 1024] if model.startswith("gemma") else [None]
                for w in windows:
                    a = round(
                        pack.layout_density(
                            lay["document_lens"],
                            lay["split_lens"],
                            lay["attn_modes"],
                            L,
                            w,
                        )
                        * L
                        * L
                    )
                    b = brute(lay, w)
                    nb = sum(m != "causal" for m in lay["attn_modes"])
                    flag = "OK " if a == b else "BAD"
                    ok &= a == b
                    print(
                        f"{flag} {model:9s} {mix:11s} seed={seed} window={w} docs={len(seq)} splits={len(lay['split_lens'])} "
                        f"bidir={nb} analytic={a} brute={b} density={b / L / L:.4f}"
                    )
    print("ALL MATCH" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
