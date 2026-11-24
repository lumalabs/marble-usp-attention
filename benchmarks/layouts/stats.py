"""Extract model-agnostic document statistics from public multimodal datasets.

Writes one JSON line per document: {"src": ..., "items": [...]} where each item is
["t", text_tokens] (Qwen3 tokenizer), ["i", width, height] or ["v", seconds].

    uv run --with pyarrow --with tokenizers python -I stats.py DL_DIR OUT_DIR
"""

from __future__ import annotations

import ast
import glob
import json
import os
import sys

import pyarrow.parquet as pq
from tokenizers import Tokenizer

dl, out = sys.argv[1], sys.argv[2]
tok = Tokenizer.from_file(os.path.join(dl, "qwen3_tokenizer.json"))


def ntok(texts):
    return [len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False)]


def obelics():
    with open(os.path.join(out, "obelics.jsonl"), "w") as f:
        for path in sorted(glob.glob(os.path.join(dl, "obelics_*.parquet"))):
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(
                columns=["texts", "metadata"], batch_size=2048
            ):
                d = batch.to_pydict()
                for texts, meta in zip(d["texts"], d["metadata"]):
                    meta = json.loads(meta)
                    present = [t for t in texts if t]
                    counts = iter(ntok(present))
                    items = []
                    for t, m in zip(texts, meta):
                        if t:
                            items.append(["t", next(counts)])
                        elif m:
                            w = m.get("original_width") or m.get("rendered_width")
                            h = m.get("original_height") or m.get("rendered_height")
                            if w and h:
                                items.append(["i", int(w), int(h)])
                    f.write(json.dumps({"src": "obelics", "items": items}) + "\n")
            print("done", path, flush=True)


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def molmo2_cap():
    with open(os.path.join(out, "molmo2_cap.jsonl"), "w") as f:
        for path in sorted(glob.glob(os.path.join(dl, "molmo2cap_*.parquet"))):
            cols = [
                "video_start",
                "video_end",
                "clip_timestamps",
                "merged_caption",
                "video_caption",
            ]
            d = pq.read_table(path, columns=cols).to_pydict()
            caps = [c or "" for c in d["merged_caption"]]
            short = [c or "" for c in d["video_caption"]]
            n_merged, n_short = ntok(caps), ntok(short)
            for i in range(len(caps)):
                start, end = _num(d["video_start"][i]), _num(d["video_end"][i])
                if start is not None and end is not None and end > start:
                    dur = end - start
                else:
                    clips = d["clip_timestamps"][i]
                    clips = ast.literal_eval(clips) if isinstance(clips, str) else clips
                    dur = float(clips[-1][1]) if clips else 0.0
                if dur <= 0:
                    continue
                rec = {
                    "src": "molmo2_cap",
                    "items": [["v", round(dur, 2)], ["t", n_merged[i]]],
                    "short_caption_tokens": n_short[i],
                }
                f.write(json.dumps(rec) + "\n")
            print("done", path, flush=True)


if __name__ == "__main__":
    molmo2_cap()
    obelics()
