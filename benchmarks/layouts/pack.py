"""Pack real Molmo2-Cap / OBELICS document statistics into long-context VLM sequence layouts.

Each document becomes a list of contiguous token-level splits with an attention mode:
  causal - text, markers, timestamps (and every Qwen3-VL token)
  full   - one bidirectional block (one Gemma 3 image / frame / crop)
  group  - Molmo2: all group splits of one document are mutually visible

Token rules (see token_rules.md for sources):
  molmo2   - allenai/Molmo2-8B HF processor defaults (max_crops=8, 2x2 image pooling,
             3x3 video pooling, uniform_last_frame at max_fps=2 up to 384 frames).
  qwen3vl  - Qwen/Qwen3-VL-8B-Instruct: images via preprocessor_config (min 65536 / max 16777216 px,
             factor 32); video via qwen-vl-utils reference (2 fps, 4..768 frames, per-frame
             128..768 tokens, total pixel budget MODEL_SEQ_LEN=128000*32*32*0.9).
  gemma3   - google/gemma-3-27b-it: 256 tokens per image, pan&scan off (default).
  gemma3ps - same with pan&scan on for OBELICS images (min crop 256, max 4 crops, ratio 1.2).
  qwen38   - Qwen/Qwen3.8-27B: preprocessor/video_preprocessor configs and processor class are identical to
             Qwen3-VL-8B, so the qwen3vl vision rules apply. Differences: Qwen3.8 tokenizer (248k vocab; text
             counts scaled by the measured ratio) and transformers>=5 keeps the chat template's outer
             <|vision_start|>...<|vision_end|> around a video (+2 tokens per video).
  qwen25vl - (implemented, not run) Qwen/Qwen2.5-VL-7B-Instruct and -32B-Instruct (identical vision tokenization): images via
             preprocessor_config (min 3136 / max 12845056 px, factor 28); video via qwen-vl-utils 0.0.10
             (2 fps, 4..768 frames, per-frame 128*28^2..768*28^2 px, VIDEO_TOTAL_PIXELS 128000*28^2*0.9),
             one <|vision_start|> pads <|vision_end|> block per video, no text timestamps.

Usage:
  uv run -q --with numpy --with tokenizers python -I pack.py [--models ...] [--mixes ...] [--seq-lens ...]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from functools import lru_cache

import numpy as np
from tokenizers import Tokenizer

R = os.path.dirname(os.path.abspath(__file__))
QWEN_TOK = Tokenizer.from_file(os.path.join(R, "dl", "qwen3_tokenizer.json"))
GEMMA_TOK = Tokenizer.from_file(os.path.join(R, "src", "gemma3", "tokenizer.json"))

# Gemma3 / Qwen3 text-token ratio measured on 8277 OBELICS texts and 4000 Molmo2-Cap captions
# (work/tokratio.py): 1.0065 and 0.9938. Text counts in the inputs are Qwen3 counts.
GEMMA_TEXT_RATIO = {"obelics": 1.0065, "molmo2_cap": 0.9938}
# Qwen3.8 / Qwen3 text-token ratio on the same samples (work/tokratio38.py): 1.0103 and 1.0002.
QWEN38_TEXT_RATIO = {"obelics": 1.0103, "molmo2_cap": 1.0002}
# Assumed source video properties for Molmo2-Cap (YouTube); the inputs only carry durations.
VIDEO_FPS = 30.0
VIDEO_HW = (720, 1280)

CAUSAL, FULL, GROUP = "causal", "full", "group"
MODELS = ["molmo2", "qwen3vl", "gemma3", "gemma3ps", "qwen38"]
# "qwen25vl" (Qwen2.5-VL-7B/32B) rules are implemented below but were not run (superseded by qwen38).
MIXES = ["video", "interleaved", "mixed"]
SEQ_LENS = [131072, 262144, 524288, 1048576]
QWEN38_SEQ_LENS = [262144, 524288, 1048576, 2097152]
ALL_SEQ_LENS = [131072, 262144, 524288, 1048576, 2097152]


@lru_cache(maxsize=None)
def qwen_ntok(s):
    return len(QWEN_TOK.encode(s, add_special_tokens=False).ids)


@lru_cache(maxsize=None)
def gemma_ntok(s):
    return len(GEMMA_TOK.encode(s, add_special_tokens=False).ids)


# ----------------------------------------------------------------------------- Molmo2
MOLMO_PATCH = 14
MOLMO_CROP = 378
MOLMO_MARGINS = (4, 4)
MOLMO_MAX_CROPS = 8
MOLMO_IMG_POOL = 2
MOLMO_VID_POOL = 3
MOLMO_MAX_FRAMES = 384
MOLMO_MAX_FPS = 2.0


def _molmo_select_tiling(h, w, patch_size, max_num_crops):
    # Port of image_processing_molmo2.select_tiling.
    tilings = [
        (i, j)
        for i in range(1, max_num_crops + 1)
        for j in range(1, max_num_crops + 1)
        if i * j <= max_num_crops
    ]
    tilings.sort(key=lambda x: (x[0] * x[1], x[0]))
    cand = np.array(tilings, dtype=np.int32)
    res = cand * patch_size
    orig = np.array([h, w], dtype=np.float32)
    with np.errstate(divide="ignore"):
        req = res.astype(np.float32) / orig
    req = np.min(req, axis=-1, keepdims=True)
    if np.all(req < 1):
        ix = np.argmax(req)
    else:
        req = np.where(req < 1.0, 10e9, req)
        ix = np.argmin(req)
    return tuple(int(x) for x in cand[ix])


@lru_cache(maxsize=None)
def molmo_image_tokens(w, h):
    """Tokens of one image: low-res global crop + overlapping high-res crops (all image tokens)."""
    margin_px = MOLMO_PATCH * sum(MOLMO_MARGINS)
    crop_patches = MOLMO_CROP // MOLMO_PATCH  # 27
    window_patches = crop_patches - sum(MOLMO_MARGINS)  # 19
    rows, cols = _molmo_select_tiling(
        h - margin_px, w - margin_px, window_patches * MOLMO_PATCH, MOLMO_MAX_CROPS
    )
    ph = math.ceil((rows * window_patches + sum(MOLMO_MARGINS)) / MOLMO_IMG_POOL)
    pw = math.ceil((cols * window_patches + sum(MOLMO_MARGINS)) / MOLMO_IMG_POOL)
    g = math.ceil(crop_patches / MOLMO_IMG_POOL)  # 14
    low = 1 + g * (g + 1) + 1  # <low_res_im_start> rows(patches + <im_col>) <im_end>
    high = 1 + ph * (pw + 1) + 1  # <im_start> rows(patches + <im_col>) <im_end>
    return low + high


def molmo_frame_times(dur):
    max_duration = (MOLMO_MAX_FRAMES - 1) / MOLMO_MAX_FPS
    if max_duration < dur:
        return np.linspace(0, dur, num=MOLMO_MAX_FRAMES, endpoint=True)
    t = np.arange(0.0, dur, 1 / MOLMO_MAX_FPS)
    return np.concatenate([t, [dur]])


MOLMO_FRAME_TOKENS = (
    1 + (math.ceil((MOLMO_CROP // MOLMO_PATCH) / MOLMO_VID_POOL)) ** 2 + 1
)  # 83


# ----------------------------------------------------------------------------- Qwen3-VL
QWEN_FACTOR = 32  # patch 16 * merge 2
QWEN_IMG_MIN, QWEN_IMG_MAX = 65536, 16777216  # preprocessor_config.json size
QWEN_VID_MIN_TOK, QWEN_VID_MAX_TOK = 128, 768
QWEN_MODEL_SEQ_LEN = 128000
QWEN_FPS, QWEN_MIN_FRAMES, QWEN_MAX_FRAMES = 2.0, 4, 768


def _smart_resize(h, w, factor, min_pixels, max_pixels):
    # transformers image_processing_qwen2_vl.smart_resize
    if h < factor or w < factor:  # qwen-vl-utils / video path upscales tiny inputs
        s = max(factor / h, factor / w)
        h, w = int(h * s), int(w * s)
    if max(h, w) / min(h, w) > 200:
        return None
    hb = round(h / factor) * factor
    wb = round(w / factor) * factor
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb = max(factor, math.floor(h / beta / factor) * factor)
        wb = max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb = math.ceil(h * beta / factor) * factor
        wb = math.ceil(w * beta / factor) * factor
    return hb, wb


@lru_cache(maxsize=None)
def qwen_image_tokens(w, h):
    r = _smart_resize(h, w, QWEN_FACTOR, QWEN_IMG_MIN, QWEN_IMG_MAX)
    if r is None:  # aspect ratio > 200 raises in the processor; such images are skipped
        return 0
    return (r[0] // QWEN_FACTOR) * (r[1] // QWEN_FACTOR)


def qwen_video(dur):
    """Return (tokens per temporal patch, list of timestamps) following qwen-vl-utils + HF processor."""
    total_frames = max(1, int(round(dur * VIDEO_FPS)))
    n = total_frames / VIDEO_FPS * QWEN_FPS
    n = min(min(max(n, QWEN_MIN_FRAMES), QWEN_MAX_FRAMES), total_frames)
    n = max(2, math.floor(n / 2) * 2)
    idx = np.linspace(0, total_frames - 1, n).round().astype(int)
    min_px = QWEN_VID_MIN_TOK * QWEN_FACTOR**2
    total_px = QWEN_MODEL_SEQ_LEN * QWEN_FACTOR**2 * 0.9
    max_px = max(
        min(QWEN_VID_MAX_TOK * QWEN_FACTOR**2, total_px / n * 2), int(min_px * 1.05)
    )
    hb, wb = _smart_resize(VIDEO_HW[0], VIDEO_HW[1], QWEN_FACTOR, min_px, max_px)
    per_patch = (hb // QWEN_FACTOR) * (wb // QWEN_FACTOR)
    ts = idx / VIDEO_FPS
    ts = (ts[0::2] + ts[1::2]) / 2  # Qwen3VLProcessor._calculate_timestamps (merge 2)
    return per_patch, ts


# ----------------------------------------------------------------------------- Qwen2.5-VL (7B / 32B)
Q25_FACTOR = 28  # patch 14 * merge 2
Q25_IMG_MIN, Q25_IMG_MAX = (
    3136,
    12845056,
)  # preprocessor_config.json min_pixels / max_pixels
Q25_VID_MIN_PX, Q25_VID_MAX_PX = 128 * 28 * 28, 768 * 28 * 28  # qwen-vl-utils 0.0.10
Q25_VID_TOTAL_PX = int(128000 * 28 * 28 * 0.9)


@lru_cache(maxsize=None)
def qwen25_image_tokens(w, h):
    r = _smart_resize(h, w, Q25_FACTOR, Q25_IMG_MIN, Q25_IMG_MAX)
    if r is None:
        return 0
    return (r[0] // Q25_FACTOR) * (r[1] // Q25_FACTOR)


def qwen25_video_tokens(dur):
    """Total <|video_pad|> tokens for one video (qwen-vl-utils 0.0.10 fetch_video + HF processor)."""
    total_frames = max(1, int(round(dur * VIDEO_FPS)))
    n = total_frames / VIDEO_FPS * QWEN_FPS
    n = min(
        min(
            max(n, QWEN_MIN_FRAMES),
            math.floor(min(QWEN_MAX_FRAMES, total_frames) / 2) * 2,
        ),
        total_frames,
    )
    n = max(2, math.floor(n / 2) * 2)
    max_px = max(
        min(Q25_VID_MAX_PX, Q25_VID_TOTAL_PX / n * 2), int(Q25_VID_MIN_PX * 1.05)
    )
    hb, wb = _smart_resize(VIDEO_HW[0], VIDEO_HW[1], Q25_FACTOR, Q25_VID_MIN_PX, max_px)
    return (n // 2) * (hb // Q25_FACTOR) * (wb // Q25_FACTOR)


# ----------------------------------------------------------------------------- Gemma 3
GEMMA_IMG = 256
GEMMA_PAS_MIN_CROP, GEMMA_PAS_MAX_CROPS, GEMMA_PAS_RATIO = 256, 4, 1.2
GEMMA_VID_FPS, GEMMA_VID_MAX_FRAMES = (
    1.0,
    128,
)  # ASSUMPTION: Gemma 3 has no video processor


@lru_cache(maxsize=None)
def gemma_pas_crops(w, h):
    if w >= h:
        if w / h < GEMMA_PAS_RATIO:
            return 0
        nw = int(math.floor(w / h + 0.5))
        nw = min(int(math.floor(w / GEMMA_PAS_MIN_CROP)), nw)
        nw = min(GEMMA_PAS_MAX_CROPS, max(2, nw))
        nh = 1
    else:
        if h / w < GEMMA_PAS_RATIO:
            return 0
        nh = int(math.floor(h / w + 0.5))
        nh = min(int(math.floor(h / GEMMA_PAS_MIN_CROP)), nh)
        nh = min(GEMMA_PAS_MAX_CROPS, max(2, nh))
        nw = 1
    if min(math.ceil(w / nw), math.ceil(h / nh)) < GEMMA_PAS_MIN_CROP:
        return 0
    return nw * nh


def gemma_image_spans(w, h, pas):
    # "\n\n<start_of_image>" + 256 x <image_soft_token> + "<end_of_image>\n\n"
    one = [(2, CAUSAL, 0), (GEMMA_IMG, FULL, 1), (2, CAUSAL, 0)]
    k = gemma_pas_crops(w, h) if pas else 0
    if not k:
        return one
    spans = [(gemma_ntok("Here is the original image "), CAUSAL, 0)] + one
    spans.append(
        (gemma_ntok(" and here are some crops to help you see better "), CAUSAL, 0)
    )
    for i in range(k):
        if i:
            spans.append((1, CAUSAL, 0))  # " " joiner
        spans += one
    return spans


# ----------------------------------------------------------------------------- documents
def doc_spans(model, doc):
    """List of (length, mode, is_vision) spans for one document (before merging)."""
    src, items = doc
    sp = []
    if src == "molmo2_cap":
        dur, cap = items
        if model == "molmo2":
            sp.append((1, CAUSAL, 0))  # BOS inserted by Molmo2Processor.insert_bos
            for i, t in enumerate(molmo_frame_times(dur)):
                sp.append((qwen_ntok((" " if i else "") + f"{t:.1f} "), CAUSAL, 0))
                sp.append((MOLMO_FRAME_TOKENS, GROUP, 1))
            sp.append(
                (
                    3 + qwen_ntok("Describe this video in detail.") + 5 + cap + 1,
                    CAUSAL,
                    0,
                )
            )
        elif model in ("qwen3vl", "qwen38"):
            q38 = model == "qwen38"
            sp.append(
                (3 + q38, CAUSAL, 0)
            )  # <|im_start|>user\n (+ outer <|vision_start|> for qwen38)
            per_patch, ts = qwen_video(dur)
            for t in ts:
                sp.append(
                    (qwen_ntok(f"<{t:.1f} seconds>") + 1, CAUSAL, 0)
                )  # + <|vision_start|>
                sp.append((per_patch, CAUSAL, 1))
                sp.append((1, CAUSAL, 0))  # <|vision_end|>
            capq = max(1, round(cap * QWEN38_TEXT_RATIO["molmo2_cap"])) if q38 else cap
            # (+ outer <|vision_end|> for qwen38) prompt <|im_end|>\n<|im_start|>assistant\n caption <|im_end|>\n
            sp.append(
                (
                    q38 + qwen_ntok("Describe this video in detail.") + 5 + capq + 2,
                    CAUSAL,
                    0,
                )
            )
        elif model == "qwen25vl":
            # <|im_start|>user\n<|vision_start|> pads <|vision_end|> prompt <|im_end|>\n<|im_start|>assistant\n cap <|im_end|>\n
            sp.append((3 + 1, CAUSAL, 0))
            sp.append((qwen25_video_tokens(dur), CAUSAL, 1))
            sp.append(
                (
                    1 + qwen_ntok("Describe this video in detail.") + 5 + cap + 2,
                    CAUSAL,
                    0,
                )
            )
        else:
            sp.append((1 + 3, CAUSAL, 0))  # <bos><start_of_turn>user\n
            n = math.ceil(dur * GEMMA_VID_FPS)
            ts = (
                np.arange(n) / GEMMA_VID_FPS
                if n <= GEMMA_VID_MAX_FRAMES
                else np.linspace(0, dur, GEMMA_VID_MAX_FRAMES, endpoint=False)
            )
            for t in ts:
                sp.append((gemma_ntok(f"Frame at {t:.1f}s:"), CAUSAL, 0))
                sp += gemma_image_spans(VIDEO_HW[1], VIDEO_HW[0], pas=False)
            capg = max(1, round(cap * GEMMA_TEXT_RATIO["molmo2_cap"]))
            sp.append(
                (gemma_ntok("Describe this video in detail.") + 5 + capg + 2, CAUSAL, 0)
            )
        return sp
    # OBELICS interleaved web document: BOS ... EOS, original item order
    sp.append((1, CAUSAL, 0))
    for it in items:
        if it[0] == "t":
            if model in ("molmo2", "qwen3vl", "qwen25vl"):
                n = it[1]
            elif model == "qwen38":
                n = max(1, round(it[1] * QWEN38_TEXT_RATIO["obelics"]))
            else:
                n = max(1, round(it[1] * GEMMA_TEXT_RATIO["obelics"]))
            if n:
                sp.append((n, CAUSAL, 0))
        else:
            w, h = it[1], it[2]
            if model == "molmo2":
                sp.append((molmo_image_tokens(w, h), GROUP, 1))
            elif model in ("qwen3vl", "qwen38", "qwen25vl"):
                n = (
                    qwen25_image_tokens(w, h)
                    if model == "qwen25vl"
                    else qwen_image_tokens(w, h)
                )
                if n:
                    sp += [(1, CAUSAL, 0), (n, CAUSAL, 1), (1, CAUSAL, 0)]
            else:
                sp += gemma_image_spans(w, h, pas=(model == "gemma3ps"))
    sp.append((1, CAUSAL, 0))
    return sp


def merge_spans(spans):
    """Merge adjacent same-mode spans; never merge two 'full' spans."""
    lens, modes = [], []
    for n, m, _ in spans:
        if n <= 0:
            continue
        if modes and modes[-1] == m and m != FULL:
            lens[-1] += n
        else:
            lens.append(n)
            modes.append(m)
    return lens, modes


def doc_len(model, doc):
    return sum(s[0] for s in doc_spans(model, doc))


# ----------------------------------------------------------------------------- mask density
def doc_visible_pairs(lens, modes, window=None):
    """Visible (q, k) pairs inside one document, analytically.

    causal: earlier tokens + self; full: earlier + whole own split; group: earlier + every group token of
    the doc. With `window` (Gemma sliding layers): the causal part is limited to the last `window` keys
    (k > q - window), bidirectional extras are added on top (HF Gemma3 ORs the image block into the
    sliding mask).
    """
    g_total = sum(n for n, m in zip(lens, modes) if m == GROUP)
    g_before = 0
    off = 0
    tot = 0
    for n, m in zip(lens, modes):
        if window is None:
            causal = n * off + n * (n + 1) // 2  # sum_{p=off}^{off+n-1} (p + 1)
        else:
            causal = _sum_min(off, off + n, window)
        if m == CAUSAL:
            tot += causal
        elif m == FULL:
            tot += (
                causal + n * (n - 1) // 2
            )  # later tokens of own split (n <= window assumed)
        else:  # group: later tokens of own split + group tokens of later splits
            later_groups = g_total - g_before - n
            tot += causal + n * (n - 1) // 2 + n * later_groups
            assert window is None, (
                "sliding-window density is only defined here for models without group splits"
            )
            g_before += n
        off += n
    return tot


def _sum_min(a, b, w):
    """sum_{p=a}^{b-1} min(p + 1, w)."""

    def s(x):  # sum_{p=0}^{x-1} min(p+1, w)
        if x <= w:
            return x * (x + 1) // 2
        return w * (w + 1) // 2 + (x - w) * w

    return s(b) - s(a)


def layout_density(doc_lens, split_lens, modes, L, window=None):
    tot = 0
    i = 0
    for dl in doc_lens:
        acc, ls, ms = 0, [], []
        while acc < dl:
            ls.append(split_lens[i])
            ms.append(modes[i])
            acc += split_lens[i]
            i += 1
        assert acc == dl
        tot += doc_visible_pairs(ls, ms, window)
    return tot / (L * L)


# ----------------------------------------------------------------------------- data + packing
def load_docs(max_obelics=None):
    mol = []
    with open(os.path.join(R, "out", "molmo2_cap.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            items = dict((k, v) for k, v in d["items"])
            mol.append(("molmo2_cap", (items["v"], items["t"])))
    obe = []
    with open(os.path.join(R, "out", "obelics.jsonl")) as f:
        for line in f:
            if not line.endswith("\n"):
                break  # partial trailing line while extraction is still running
            obe.append(("obelics", json.loads(line)["items"]))
            if max_obelics and len(obe) >= max_obelics:
                break
    return mol, obe


class Stream:
    """Shuffled document stream for one mix; 'mixed' balances cumulative tokens 50/50 by source."""

    def __init__(self, mix, pools, lengths, L, seed):
        self.L = L
        self.lengths = lengths
        rng = np.random.default_rng(seed)
        self.srcs = (
            ["molmo2_cap"]
            if mix == "video"
            else ["obelics"]
            if mix == "interleaved"
            else ["molmo2_cap", "obelics"]
        )
        self.perm = {s: rng.permutation(len(pools[s])) for s in self.srcs}
        self.pos = {s: 0 for s in self.srcs}
        self.tok = {s: 0 for s in self.srcs}
        self.pulled = {s: 0 for s in self.srcs}
        self.dropped = {s: 0 for s in self.srcs}

    def next(self):
        while True:
            s = min(self.srcs, key=lambda x: self.tok[x])
            if self.pos[s] >= len(self.perm[s]):
                self.perm[s] = np.random.default_rng(self.pos[s]).permutation(
                    len(self.perm[s])
                )
                self.pos[s] = 0
            i = int(self.perm[s][self.pos[s]])
            self.pos[s] += 1
            self.pulled[s] += 1
            n = int(self.lengths[s][i])
            if n > self.L:
                self.dropped[s] += 1
                continue
            self.tok[s] += n
            return s, i, n


def pack_sequences(stream, L, nseq, lookahead=64):
    buf = deque()
    seqs = []
    for _ in range(nseq):
        cur, used = [], 0
        while used < L:
            while len(buf) < lookahead:
                buf.append(stream.next())
            rem = L - used
            hit = next((k for k, d in enumerate(buf) if d[2] <= rem), None)
            if hit is None:
                break
            d = buf[hit]
            del buf[hit]
            cur.append(d)
            used += d[2]
        seqs.append((cur, L - used))
    return seqs


def build_layout(model, pools, seq, pad):
    doc_lens, split_lens, modes = [], [], []
    vision = 0
    for s, i, n in seq:
        spans = doc_spans(model, pools[s][i])
        vision += sum(x[0] for x in spans if x[2])
        ls, ms = merge_spans(spans)
        assert sum(ls) == n
        doc_lens.append(n)
        split_lens += ls
        modes += ms
    if pad:
        doc_lens.append(pad)
        split_lens.append(pad)
        modes.append(CAUSAL)
    return {
        "document_lens": doc_lens,
        "split_lens": split_lens,
        "attn_modes": modes,
    }, vision


def pct(a, q):
    return float(np.percentile(a, q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--mixes", nargs="+", default=MIXES)
    ap.add_argument("--seq-lens", nargs="+", type=int, default=SEQ_LENS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument(
        "--stat-seqs",
        type=int,
        default=8,
        help="sequences packed per seed for padding stats",
    )
    ap.add_argument("--max-obelics", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(R, "layouts"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    mol, obe = load_docs(args.max_obelics)
    pools = {"molmo2_cap": mol, "obelics": obe}
    print(
        f"loaded {len(mol)} videos, {len(obe)} obelics docs in {time.time() - t0:.0f}s",
        flush=True,
    )

    raw = {
        "n_videos": len(mol),
        "n_obelics_docs": len(obe),
        "video_duration_s": {},
        "images_per_obelics_doc": {},
        "doc_tokens": {},
    }
    dur = np.array([d[1][0] for d in mol])
    raw["video_duration_s"] = {
        "p50": pct(dur, 50),
        "p90": pct(dur, 90),
        "p99": pct(dur, 99),
        "max": float(dur.max()),
        "mean": float(dur.mean()),
    }
    nimg = np.array([sum(1 for it in d[1] if it[0] == "i") for d in obe])
    raw["images_per_obelics_doc"] = {
        "mean": float(nimg.mean()),
        "p50": pct(nimg, 50),
        "p90": pct(nimg, 90),
        "p99": pct(nimg, 99),
        "max": int(nimg.max()),
        "frac_zero": float((nimg == 0).mean()),
    }
    txt = np.array([sum(it[1] for it in d[1] if it[0] == "t") for d in obe])
    raw["obelics_text_tokens_qwen3"] = {
        "p50": pct(txt, 50),
        "p90": pct(txt, 90),
        "p99": pct(txt, 99),
        "max": int(txt.max()),
    }

    summary_rows = []
    for model in args.models:
        t1 = time.time()
        lengths = {
            s: np.array([doc_len(model, d) for d in pools[s]], dtype=np.int64)
            for s in pools
        }
        raw["doc_tokens"][model] = {
            s: {
                "mean": float(a.mean()),
                "p50": pct(a, 50),
                "p90": pct(a, 90),
                "p99": pct(a, 99),
                "max": int(a.max()),
                **{
                    f"frac_gt_{L // 1024}k": float((a > L).mean()) for L in ALL_SEQ_LENS
                },
            }
            for s, a in lengths.items()
        }
        print(
            f"[{model}] doc lengths in {time.time() - t1:.0f}s: "
            + ", ".join(
                f"{s} p50={raw['doc_tokens'][model][s]['p50']:.0f}" for s in pools
            ),
            flush=True,
        )
        for mix in args.mixes:
            for L in args.seq_lens:
                layouts, ndocs, vis, nbid, splen, dens, dens_swa, pads, pads_all = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                src_docs = {"molmo2_cap": 0, "obelics": 0}
                pulled = {"molmo2_cap": 0, "obelics": 0}
                dropped = {"molmo2_cap": 0, "obelics": 0}
                for seed in args.seeds:
                    st = Stream(mix, pools, lengths, L, seed)
                    seqs = pack_sequences(st, L, args.stat_seqs)
                    for s in st.srcs:
                        pulled[s] += st.pulled[s]
                        dropped[s] += st.dropped[s]
                    pads_all += [p / L for _, p in seqs]
                    seq, pad = seqs[0]
                    lay, v = build_layout(model, pools, seq, pad)
                    assert sum(lay["split_lens"]) == sum(lay["document_lens"]) == L
                    layouts.append(lay)
                    for s, _, _ in seq:
                        src_docs[s] += 1
                    ndocs.append(len(seq))
                    vis.append(v / L)
                    nbid.append(sum(1 for m in lay["attn_modes"] if m != CAUSAL))
                    splen.append(L / len(lay["split_lens"]))
                    dens.append(
                        layout_density(
                            lay["document_lens"],
                            lay["split_lens"],
                            lay["attn_modes"],
                            L,
                        )
                    )
                    if model.startswith("gemma"):
                        dens_swa.append(
                            layout_density(
                                lay["document_lens"],
                                lay["split_lens"],
                                lay["attn_modes"],
                                L,
                                window=1024,
                            )
                        )
                    pads.append(pad / L)
                npulled = sum(pulled.values())
                stats = {
                    "docs_per_seq": {
                        "mean": float(np.mean(ndocs)),
                        "min": int(min(ndocs)),
                        "max": int(max(ndocs)),
                    },
                    "docs_by_source": src_docs,
                    "vision_token_frac": float(np.mean(vis)),
                    "bidir_splits_per_seq": float(np.mean(nbid)),
                    "mean_split_len": float(np.mean(splen)),
                    "padding_frac": float(np.mean(pads)),
                    f"padding_frac_{len(pads_all)}seq": float(np.mean(pads_all)),
                    "dropped_doc_frac": float(sum(dropped.values()) / max(npulled, 1)),
                    "dropped_docs": dropped,
                    "docs_pulled": pulled,
                    "mask_density": float(np.mean(dens)),
                    "mask_density_per_seq": dens,
                }
                if dens_swa:
                    stats["mask_density_sliding1024"] = float(np.mean(dens_swa))
                src = {
                    "video": "allenai/Molmo2-Cap",
                    "interleaved": "HuggingFaceM4/OBELICS",
                    "mixed": "allenai/Molmo2-Cap + HuggingFaceM4/OBELICS (50/50 by tokens)",
                }[mix]
                out = {
                    "model": model,
                    "mix": mix,
                    "seq_len": L,
                    "source": f"{src}; token rules in token_rules.md; generated by pack.py",
                    "layouts": layouts,
                    "stats": stats,
                }
                path = os.path.join(args.out, f"{model}_{mix}_{L // 1024}k.json")
                with open(path, "w") as f:
                    json.dump(out, f, separators=(",", ":"))
                summary_rows.append((model, mix, L, stats))
                print(
                    f"  {model} {mix} {L // 1024}k docs={stats['docs_per_seq']['mean']:.1f} vis={stats['vision_token_frac']:.3f} "
                    f"bidir={stats['bidir_splits_per_seq']:.0f} split={stats['mean_split_len']:.0f} pad={stats['padding_frac']:.4f} "
                    f"drop={stats['dropped_doc_frac']:.4f} dens={stats['mask_density']:.2e}",
                    flush=True,
                )
    raw_path = os.path.join(args.out, "raw_stats.json")
    if os.path.exists(raw_path):  # accumulate per-model doc stats across runs
        with open(raw_path) as f:
            prev = json.load(f)
        raw["doc_tokens"] = {**prev.get("doc_tokens", {}), **raw["doc_tokens"]}
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=1)
    write_summary(args.out, raw, load_rows(args.out))
    print(f"done in {time.time() - t0:.0f}s")


def load_rows(out):
    """Summary rows from every layout file in `out` (so per-model runs accumulate)."""
    rows = []
    for name in os.listdir(out):
        if name.endswith(".json") and name != "raw_stats.json":
            with open(os.path.join(out, name)) as f:
                d = json.load(f)
            rows.append((d["model"], d["mix"], d["seq_len"], d["stats"]))
    order = {m: i for i, m in enumerate(MODELS)}
    rows.sort(key=lambda r: (order.get(r[0], 99), MIXES.index(r[1]), r[2]))
    return rows


def write_summary(out, raw, rows):
    L_ = lambda L: f"{L // 1024}k"
    lines = [
        "# Packed long-context VLM layouts (real data statistics)",
        "",
        "Generated by `pack.py` from `out/molmo2_cap.jsonl` (allenai/Molmo2-Cap) and `out/obelics.jsonl` "
        "(HuggingFaceM4/OBELICS subset). Token rules: `token_rules.md`. Per (model, mix, L): 4 layouts "
        "(seeds 0-3, first packed sequence of each seed); padding is also reported over "
        "all sequences packed for statistics.",
        "",
        "Columns: docs/seq mean (min-max), excluding the padding doc; vis = fraction of tokens that are image/video "
        "tokens (Molmo2: whole group spans incl. image markers/col tokens; Qwen3-VL: <|image_pad|>/<|video_pad|>; "
        "Gemma 3: the 256 soft tokens per image); bidir = bidirectional ('full' or 'group') splits per sequence; "
        "split = mean split length; pad = padding fraction (4 output layouts / all packed stat sequences); "
        "drop = fraction of docs pulled from the stream that were longer than L; density = visible (q,k) pairs / L^2 "
        "(same doc only; padding counted as a causal doc); Gemma rows also give density on the sliding-window-1024 "
        "layers (5 of 6 layers).",
        "",
        "`qwen38` = Qwen/Qwen3.8-27B (Aug 2026). Its preprocessor_config.json / video_preprocessor_config.json are "
        "byte-identical to Qwen3-VL-8B's and it uses the same Qwen3VLProcessor, so image/video/timestamp tokenization "
        "follows the `qwen3vl` rules; only text counts differ (Qwen3.8 tokenizer, +1.0% on OBELICS text, +0.02% on "
        "captions) plus 2 outer vision markers per video. Run at L = 256k-2M (native context 262,144; 1M via YaRN; "
        "2M is layouts only). All attention is causal; note 48 of its 64 layers are linear attention (Gated DeltaNet), "
        "so the mask density applies to the 16 full-attention layers.",
        "",
        "| model | mix | L | docs/seq | vis | bidir/seq | split | pad | pad(all) | drop | density | swa1024 density |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model, mix, L, s in rows:
        pa = next(
            v
            for k, v in s.items()
            if k.startswith("padding_frac_") and k.endswith("seq")
        )
        d = s["docs_per_seq"]
        swa = (
            f"{s['mask_density_sliding1024']:.2e}"
            if "mask_density_sliding1024" in s
            else "-"
        )
        lines.append(
            f"| {model} | {mix} | {L_(L)} | {d['mean']:.1f} ({d['min']}-{d['max']}) | {s['vision_token_frac']:.3f} | "
            f"{s['bidir_splits_per_seq']:.0f} | {s['mean_split_len']:.0f} | {s['padding_frac']:.4f} | {pa:.4f} | "
            f"{s['dropped_doc_frac']:.4f} | {s['mask_density']:.2e} | {swa} |"
        )
    lines += [
        "",
        "## Raw data distributions",
        "",
        f"- Videos (Molmo2-Cap): {raw['n_videos']:,}; duration s p50/p90/p99/max = "
        + "/".join(
            f"{raw['video_duration_s'][k]:.1f}" for k in ["p50", "p90", "p99", "max"]
        )
        + f" (mean {raw['video_duration_s']['mean']:.1f})",
        f"- OBELICS docs: {raw['n_obelics_docs']:,}; images/doc mean {raw['images_per_obelics_doc']['mean']:.2f}, "
        f"p50/p90/p99/max = "
        + "/".join(
            str(raw["images_per_obelics_doc"][k]) for k in ["p50", "p90", "p99", "max"]
        )
        + f", docs with no image {raw['images_per_obelics_doc']['frac_zero']:.3f}",
        "- OBELICS text tokens/doc (Qwen3 tokenizer) p50/p90/p99/max = "
        + "/".join(
            str(raw["obelics_text_tokens_qwen3"][k])
            for k in ["p50", "p90", "p99", "max"]
        ),
        "",
        "Document token lengths per model:",
        "",
        "| model | source | mean | p50 | p90 | p99 | max | >128k | >256k | >512k | >1M | >2M |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model, per in raw["doc_tokens"].items():
        for src, a in per.items():
            lines.append(
                f"| {model} | {src} | {a['mean']:.0f} | {a['p50']:.0f} | {a['p90']:.0f} | {a['p99']:.0f} | {a['max']} | "
                + " | ".join(
                    f"{a[f'frac_gt_{L // 1024}k']:.2e}"
                    if f"frac_gt_{L // 1024}k" in a
                    else "-"
                    for L in ALL_SEQ_LENS
                )
                + " |"
            )
    with open(os.path.join(out, "SUMMARY.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
