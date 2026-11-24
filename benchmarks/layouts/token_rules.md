# Image/video → LM token rules (as implemented in `pack.py`)

Sources were downloaded on 2026-10-15 into `src/` (HF model repos at `main`, transformers at `main` plus
`v4.57.1` for Qwen3-VL's placeholder replacement). Text token counts in the inputs are Qwen3-tokenizer counts;
Molmo2-8B and Qwen3-VL use the Qwen3 vocabulary. For Gemma 3, text counts are scaled by the measured
Gemma3/Qwen3 ratio (1.0065 on 8,277 OBELICS texts, 0.9938 on 4,000 Molmo2-Cap captions; `work/tokratio.py`).
Marker and timestamp strings are tokenized exactly with the respective tokenizer.

## (a) allenai/Molmo2-8B

Sources:
- https://huggingface.co/allenai/Molmo2-8B/blob/main/preprocessor_config.json (max_crops 8, overlap_margins [4,4], patch 14, pooling 2x2, size 378)
- https://huggingface.co/allenai/Molmo2-8B/blob/main/video_preprocessor_config.json (num_frames 384, max_fps 2, frame_sample_mode uniform_last_frame, pooling 3x3, size 378)
- https://huggingface.co/allenai/Molmo2-8B/blob/main/processor_config.json (image_use_col_tokens true, use_single_crop_col_tokens unset → follows image_use_col_tokens, use_single_crop_start_token true, video_use_col_tokens false, use_frame_special_tokens false)
- https://huggingface.co/allenai/Molmo2-8B/blob/main/image_processing_molmo2.py (`select_tiling`, `build_overlapping_crops`, `arange_for_pooling`)
- https://huggingface.co/allenai/Molmo2-8B/blob/main/video_processing_molmo2.py (`sample_times`, `uniform_last_frame`)
- https://huggingface.co/allenai/Molmo2-8B/blob/main/processing_molmo2.py (`get_image_tokens`, `get_video_string`, `insert_bos`, token_type_ids = all IMAGE_TOKENS)
- https://huggingface.co/allenai/Molmo2-8B/blob/main/modeling_molmo2.py (`token_type_ids_mask_function`: every image-token q/k pair is visible, OR'ed onto the causal mask)
- Paper: https://arxiv.org/abs/2601.10611 (2 fps, up to 128 frames in standard training, 384 frames in the long-context stage)

Image W×H:
- Global crop: resize to 378×378 → 27×27 patches → 2×2 pool → 14×14. Tokens: `<low_res_im_start>` + 14 rows × (14 `<im_patch>` + `<im_col>`) + `<im_end>` = **212**.
- High-res: `select_tiling(H−112, W−112, 266, max_crops=8)` → (rows, cols), with rows·cols ≤ 8, chosen to minimise upscaling, or to minimise downscaling if every tiling needs it. The image is resized to (rows·266+112, cols·266+112), giving (rows·19+8)×(cols·19+8) patches without overlap, then a 2×2 pool gives ph=⌈(rows·19+8)/2⌉ and pw=⌈(cols·19+8)/2⌉. Tokens: `<im_start>` + ph × (pw `<im_patch>` + `<im_col>`) + `<im_end>`.
- Examples: 300×200 → 424; 626×626 → 766; 1024×683 and 1920×1080 → 1203 (2×4 tiling).
- All of these tokens (patch, col, start, end, low-res start) are image tokens. They form one contiguous **group** span per image.

Video of D seconds:
- Frames are 378×378 → 27×27 patches → 3×3 pool → 9×9 = 81 tokens. Each frame is `<im_start>` + 81 `<im_patch>` + `<im_end>` = **83 group tokens**, with no column tokens and no frame special tokens.
- Sampling (uniform_last_frame, max_fps 2, num_frames 384): if D ≤ 383/2 = 191.5 s, the timestamps are `arange(0, D, 0.5)` plus D, so ⌈2D⌉+1 frames. Otherwise 384 frames from `linspace(0, D, 384)`.
- Before each frame the processor inserts a causal timestamp `f"{t:.1f} "`, with a leading space from the second frame on (4–7 tokens).
- All frames of a video belong to one group, and so do all images of a document. The HF mask makes every image token visible to every other image token in the row; in the packed layout this is scoped to the document.
- BOS is inserted at the start. The chat template puts `<|video|>` / `<|image|>` before `<|im_start|>user\n…`.
- Long-video setting: the HF processor default is 384 frames, which matches the paper's long-context stage (config `max_position_embeddings` = 36864). Standard SFT used 128 frames. **pack.py uses 384.**

## (b) Qwen/Qwen3-VL-8B-Instruct

Sources:
- https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/preprocessor_config.json (patch 16, merge 2, temporal_patch 2, images: shortest_edge 65536 / longest_edge 16777216 pixels)
- https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/video_preprocessor_config.json (videos: 4096 / 25165824 total pixels)
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py (`smart_resize`)
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/video_processing_qwen3_vl.py (fps 2, min_frames 4, max_frames 768, `cap_pixels_per_frame`, max_video_tokens 768)
- https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/processing_qwen3_vl.py (per-temporal-patch `<{t:.1f} seconds>` + `<|vision_start|>` + pads + `<|vision_end|>`; this replaces the template's outer `<|vision_start|><|video_pad|><|vision_end|>`)
- https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py (reference: VIDEO_MIN/MAX_TOKEN_NUM 128/768, FPS 2, FPS_MIN/MAX_FRAMES 4/768, MODEL_SEQ_LEN 128000)

Image W×H:
- `smart_resize(H, W, factor=32, min_pixels=65536, max_pixels=16777216)`: each side is rounded to a multiple of 32, then scaled down (floor) or up (ceil) to fit the pixel budget.
- Tokens: (H'/32)·(W'/32), between 64 and 16384, as `<|image_pad|>`, wrapped in `<|vision_start|>` … `<|vision_end|>`. Examples: 1920×1080 → 2040; 1024×683 → 672.
- Attention is causal everywhere, including image tokens.

Video of D seconds:
- Frames: n = clamp(2D, 4, 768), floored to an even number, sampled by `linspace` over the source frames.
- Per frame pair (temporal patch 2): `<{t:.1f} seconds>` (the mean time of the pair, 6–8 tokens) + `<|vision_start|>` + P × `<|video_pad|>` + `<|vision_end|>`.
- P follows the reference qwen-vl-utils, which is the same as transformers `cap_pixels_per_frame=True`, the stated default from v5.22. The frame is resized with `smart_resize(factor=32, min=128·32², max=max(min(768·32², 0.9·128000·32²·2/n), 1.05·128·32²))`.
- For a 1280×720 source, P = 720 while n ≤ 300 (D ≤ 150 s), then shrinks (D=300 s → 364). The total stays ≤ 115,200 tokens per video. Examples: 30 s → 21,600 video tokens; 120 s → 86,400.
- Alternative: the uncapped HF v4.57 default spends `longest_edge` = 25,165,824 pixels on the whole clip, i.e. ≤ 12,288 tokens per video. That would make Qwen video docs about 4–7× shorter.

## (c) google/gemma-3-27b-it

Sources:
- https://huggingface.co/google/gemma-3-27b-it (gated). The identical public mirror https://huggingface.co/unsloth/gemma-3-27b-it supplied preprocessor_config.json (size 896×896, image_seq_length 256, do_pan_and_scan null), processor_config.json, config.json (mm_tokens_per_image 256, sliding_window 1024, sliding_window_pattern 6) and tokenizer.json.
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/processing_gemma3.py (`full_image_sequence = "\n\n<start_of_image>" + 256×<image_soft_token> + "<end_of_image>\n\n"`; pan&scan defaults: off, min crop 256, max 4 crops, min ratio 1.2; P&S prompt text)
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/image_processing_gemma3.py (`pan_and_scan_batched`)
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/configuration_gemma3.py (layer i is sliding unless (i+1) % 6 == 0, so 5 local : 1 global)

Image (any W×H):
- The image is resized to 896×896 and produces 256 soft tokens: `\n\n<start_of_image>` (2 causal) + **256 full** + `<end_of_image>\n\n` (2 causal) = 260 tokens. Each image is its own bidirectional block.
- Pan&scan (`gemma3ps` variant, OBELICS images only) applies when the aspect ratio is ≥ 1.2. It takes k = clamp(round(ratio), 2, 4) crops along the long side, limited by crop size ≥ 256.
- Text with P&S: "Here is the original image " + block + " and here are some crops to help you see better " + k blocks joined by " ". Each crop is its own full block.
- Sliding window 1024 on 5 of 6 layers. Bidirectional image blocks are OR'ed into the sliding mask too, as in HF Gemma3; `mask_density_sliding1024` reports that.

Video: Gemma 3 has **no video processor**. ASSUMPTION used here:
- frames at 1 fps, up to 128 uniformly spaced frames;
- each frame is a 260-token image block preceded by `Frame at {t:.1f}s:` (≈9 tokens);
- no pan&scan on frames.

## (d) Qwen/Qwen3.8-27B (released Aug 2026; layouts named `qwen38_*`)

Sources (downloaded 2026-10-15 into `src/qwen38/`):
- https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/preprocessor_config.json and https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/video_preprocessor_config.json: **byte-identical** to Qwen3-VL-8B's. Patch 16, merge 2, temporal patch 2; images 65,536–16,777,216 px; video total 4,096–25,165,824 px; `processor_class` Qwen3VLProcessor; video processor Qwen3VLVideoProcessor.
- https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json: `model_type` qwen3_5, vision patch 16 / spatial_merge 2 / temporal 2, vocab 248,320, max_position_embeddings 262,144. 48 of the 64 layers are `linear_attention`; layers 3, 7, …, 63 are `full_attention`. transformers 5.8.0.dev0.
- https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/chat_template.jinja: the same `<|vision_start|><|video_pad|><|vision_end|>` and `<|image_pad|>` markup as Qwen3-VL.
- https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/README.md:
  - default video sampling is `fps=2`, `do_sample_frames=True`;
  - native context is 262,144, extensible to about 1M with YaRN (factor 4);
  - for long video it recommends raising `longest_edge` to 469,762,048, i.e. about 224k video tokens.
- https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_5: there is no model-specific processor, so the Qwen3-VL processor code applies. https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py expands only the `<|video_pad|>` token. So on transformers ≥5 the template's outer `<|vision_start|>`/`<|vision_end|>` around a video is kept, unlike v4.57.1 where it was replaced.

Rule:
- Vision tokenization is **identical to Qwen3-VL**: images, video frames, 2 fps, 4–768 frames, the `<t seconds>` timestamps and the per-frame markers all follow §(b). pack.py uses the same qwen-vl-utils reference budget: 720 tokens per frame pair at 720p, at most ~115k tokens per video.
- Differences:
  - The Qwen3.8 tokenizer has a 248k vocabulary. Text counts are scaled by the measured Qwen3.8/Qwen3 ratio (1.0103 on OBELICS text, 1.0002 on captions; `work/tokratio38.py`). Chat markers and timestamp strings tokenize to the same counts as with Qwen3.
  - There are 2 extra outer vision markers per video.
- Attention is causal everywhere. The mask density applies to the 16 full-attention layers; the linear-attention layers only need document boundaries.
- Not applied: the README's long-video `longest_edge` of 469,762,048. With the current uncapped processor this would give 720p frames at native 704×1280, i.e. 880 tokens per frame pair (a 180 s video would be about 158k tokens). With the shipped config, a whole video is at most 12,288 tokens.

## Document templates used

- Molmo2-Cap docs:
  - Molmo2: BOS, video, `<|im_start|>user\nDescribe this video in detail.<|im_end|>\n<|im_start|>assistant\n` + merged caption + 1.
  - Qwen3-VL: `<|im_start|>user\n` + video + prompt + `<|im_end|>\n<|im_start|>assistant\n` + caption + `<|im_end|>\n`.
  - Gemma: `<bos><start_of_turn>user\n` + frames + prompt + `<end_of_turn>\n<start_of_turn>model\n` + caption + `<end_of_turn>\n`.
- OBELICS docs: BOS + items in original order (text causal; images as above) + EOS, with no chat template. Molmo2's chat template would hoist images to the front; here they stay interleaved.

## Assumptions / UNVERIFIED

- Molmo2-Cap source resolution is assumed to be 1280×720 at 30 fps; the inputs carry durations only. The resolution only affects Qwen3-VL: any source ≥ 720p (16:9) gives P = 720, while 360p would give 220.
- The Molmo2 rules were ported from the processor code and checked by hand on examples. They were not executed against the official processor, because downloaded remote code was not run.
- Molmo2 training may use fewer crops for multi-image inputs. Not verified; pack.py uses max_crops=8 for every image.
- The Gemma 3 video treatment is an assumption.
- OBELICS image sizes are `original_width/height`, falling back to the rendered size when missing.
- Qwen3.8: the outer-wrapper behaviour (+2 tokens per video) was read from transformers main, not executed. The Qwen3.8 rules were read from the configs; the processor was not run.
- `pack.py` also contains Qwen2.5-VL-7B/32B rules (`qwen25vl`: factor 28, images 3,136–12,845,056 px, qwen-vl-utils 0.0.10 video budget, one vision block per video with no timestamps). They were superseded by Qwen3.8 before any run; no layouts were generated.
