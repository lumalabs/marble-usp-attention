# One node: ours vs FlashMask, same session (8 H100s, 2026-10-18)

Every number comes from one 8-GPU node in one session:
FlashMask first (`s8_fm.jsonl`: FlashMask-4 CP with IPO, FM + Ulysses U = 2, 4,
8, FlashMask-3 CP, FlashMask-3 + Ulysses-8 and FlashMask-3's overlapped
forward through PaddlePaddle 3.4), then ours (`s8_ours.jsonl`: R x U over
{1x8, 2x4, 4x2, 8x1}, ring and sparse all-to-all). Forward + backward
attention, median of 10 iterations after 3 warmups; ms.

Ours ran commit a3c88f759a4 (separable masks). The best configuration is
R1U8 on every layout but one, and its column-range path is unchanged in
7b9e20d9f31; that commit speeds up the ring and sparse all-to-all rows
(see `../kernels/`), which matters only for `synth_video_group_512k` here.

| layout | ours (config) | FM CP | FM + U (U) | FM3 CP | FM3 + U (U) | FM3 overlap fwd | best FlashMask / ours |
|---|---|---:|---|---:|---|---:|---:|
| molmo2_video_512k | 135.0 R1U8 | 157.5 | 132.8 U8 | 216.8 | 133.0 U8 | 208.2 | 0.984x |
| molmo2_mixed_512k | 88.6 R1U8 | 113.0 | 88.0 U8 | 165.5 | 89.4 U8 | 153.8 | 0.993x |
| molmo2_interleaved_512k | 48.5 R1U8 | 75.8 | 49.1 U8 | 121.8 | 50.9 U8 | 110.3 | 1.012x |
| molmo2_video_1024k | 261.5 R1U8 | 308.0 | 258.2 U8 | 376.0 | 268.9 U8 | 351.5 | 0.987x |
| molmo2_mixed_1024k | 182.1 R1U8 | 230.5 | 180.0 U8 | 325.0 | 193.2 U8 | 302.5 | 0.988x |
| synth_vlm_group_512k | 568.8 R1U8 | 614.8 | 553.8 U4 | - | 545.0 U8 | - | 0.958x |
| synth_video_group_512k | 2672.8 R8U1 gather | 2586.9 | 2588.5 U2 | - | 2681.2 U8 | - | 0.968x |

On one node every method reduces to Ulysses around a masked kernel, and
FlashMask leads by 0.7-4.2% on six of seven layouts (we lead on interleaved
by 1.2%). The remaining gap is full-tile kernel throughput: the profiles in
`../kernels/` put FlashMask-4's backward at 82.2 ms against FA4's 84.1 ms on
Molmo2 video 512k with an always-true mask. FlashMask-3 CP (all-gather) and
its overlapped forward are 1.4-2.6x slower than FM + Ulysses; FlashMask-3 +
Ulysses-8 (one rank per CP group, a plain FlashMask-3 call) is within 0-7% of
FlashMask-4 + Ulysses. FlashMask-3 CP and overlap fail on the synthetic
32-KV-head layouts (Paddle: "Failed to initialize the TMA descriptor").
