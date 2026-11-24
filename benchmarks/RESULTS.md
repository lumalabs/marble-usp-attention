# usp-attention benchmark results

## Accuracy vs fp32 (mean relative error)

| layout | seq | ring x ulysses | PTRR | out err online | flash | python | single-GPU FA4 | dq / dk / dv err (online) |
|---|---|---|---|---|---|---|---|---|
| causal | 32k | 1 x 8 | yes | 2.19e-03 | 2.19e-03 | 2.19e-03 | 2.19e-03 | 1.69e-03 / 1.66e-03 / 1.66e-03 |
| causal | 32k | 2 x 4 | yes | 2.19e-03 | 2.64e-03 | 3.12e-03 | 2.19e-03 | 2.39e-03 / 2.39e-03 / 2.40e-03 |
| causal | 32k | 4 x 2 | yes | 2.19e-03 | 3.10e-03 | 2.96e-03 | 2.19e-03 | 2.34e-03 / 2.34e-03 / 2.34e-03 |
| causal | 32k | 8 x 1 | yes | 2.19e-03 | 3.68e-03 | 2.86e-03 | 2.19e-03 | 2.34e-03 / 2.33e-03 / 2.33e-03 |
| docs | 32k | 1 x 8 | yes | 2.11e-03 | 2.11e-03 | 2.11e-03 | 2.11e-03 | 1.69e-03 / 1.67e-03 / 1.65e-03 |
| docs | 32k | 2 x 4 | yes | 2.11e-03 | 2.30e-03 | 2.63e-03 | 2.11e-03 | 2.01e-03 / 2.22e-03 / 2.21e-03 |
| docs | 32k | 4 x 2 | yes | 2.11e-03 | 2.53e-03 | 2.63e-03 | 2.11e-03 | 2.16e-03 / 2.28e-03 / 2.27e-03 |
| docs | 32k | 8 x 1 | yes | 2.11e-03 | 2.89e-03 | 2.66e-03 | 2.11e-03 | 2.30e-03 / 2.32e-03 / 2.31e-03 |
| full | 32k | 1 x 8 | yes | 2.26e-03 | 2.26e-03 | 2.26e-03 | 2.26e-03 | 1.69e-03 / 1.66e-03 / 1.66e-03 |
| full | 32k | 2 x 4 | yes | 2.26e-03 | 2.74e-03 | 3.22e-03 | 2.26e-03 | 2.43e-03 / 2.40e-03 / 2.41e-03 |
| full | 32k | 4 x 2 | yes | 2.26e-03 | 3.23e-03 | 3.02e-03 | 2.26e-03 | 2.38e-03 / 2.34e-03 / 2.36e-03 |
| full | 32k | 8 x 1 | yes | 2.26e-03 | 3.91e-03 | 2.90e-03 | 2.26e-03 | 2.37e-03 / 2.34e-03 / 2.34e-03 |
| mixed | 32k | 1 x 8 | yes | 2.14e-03 | 2.14e-03 | 2.14e-03 | 2.14e-03 | 1.69e-03 / 1.67e-03 / 1.65e-03 |
| mixed | 32k | 2 x 4 | yes | 2.14e-03 | 2.48e-03 | 2.93e-03 | 2.14e-03 | 2.24e-03 / 2.38e-03 / 2.36e-03 |
| mixed | 32k | 4 x 2 | yes | 2.14e-03 | 2.88e-03 | 2.86e-03 | 2.14e-03 | 2.31e-03 / 2.33e-03 / 2.33e-03 |
| mixed | 32k | 8 x 1 | yes | 2.14e-03 | 3.34e-03 | 2.72e-03 | 2.14e-03 | 2.30e-03 / 2.33e-03 / 2.32e-03 |
| mixed | 64k | 2 x 8 | yes | 2.16e-03 | 2.51e-03 | - | 2.16e-03 | 2.24e-03 / 2.38e-03 / 2.38e-03 |
| tiny_kv | 32k | 2 x 4 | no | 2.17e-03 | 2.37e-03 | 2.38e-03 | 2.11e-03 | 1.99e-03 / 1.67e-03 / 1.66e-03 |
| tiny_kv | 32k | 4 x 2 | no | 2.14e-03 | 2.44e-03 | 2.46e-03 | 2.11e-03 | 2.15e-03 / 1.67e-03 / 1.66e-03 |
| tiny_kv | 32k | 8 x 1 | no | 2.11e-03 | 2.46e-03 | 2.49e-03 | 2.11e-03 | 2.23e-03 / 1.68e-03 / 1.67e-03 |

## Deviation from single-GPU FA4 (mean relative error, longer sequences)

The single-GPU column is empty here by construction.

| layout | seq | ring x ulysses | PTRR | out err online | flash | python | single-GPU FA4 | dq / dk / dv err (online) |
|---|---|---|---|---|---|---|---|---|
| causal | 32k | 1 x 8 | yes | 0.00e+00 | 0.00e+00 | 0.00e+00 | 2.19e-03 | 7.74e-08 / 0.00e+00 / 0.00e+00 |
| causal | 32k | 2 x 4 | yes | 1.22e-03 | 2.07e-03 | 2.60e-03 | 2.19e-03 | 1.65e-03 / 1.70e-03 / 1.65e-03 |
| causal | 32k | 4 x 2 | yes | 1.41e-03 | 2.79e-03 | 2.60e-03 | 2.19e-03 | 1.63e-03 / 1.67e-03 / 1.62e-03 |
| causal | 32k | 8 x 1 | yes | 1.42e-03 | 3.48e-03 | 2.59e-03 | 2.19e-03 | 1.63e-03 / 1.67e-03 / 1.63e-03 |
| causal | 128k | 8 x 1 | yes | 1.63e-03 | 3.69e-03 | 2.70e-03 | - | 1.66e-03 / 1.67e-03 / 1.64e-03 |
| docs | 32k | 1 x 8 | yes | 0.00e+00 | 0.00e+00 | 0.00e+00 | 2.11e-03 | 1.42e-08 / 0.00e+00 / 0.00e+00 |
| docs | 32k | 2 x 4 | yes | 3.36e-04 | 8.22e-04 | 1.19e-03 | 2.11e-03 | 7.53e-04 / 1.32e-03 / 1.27e-03 |
| docs | 32k | 4 x 2 | yes | 5.89e-04 | 1.45e-03 | 1.56e-03 | 2.11e-03 | 1.15e-03 / 1.51e-03 / 1.44e-03 |
| docs | 32k | 8 x 1 | yes | 9.87e-04 | 2.28e-03 | 2.01e-03 | 2.11e-03 | 1.52e-03 / 1.66e-03 / 1.58e-03 |
| docs | 128k | 2 x 4 | yes | 5.39e-04 | 1.07e-03 | 1.53e-03 | - | 9.17e-04 / 1.48e-03 / 1.45e-03 |
| docs | 128k | 8 x 1 | yes | 1.29e-03 | 2.78e-03 | 2.36e-03 | - | 1.61e-03 / 1.66e-03 / 1.59e-03 |
| full | 32k | 1 x 8 | yes | 0.00e+00 | 0.00e+00 | 0.00e+00 | 2.26e-03 | 2.33e-07 / 0.00e+00 / 0.00e+00 |
| full | 32k | 2 x 4 | yes | 1.45e-03 | 2.29e-03 | 2.81e-03 | 2.26e-03 | 1.71e-03 / 1.68e-03 / 1.66e-03 |
| full | 32k | 4 x 2 | yes | 1.72e-03 | 3.09e-03 | 2.82e-03 | 2.26e-03 | 1.69e-03 / 1.66e-03 / 1.64e-03 |
| full | 32k | 8 x 1 | yes | 1.82e-03 | 3.89e-03 | 2.81e-03 | 2.26e-03 | 1.69e-03 / 1.66e-03 / 1.64e-03 |
| full | 128k | 8 x 1 | yes | 1.84e-03 | 3.90e-03 | 2.83e-03 | - | 1.68e-03 / 1.65e-03 / 1.65e-03 |
| mixed | 32k | 1 x 8 | yes | 0.00e+00 | 0.00e+00 | 0.00e+00 | 2.14e-03 | 7.88e-08 / 0.00e+00 / 0.00e+00 |
| mixed | 32k | 2 x 4 | yes | 8.67e-04 | 1.53e-03 | 2.03e-03 | 2.14e-03 | 1.30e-03 / 1.67e-03 / 1.61e-03 |
| mixed | 32k | 4 x 2 | yes | 1.16e-03 | 2.35e-03 | 2.30e-03 | 2.14e-03 | 1.53e-03 / 1.67e-03 / 1.60e-03 |
| mixed | 32k | 8 x 1 | yes | 1.21e-03 | 2.93e-03 | 2.23e-03 | 2.14e-03 | 1.54e-03 / 1.67e-03 / 1.61e-03 |
| mixed | 64k | 2 x 8 | yes | 9.36e-04 | 1.59e-03 | - | 2.16e-03 | 1.31e-03 / 1.66e-03 / 1.62e-03 |
| mixed | 128k | 2 x 4 | yes | 9.49e-04 | 1.61e-03 | 2.10e-03 | - | 1.30e-03 / 1.65e-03 / 1.61e-03 |
| mixed | 128k | 8 x 1 | yes | 1.49e-03 | 3.32e-03 | 2.51e-03 | - | 1.63e-03 / 1.67e-03 / 1.62e-03 |
| tiny_kv | 32k | 2 x 4 | no | 5.26e-05 | 7.11e-04 | 7.23e-04 | 2.11e-03 | 7.23e-04 / 9.25e-05 / 1.67e-05 |
| tiny_kv | 32k | 4 x 2 | no | 9.35e-05 | 1.07e-03 | 1.10e-03 | 2.11e-03 | 1.09e-03 / 1.67e-04 / 3.38e-05 |
| tiny_kv | 32k | 8 x 1 | no | 1.25e-04 | 1.25e-03 | 1.29e-03 | 2.11e-03 | 1.29e-03 / 2.31e-04 / 5.63e-05 |

## fp32 gradient accumulation across ring steps

Mixed layout, 32k tokens, 8 GPUs, mean relative error of the gradients vs fp32. Rows with `kv=4` use GQA (16 query / 4 KV heads); `kv=16` is MHA, where dk/dv stay per-step bf16.

| kv heads | ring x ulysses | fp32 grads | dq | dk | dv |
|---|---|---|---|---|---|
| 4 | 8 x 1 | True | 1.69e-03 | 1.67e-03 | 1.66e-03 |
| 4 | 2 x 4 | True | 1.69e-03 | 1.67e-03 | 1.66e-03 |
| 16 | 8 x 1 | True | 1.69e-03 | 2.33e-03 | 2.32e-03 |
| 16 | 2 x 4 | True | 1.69e-03 | 2.38e-03 | 2.36e-03 |
| 4 | 8 x 1 | False | 2.30e-03 | 2.34e-03 | 2.34e-03 |
| 4 | 2 x 4 | False | 2.24e-03 | 2.38e-03 | 2.38e-03 |
| 16 | 8 x 1 | False | 2.30e-03 | 2.33e-03 | 2.32e-03 |
| 16 | 2 x 4 | False | 2.24e-03 | 2.38e-03 | 2.36e-03 |
| single-GPU FA4 | - | - | 1.69e-03 | 1.67e-03 | 1.66e-03 |

| 256k step, 8 GPUs | bf16 grads ms | fp32 grads ms |
|---|---|---|
| llama_32q8kv_docs_256k_r8u1 | 138.7 | 133.5 |
| llama_32q8kv_docs_256k_r2u4 | 108.2 | 105.7 |
| mha_32q_mixed_256k_r8u1 | 224.2 | 220.5 |
| mha_32q_mixed_256k_r2u4 | 193.1 | 191.3 |

## Ablation on 8 GPUs (32 MHA heads)

### causal_128k (world 8, heads 32/32, mask density 0.500)

| config | median ms | p10 / p90 | TFLOP/s/GPU | vs naive ring | ring imbalance (contiguous) | peak GiB |
|---|---|---|---|---|---|---|
| ulysses8 | 106.7 | 103.4 / 116.4 | 577 | - | - | 1.9 |
| usp_r2u4_online_ptrr | 115.7 | 113.6 / 120.4 | 532 | - | 1.000 (1.500) | 4.6 |
| usp_r2u4_python_ptrr | 115.9 | 114.4 / 120.8 | 531 | - | 1.000 (1.500) | 4.6 |
| usp_r2u4_flash_ptrr | 116.3 | 114.7 / 121.3 | 529 | - | 1.000 (1.500) | 4.6 |
| usp_r4u2_online_ptrr | 122.9 | 120.5 / 129.2 | 501 | - | 1.000 (1.750) | 5.0 |

### causal_512k (world 8, heads 32/32, mask density 0.500)

| config | median ms | p10 / p90 | TFLOP/s/GPU | vs naive ring | ring imbalance (contiguous) | peak GiB |
|---|---|---|---|---|---|---|
| ulysses8 | 1636.5 | 1633.2 / 1639.7 | 602 | - | - | 7.5 |
| usp_r2u4_online_ptrr | 1763.3 | 1745.4 / 1783.8 | 559 | - | 1.000 (1.500) | 18.7 |
| usp_r2u4_flash_ptrr | 1767.4 | 1743.7 / 1784.0 | 557 | - | 1.000 (1.500) | 18.7 |
| usp_r2u4_python_ptrr | 1768.8 | 1742.4 / 1776.6 | 557 | - | 1.000 (1.500) | 18.7 |
| usp_r4u2_online_ptrr | 1786.7 | 1775.1 / 1804.4 | 551 | - | 1.000 (1.750) | 20.1 |

### docs_128k (world 8, heads 32/32, mask density 0.101)

| config | median ms | p10 / p90 | TFLOP/s/GPU | vs naive ring | ring imbalance (contiguous) | peak GiB |
|---|---|---|---|---|---|---|
| ulysses8 | 29.4 | 29.1 / 31.1 | 425 | 3.28x | - | 1.9 |
| usp_r2u4_online_ptrr | 34.8 | 34.6 / 35.4 | 359 | 2.77x | 1.001 (1.276) | 4.6 |
| usp_r2u4_python_ptrr | 35.5 | 35.4 / 36.1 | 352 | 2.71x | 1.001 (1.276) | 4.6 |
| usp_r2u4_flash_ptrr | 35.9 | 35.7 / 36.3 | 348 | 2.68x | 1.001 (1.276) | 4.6 |
| usp_r4u2_online_ptrr | 44.5 | 43.9 / 44.9 | 280 | 2.16x | 1.005 (1.940) | 5.0 |
| ring8_online_ptrr | 63.5 | 63.0 / 64.3 | 196 | 1.52x | 1.006 (2.557) | 3.9 |
| ring8_online_contig | 94.9 | 93.3 / 97.7 | 131 | 1.01x | 2.557 (2.557) | 3.9 |
| ring8_python_contig | 96.3 | 95.0 / 98.9 | 130 | 1.00x | 2.557 (2.557) | 3.9 |
| ring8_flash_contig | 98.3 | 97.4 / 99.3 | 127 | 0.98x | 2.557 (2.557) | 3.9 |

### docs_512k (world 8, heads 32/32, mask density 0.101)

| config | median ms | p10 / p90 | TFLOP/s/GPU | vs naive ring | ring imbalance (contiguous) | peak GiB |
|---|---|---|---|---|---|---|
| ulysses8 | 374.4 | 369.0 / 378.8 | 532 | 2.72x | - | 7.8 |
| usp_r2u4_online_ptrr | 394.7 | 391.6 / 398.8 | 505 | 2.58x | 1.001 (1.278) | 18.7 |
| usp_r2u4_python_ptrr | 397.6 | 394.1 / 402.4 | 501 | 2.56x | 1.001 (1.278) | 18.7 |
| usp_r2u4_flash_ptrr | 399.1 | 395.5 / 404.8 | 499 | 2.56x | 1.001 (1.278) | 18.7 |
| usp_r4u2_online_ptrr | 437.7 | 436.1 / 442.4 | 455 | 2.33x | 1.001 (2.003) | 20.1 |
| ring8_online_ptrr | 482.6 | 478.6 / 488.0 | 413 | 2.11x | 1.001 (2.622) | 15.6 |
| ring8_online_contig | 1010.3 | 1003.2 / 1018.9 | 197 | 1.01x | 2.622 (2.622) | 15.6 |
| ring8_python_contig | 1019.7 | 1011.1 / 1030.9 | 195 | 1.00x | 2.622 (2.622) | 15.6 |
| ring8_flash_contig | 1028.5 | 1022.5 / 1038.2 | 194 | 0.99x | 2.622 (2.622) | 15.6 |

### full_128k (world 8, heads 32/32, mask density 1.000)

| config | median ms | p10 / p90 | TFLOP/s/GPU | vs naive ring | ring imbalance (contiguous) | peak GiB |
|---|---|---|---|---|---|---|
| ulysses8 | 204.6 | 202.2 / 210.9 | 602 | - | - | 1.9 |
| usp_r2u4_online_ptrr | 220.7 | 215.9 / 227.3 | 558 | - | 1.000 (1.000) | 4.7 |
| usp_r2u4_python_ptrr | 220.8 | 218.6 / 225.6 | 558 | - | 1.000 (1.000) | 4.7 |
| usp_r2u4_flash_ptrr | 221.6 | 218.6 / 225.8 | 556 | - | 1.000 (1.000) | 4.7 |
| usp_r4u2_online_ptrr | 227.3 | 222.1 / 234.6 | 542 | - | 1.000 (1.000) | 5.0 |

### full_512k (world 8, heads 32/32, mask density 1.000)

| config | median ms | p10 / p90 | TFLOP/s/GPU | vs naive ring | ring imbalance (contiguous) | peak GiB |
|---|---|---|---|---|---|---|
| ulysses8 | 3266.1 | 3243.5 / 3283.1 | 603 | - | - | 7.5 |
| usp_r4u2_online_ptrr | 3443.0 | 3426.2 / 3471.0 | 572 | - | 1.000 (1.000) | 20.1 |
| usp_r2u4_python_ptrr | 3462.9 | 3414.3 / 3486.6 | 569 | - | 1.000 (1.000) | 18.7 |
| usp_r2u4_flash_ptrr | 3464.8 | 3429.5 / 3491.0 | 569 | - | 1.000 (1.000) | 18.7 |
| usp_r2u4_online_ptrr | 3477.5 | 3439.4 / 3511.2 | 567 | - | 1.000 (1.000) | 18.7 |

### mixed_1024k (world 8, heads 32/32, mask density 0.192)

| config | median ms | p10 / p90 | TFLOP/s/GPU | vs naive ring | ring imbalance (contiguous) | peak GiB |
|---|---|---|---|---|---|---|
| ulysses8 | 2657.9 | 2653.0 / 2663.0 | 569 | 2.18x | - | 16.1 |
| usp_r2u4_online_ptrr | 2711.4 | 2693.7 / 2725.6 | 558 | 2.13x | 1.000 (1.067) | 37.6 |
| usp_r2u4_python_ptrr | 2718.6 | 2697.5 / 2731.5 | 557 | 2.13x | 1.000 (1.067) | 37.6 |
| usp_r2u4_flash_ptrr | 2724.0 | 2704.0 / 2734.9 | 556 | 2.12x | 1.000 (1.067) | 37.6 |
| usp_r4u2_online_ptrr | 2794.2 | 2782.8 / 2809.4 | 542 | 2.07x | 1.000 (1.404) | 40.4 |
| ring8_online_ptrr | 2869.4 | 2856.9 / 2878.5 | 528 | 2.02x | 1.001 (2.011) | 31.3 |
| ring8_online_contig | 5757.5 | 5727.2 / 5806.6 | 263 | 1.00x | 2.011 (2.011) | 31.2 |
| ring8_python_contig | 5782.8 | 5748.6 / 5825.1 | 262 | 1.00x | 2.011 (2.011) | 31.3 |
| ring8_flash_contig | 5797.9 | 5773.1 / 5850.9 | 261 | 1.00x | 2.011 (2.011) | 31.3 |

### mixed_128k (world 8, heads 32/32, mask density 0.195)

| config | median ms | p10 / p90 | TFLOP/s/GPU | vs naive ring | ring imbalance (contiguous) | peak GiB |
|---|---|---|---|---|---|---|
| ulysses8 | 51.5 | 49.9 / 53.1 | 465 | 2.39x | - | 1.9 |
| usp_r2u4_online_ptrr | 54.9 | 54.1 / 56.8 | 437 | 2.24x | 1.001 (1.084) | 4.7 |
| usp_r2u4_python_ptrr | 55.3 | 54.6 / 56.8 | 433 | 2.22x | 1.001 (1.084) | 4.7 |
| usp_r2u4_flash_ptrr | 55.6 | 55.0 / 58.3 | 431 | 2.21x | 1.001 (1.084) | 4.7 |
| usp_r4u2_online_ptrr | 62.6 | 61.7 / 64.0 | 383 | 1.96x | 1.005 (1.361) | 5.0 |
| ring8_online_ptrr | 81.1 | 80.2 / 81.8 | 295 | 1.51x | 1.007 (2.021) | 3.9 |
| ring8_online_contig | 121.3 | 118.8 / 123.1 | 197 | 1.01x | 2.021 (2.021) | 3.9 |
| ring8_python_contig | 122.9 | 120.6 / 124.0 | 195 | 1.00x | 2.021 (2.021) | 3.9 |
| ring8_flash_contig | 125.0 | 123.7 / 126.1 | 192 | 0.98x | 2.021 (2.021) | 3.9 |

### mixed_512k (world 8, heads 32/32, mask density 0.193)

| config | median ms | p10 / p90 | TFLOP/s/GPU | vs naive ring | ring imbalance (contiguous) | peak GiB |
|---|---|---|---|---|---|---|
| ulysses8 | 678.8 | 674.0 / 683.2 | 559 | 2.18x | - | 7.8 |
| usp_r2u4_online_ptrr | 698.4 | 691.4 / 702.0 | 544 | 2.12x | 1.000 (1.068) | 18.7 |
| usp_r2u4_python_ptrr | 702.1 | 697.9 / 705.8 | 541 | 2.11x | 1.000 (1.068) | 18.7 |
| usp_r2u4_flash_ptrr | 702.9 | 699.7 / 707.2 | 540 | 2.11x | 1.000 (1.068) | 18.7 |
| usp_r4u2_online_ptrr | 736.8 | 732.6 / 742.2 | 515 | 2.01x | 1.001 (1.396) | 20.1 |
| ring8_online_ptrr | 773.6 | 768.9 / 776.6 | 491 | 1.91x | 1.001 (2.009) | 15.6 |
| ring8_online_contig | 1469.0 | 1459.7 / 1478.7 | 258 | 1.01x | 2.009 (2.009) | 15.6 |
| ring8_python_contig | 1480.2 | 1471.4 / 1494.5 | 257 | 1.00x | 2.009 (2.009) | 15.6 |
| ring8_flash_contig | 1487.4 | 1480.9 / 1499.1 | 255 | 1.00x | 2.009 (2.009) | 15.6 |

**Failed or timed-out configs**

- `mixed_128k_usp_r2u4_online_ptrr_symm`: exit=1

## Real head layouts, one node vs two nodes

Cells: step time in ms (best layout of that family) / tokens per second per GPU, 8 GPUs | 16 GPUs. `-` means the head layout does not allow it.

| scenario | ulysses | ulysses_dp | naive_ring | ring_ptrr | ring_ptrr_nccl4py | usp | usp_nccl4py |
|---|---|---|---|---|---|---|---|
| qwen7b_28q4kv_mixed_1024k | - \| - | 4618 / 57k \| - | 5022 / 26k \| 2912 / 23k | 2400 / 55k \| 1283 / 51k | - \| 1241 / 53k | 2320 / 57k \| 1189 / 55k | - \| 1180 / 56k |
| qwen7b_28q4kv_mixed_256k | - \| - | 296 / 221k \| - | 330 / 99k \| 208 / 79k | 175 / 187k \| 106 / 154k | - \| 100 / 164k | 156 / 209k \| 87 / 187k | - \| 88 / 185k |
| llama_32q8kv_docs_1024k | 1403 / 93k \| 737 / 89k | - \| - | 3777 / 35k \| 2325 / 28k | 1657 / 79k \| 956 / 69k | - \| 854 / 77k | 1461 / 90k \| 749 / 88k | - \| 738 / 89k |
| llama_32q8kv_docs_256k | 96 / 342k \| 58 / 284k | - \| - | 268 / 122k \| 203 / 81k | 139 / 236k \| 120 / 137k | - \| 118 / 139k | 108 / 303k \| 60 / 275k | - \| 59 / 277k |
| mha_32q_mixed_1024k | 2665 / 49k \| 1365 / 48k | - \| - | 5834 / 22k \| 4087 / 16k | 2858 / 46k \| 1982 / 33k | - \| 1608 / 41k | 2724 / 48k \| 1423 / 46k | - \| 1419 / 46k |
| mha_32q_mixed_256k | 177 / 185k \| 101 / 163k | - \| - | 402 / 81k \| 452 / 36k | 222 / 147k \| 416 / 39k | - \| 486 / 34k | 191 / 171k \| 111 / 147k | - \| 119 / 138k |
| wan_40q_full_128k | 245 / 67k \| - | - \| 247 / 66k | 293 / 56k \| 276 / 30k | 272 / 60k \| 267 / 31k | - \| 289 / 28k | 256 / 64k \| 137 / 60k | - \| 149 / 55k |
| wan_40q_full_512k | 4099 / 16k \| - | - \| 4105 / 16k | 4361 / 15k \| 2464 / 13k | 4184 / 16k \| 2349 / 14k | - \| 2209 / 15k | 4159 / 16k \| 2041 / 16k | - \| 2115 / 15k |

## Transports (transport8)

| config | median ms |
|---|---|
| mixed_512k_r8u1_nccl | 732.1 |
| mixed_512k_r8u1_nccl4py | 724.7 |
| mixed_512k_r8u1_symm_mem | 734.9 |
| mixed_512k_r8u1_symm_mem_direct | 735.2 |
| mixed_512k_r8u1_nccl_serial | 734.6 |

## Transports (transport16)

| config | median ms |
|---|---|
| mixed_1024k_r2u8_nccl | 1354.6 |
| mixed_1024k_r2u8_nccl4py | 1344.6 |
| mixed_1024k_r2u8_nccl_serial | 1354.9 |
| mixed_1024k_r16u1_nccl | 1523.7 |
| mixed_1024k_r16u1_nccl4py | 1432.6 |
| mixed_1024k_r16u1_nccl_serial | 1534.5 |

## PyTorch-native context parallelism vs ours (8 GPUs)

### causal_128k (world 8, heads 32/32, mask density 0.500)

| config | median ms | p10 / p90 | TFLOP/s/GPU | vs naive ring | ring imbalance (contiguous) | peak GiB |
|---|---|---|---|---|---|---|
| ulysses8 | 106.3 | 103.8 / 116.3 | 579 | 2.08x | - | 1.9 |
| usp_r2u4_online_ptrr | 113.9 | 112.1 / 120.4 | 541 | 1.94x | 1.000 (1.500) | 4.6 |
| ring8_online_ptrr | 136.1 | 133.8 / 141.1 | 452 | 1.62x | 1.000 (1.875) | 4.4 |
| torch_sdpa_ring_allgather | 144.2 | 142.1 / 151.3 | 427 | 1.53x | nan (nan) | 9.6 |
| torch_sdpa_ring_alltoall | 144.7 | 142.5 / 156.4 | 425 | 1.53x | nan (nan) | 8.1 |
| torch_flex_allgather_ptrr | 179.4 | 176.8 / 183.8 | 343 | 1.23x | nan (nan) | 6.0 |
| ring8_python_contig | 220.8 | 217.3 / 228.7 | 279 | 1.00x | 1.875 (1.875) | 4.4 |

### causal_512k (world 8, heads 32/32, mask density 0.500)

| config | median ms | p10 / p90 | TFLOP/s/GPU | vs naive ring | ring imbalance (contiguous) | peak GiB |
|---|---|---|---|---|---|---|
| ulysses8 | 1637.1 | 1634.2 / 1640.9 | 602 | 1.95x | - | 7.5 |
| usp_r2u4_online_ptrr | 1726.8 | 1702.9 / 1744.2 | 571 | 1.85x | 1.000 (1.500) | 18.5 |
| ring8_online_ptrr | 1779.4 | 1775.2 / 1785.5 | 554 | 1.79x | 1.000 (1.875) | 17.5 |
| torch_sdpa_ring_allgather | 1894.6 | 1891.8 / 1898.3 | 520 | 1.68x | nan (nan) | 38.3 |
| torch_sdpa_ring_alltoall | 1904.1 | 1901.6 / 1908.4 | 517 | 1.68x | nan (nan) | 32.3 |
| torch_flex_allgather_ptrr | 2766.1 | 2758.5 / 2775.8 | 356 | 1.15x | nan (nan) | 24.3 |
| ring8_python_contig | 3191.1 | 3168.3 / 3220.7 | 309 | 1.00x | 1.875 (1.875) | 17.6 |

### full_128k (world 8, heads 32/32, mask density 1.000)

| config | median ms | p10 / p90 | TFLOP/s/GPU | vs naive ring | ring imbalance (contiguous) | peak GiB |
|---|---|---|---|---|---|---|
| ulysses8 | 204.7 | 201.1 / 211.2 | 602 | 1.18x | - | 1.9 |
| usp_r2u4_online_ptrr | 209.6 | 205.6 / 216.1 | 587 | 1.15x | 1.000 (1.000) | 4.6 |
| ring8_online_ptrr | 231.8 | 225.9 / 240.3 | 531 | 1.04x | 1.000 (1.000) | 4.4 |
| ring8_python_contig | 241.9 | 235.8 / 251.9 | 509 | 1.00x | 1.000 (1.000) | 4.4 |
| torch_sdpa_ring_alltoall | 256.5 | 248.5 / 268.7 | 480 | 0.94x | nan (nan) | 8.3 |
| torch_sdpa_ring_allgather | 257.2 | 250.2 / 265.6 | 479 | 0.94x | nan (nan) | 9.8 |
| torch_flex_allgather_ptrr | 324.0 | 320.8 / 331.8 | 380 | 0.75x | nan (nan) | 6.0 |

### full_512k (world 8, heads 32/32, mask density 1.000)

| config | median ms | p10 / p90 | TFLOP/s/GPU | vs naive ring | ring imbalance (contiguous) | peak GiB |
|---|---|---|---|---|---|---|
| ulysses8 | 3261.9 | 3249.0 / 3277.6 | 604 | 1.06x | - | 7.5 |
| usp_r2u4_online_ptrr | 3331.1 | 3287.3 / 3346.7 | 591 | 1.04x | 1.000 (1.000) | 18.5 |
| ring8_online_ptrr | 3331.2 | 3313.2 / 3343.8 | 591 | 1.04x | 1.000 (1.000) | 17.5 |
| ring8_python_contig | 3470.1 | 3457.5 / 3492.3 | 568 | 1.00x | 1.000 (1.000) | 17.6 |
| torch_sdpa_ring_allgather | 3684.4 | 3671.9 / 3693.6 | 535 | 0.94x | nan (nan) | 39.0 |
| torch_sdpa_ring_alltoall | 3693.9 | 3689.1 / 3702.5 | 533 | 0.94x | nan (nan) | 33.0 |
| torch_flex_allgather_ptrr | 5152.3 | 5137.7 / 5158.7 | 382 | 0.67x | nan (nan) | 24.3 |

### mixed_128k (world 8, heads 32/32, mask density 0.195)

| config | median ms | p10 / p90 | TFLOP/s/GPU | vs naive ring | ring imbalance (contiguous) | peak GiB |
|---|---|---|---|---|---|---|
| torch_flex_allgather_ptrr | 81.6 | 80.9 / 84.6 | 293 | - | nan (nan) | 6.0 |

### mixed_512k (world 8, heads 32/32, mask density 0.193)

| config | median ms | p10 / p90 | TFLOP/s/GPU | vs naive ring | ring imbalance (contiguous) | peak GiB |
|---|---|---|---|---|---|---|
| torch_flex_allgather_ptrr | 1047.2 | 1045.1 / 1049.4 | 363 | - | nan (nan) | 24.3 |

## Ours vs other context-parallel attention libraries (8 GPUs)

Step time (forward + backward) in ms, 8x H100, 32 MHA heads, head dim 128; lower is better. yunchang 0.6.4 ignores `causal` in its FA3 wrappers; its causal rows use the patched wrappers in `baselines/yunchang_usp.py` (outputs and gradients checked against SDPA). Each cell lists the implementations in the order of the header; for ring x Ulysses the best of r2u4 / r4u2. `-` means the implementation does not support the mask (yunchang: plain causal or full only; TE: no mixed masks) or the run did not complete.

| scenario | Ulysses (ours / TE / yunchang) | ring (ours / TE / torch / yunchang) | ring x Ulysses (ours / TE / yunchang) | all-gather / other (TE / torch flex / Magi) |
|---|---|---|---|---|
| mixed_128k | 51 / - / - | 81 / - / - / - | 55 / - / - | - / 82 / 90 |
| mixed_512k | 679 / - / - | 774 / - / - / - | 698 / - / - | - / 1047 / 1076 |
| mixed_1024k | 2658 / - / - | 2869 / - / - / - | 2711 / - / - | - / - / - |
| docs_128k | 29 / - / - | 64 / 52 / - / - | 35 / - / - | - / - / 57 |
| docs_512k | 374 / - / - | 483 / 482 / - / - | 395 / - / - | - / - / 602 |
| causal_128k | 106 / 122 / 108 | 136 / 129 / 144 / 135 | 114 / 122 / 110 | 149 / 179 / 183 |
| causal_512k | 1637 / 1892 / 1604 | 1779 / 1851 / 1895 / 1731 | 1727 / 1851 / 1641 | 2056 / 2766 / 2651 |
| full_128k | 205 / 232 / 212 | 232 / 242 / 256 / 241 | 210 / 233 / 230 | 253 / 324 / 340 |
| full_512k | 3262 / 3565 / 3328 | 3331 / 3609 / 3684 / 3419 | 3331 / 3606 / 3396 | 3637 / 5152 / 5145 |

## Correctness of the baselines (forward + backward)

Each baseline sharded with its own layout, forward + backward on 8 GPUs, 8k tokens, compared with an fp32 dense reference on every rank's tokens. bf16 attention lands at about 2e-3 to 4e-3; a baseline is only benchmarked if it passes (all errors below 1e-2). `yunchang_fa3_zigzag` without `_patched` is yunchang 0.6.4 as shipped, kept to show its causal bug.

| implementation | layout | config | out | dq | dk | dv | pass |
|---|---|---|---|---|---|---|---|
| te_a2a | causal | - | 2.17e-03 | 2.29e-03 | 2.27e-03 | 2.27e-03 | yes |
| te_p2p | causal | - | 3.32e-03 | 3.61e-03 | 3.65e-03 | 3.65e-03 | yes |
| te_all_gather | causal | - | 2.17e-03 | 2.29e-03 | 4.14e-03 | 4.14e-03 | yes |
| te_a2a+p2p | causal | ulysses 4 | 2.56e-03 | 2.63e-03 | 2.74e-03 | 2.75e-03 | yes |
| te_a2a+p2p | causal | ulysses 2 | 2.87e-03 | 3.04e-03 | 3.15e-03 | 3.15e-03 | yes |
| te_a2a | full | - | 2.22e-03 | 2.30e-03 | 2.27e-03 | 2.28e-03 | yes |
| te_p2p | full | - | 3.90e-03 | 4.12e-03 | 4.08e-03 | 4.10e-03 | yes |
| te_all_gather | full | - | 2.22e-03 | 2.30e-03 | 4.48e-03 | 4.50e-03 | yes |
| te_a2a+p2p | full | ulysses 4 | 2.90e-03 | 2.84e-03 | 2.81e-03 | 2.82e-03 | yes |
| te_a2a+p2p | full | ulysses 2 | 3.28e-03 | 3.40e-03 | 3.37e-03 | 3.38e-03 | yes |
| te_p2p | docs | - | 3.60e-03 | 3.63e-03 | 3.66e-03 | 3.63e-03 | yes |
| te_a2a | causal | 8 kv heads | 2.16e-03 | 2.29e-03 | 2.81e-03 | 2.80e-03 | yes |
| yunchang_fa3_zigzag_patched | causal | 1 x 8 | 2.18e-03 | 2.85e-03 | 2.83e-03 | 2.83e-03 | yes |
| yunchang_fa3_zigzag_patched | causal | 2 x 4 | 2.53e-03 | 3.19e-03 | 3.16e-03 | 3.16e-03 | yes |
| yunchang_fa3_zigzag_patched | causal | 4 x 2 | 2.60e-03 | 3.24e-03 | 3.21e-03 | 3.20e-03 | yes |
| yunchang_fa3_zigzag_patched | causal | 8 x 1 | 2.60e-03 | 3.25e-03 | 3.26e-03 | 3.27e-03 | yes |
| yunchang_fa3_basic_patched | full | 1 x 8 | 2.22e-03 | 2.85e-03 | 2.83e-03 | 2.85e-03 | yes |
| yunchang_fa3_basic_patched | full | 2 x 4 | 2.69e-03 | 3.31e-03 | 3.29e-03 | 3.31e-03 | yes |
| yunchang_fa3_basic_patched | full | 4 x 2 | 2.71e-03 | 3.30e-03 | 3.28e-03 | 3.29e-03 | yes |
| yunchang_fa3_basic_patched | full | 8 x 1 | 2.71e-03 | 3.30e-03 | 3.28e-03 | 3.30e-03 | yes |
| yunchang_fa3_zigzag | causal | 2 x 4 | 5.74e-01 | 5.74e-01 | 1.41e+00 | 1.41e+00 | **no** |
| torch_flex_allgather | mixed | - | 2.07e-03 | 2.30e-03 | 3.55e-03 | 3.51e-03 | yes |
| torch_flex_allgather | docs | - | 2.03e-03 | 2.30e-03 | 3.27e-03 | 3.23e-03 | yes |
| torch_flex_allgather | causal | - | 2.15e-03 | 2.29e-03 | 4.02e-03 | 4.01e-03 | yes |
| torch_flex_allgather | full | - | 2.22e-03 | 2.30e-03 | 4.08e-03 | 4.09e-03 | yes |
| torch_sdpa_ring | causal | - | 2.60e-03 | 2.77e-03 | 2.79e-03 | 2.78e-03 | yes |
| torch_sdpa_ring | full | - | 2.72e-03 | 2.84e-03 | 2.81e-03 | 2.81e-03 | yes |
| magi_attention | mixed | - | 2.06e-03 | 2.29e-03 | 2.78e-03 | 2.77e-03 | yes |
| magi_attention | docs | - | 2.02e-03 | 2.30e-03 | 2.76e-03 | 2.73e-03 | yes |
| magi_attention | causal | - | 2.14e-03 | 2.29e-03 | 2.80e-03 | 2.80e-03 | yes |
| magi_attention | full | - | 2.21e-03 | 2.30e-03 | 2.81e-03 | 2.82e-03 | yes |

## Ring accuracy vs other libraries

8-way ring, 8k tokens, 32 MHA heads, head dim 128; mean relative error vs fp32 (same shape as the baseline checks).

| implementation | causal out | causal dq | causal dk | full out | full dq | full dk |
|---|---|---|---|---|---|---|
| ours, online combine | 2.14e-03 | 1.69e-03 | 2.33e-03 | 2.22e-03 | 1.69e-03 | 2.33e-03 |
| ours, flash combine | 3.45e-03 | 1.72e-03 | 2.33e-03 | 3.86e-03 | 1.71e-03 | 2.34e-03 |
| ours, python combine | 2.84e-03 | 1.71e-03 | 2.33e-03 | 2.85e-03 | 1.69e-03 | 2.33e-03 |
| single-GPU FA4 (no parallelism) | 2.14e-03 | 1.69e-03 | 1.66e-03 | 2.22e-03 | 1.69e-03 | 1.66e-03 |
| TransformerEngine p2p | 3.32e-03 | 3.61e-03 | 3.65e-03 | 3.90e-03 | 4.12e-03 | 4.08e-03 |
| yunchang ring (patched) | 2.60e-03 | 3.25e-03 | 3.26e-03 | 2.71e-03 | 3.30e-03 | 3.28e-03 |
| PyTorch SDPA ring | 2.60e-03 | 2.77e-03 | 2.79e-03 | 2.72e-03 | 2.84e-03 | 2.81e-03 |

## Model-level parity: 7B multimodal MoT decoder

Two full-size 7B multimodal MoT decoder layers, 8k tokens, same weights and inputs. Mean relative error against an fp32 model with exact masked attention, as USP / single-GPU FA4 (the no-parallelism bf16 result). Parameter gradients of USP are summed over ranks in bf16, which adds about 2% on its own (see the Ulysses-only rows, whose attention is bitwise identical to FA4).

| layout | ring x ulysses | mask path | combine | fp32 grads | output | input grad | param grads |
|---|---|---|---|---|---|---|---|
| mixed | 2 x 4 | block-sparse | online | True | 7.602e-03 / 7.600e-03 | 9.105e-03 / 9.100e-03 | 9.345e-03 / 9.194e-03 |
| mixed | 2 x 4 | block-sparse | flash | True | 7.605e-03 / 7.600e-03 | 9.106e-03 / 9.100e-03 | 9.349e-03 / 9.194e-03 |
| mixed | 2 x 4 | block-sparse | python | True | 7.610e-03 / 7.600e-03 | 9.108e-03 / 9.100e-03 | 9.356e-03 / 9.194e-03 |
| mixed | 4 x 2 | block-sparse | online | True | 7.602e-03 / 7.600e-03 | 9.104e-03 / 9.100e-03 | 9.345e-03 / 9.194e-03 |
| mixed | 8 x 1 | block-sparse | online | True | 7.602e-03 / 7.600e-03 | 9.103e-03 / 9.100e-03 | 9.346e-03 / 9.194e-03 |
| mixed | 8 x 1 | block-sparse | flash | True | 7.612e-03 / 7.600e-03 | 9.113e-03 / 9.100e-03 | 9.359e-03 / 9.194e-03 |
| mixed | 8 x 1 | block-sparse | python | True | 7.607e-03 / 7.600e-03 | 9.106e-03 / 9.100e-03 | 9.351e-03 / 9.194e-03 |
| docs | 2 x 4 | block-sparse | online | True | 7.684e-03 / 7.683e-03 | 9.272e-03 / 9.272e-03 | 9.546e-03 / 9.399e-03 |
| docs | 2 x 4 | block-sparse | flash | True | 7.688e-03 / 7.683e-03 | 9.274e-03 / 9.272e-03 | 9.550e-03 / 9.399e-03 |
| docs | 2 x 4 | block-sparse | python | True | 7.699e-03 / 7.683e-03 | 9.281e-03 / 9.272e-03 | 9.565e-03 / 9.399e-03 |
| docs | 4 x 2 | block-sparse | online | True | 7.683e-03 / 7.683e-03 | 9.274e-03 / 9.272e-03 | 9.546e-03 / 9.399e-03 |
| docs | 8 x 1 | block-sparse | online | True | 7.683e-03 / 7.683e-03 | 9.274e-03 / 9.272e-03 | 9.547e-03 / 9.399e-03 |
| docs | 8 x 1 | block-sparse | flash | True | 7.698e-03 / 7.683e-03 | 9.282e-03 / 9.272e-03 | 9.560e-03 / 9.399e-03 |
| docs | 8 x 1 | block-sparse | python | True | 7.695e-03 / 7.683e-03 | 9.279e-03 / 9.272e-03 | 9.559e-03 / 9.399e-03 |
| causal | 2 x 4 | block-sparse | online | True | 7.529e-03 / 7.529e-03 | 8.941e-03 / 8.942e-03 | 9.214e-03 / 9.064e-03 |
| causal | 2 x 4 | block-sparse | flash | True | 7.531e-03 / 7.529e-03 | 8.946e-03 / 8.942e-03 | 9.218e-03 / 9.064e-03 |
| causal | 2 x 4 | block-sparse | python | True | 7.536e-03 / 7.529e-03 | 8.947e-03 / 8.942e-03 | 9.223e-03 / 9.064e-03 |
| causal | 4 x 2 | block-sparse | online | True | 7.530e-03 / 7.529e-03 | 8.942e-03 / 8.942e-03 | 9.213e-03 / 9.064e-03 |
| causal | 8 x 1 | block-sparse | online | True | 7.531e-03 / 7.529e-03 | 8.944e-03 / 8.942e-03 | 9.215e-03 / 9.064e-03 |
| causal | 8 x 1 | block-sparse | flash | True | 7.540e-03 / 7.529e-03 | 8.952e-03 / 8.942e-03 | 9.225e-03 / 9.064e-03 |
| causal | 8 x 1 | block-sparse | python | True | 7.534e-03 / 7.529e-03 | 8.942e-03 / 8.942e-03 | 9.219e-03 / 9.064e-03 |
| full | 2 x 4 | block-sparse | online | True | 7.498e-03 / 7.499e-03 | 8.873e-03 / 8.873e-03 | 9.111e-03 / 8.960e-03 |
| full | 2 x 4 | block-sparse | flash | True | 7.499e-03 / 7.499e-03 | 8.871e-03 / 8.873e-03 | 9.113e-03 / 8.960e-03 |
| full | 2 x 4 | block-sparse | python | True | 7.498e-03 / 7.499e-03 | 8.870e-03 / 8.873e-03 | 9.112e-03 / 8.960e-03 |
| full | 4 x 2 | block-sparse | online | True | 7.500e-03 / 7.499e-03 | 8.872e-03 / 8.872e-03 | 9.111e-03 / 8.960e-03 |
| full | 8 x 1 | block-sparse | online | True | 7.498e-03 / 7.499e-03 | 8.872e-03 / 8.873e-03 | 9.112e-03 / 8.960e-03 |
| full | 8 x 1 | block-sparse | flash | True | 7.502e-03 / 7.499e-03 | 8.874e-03 / 8.873e-03 | 9.117e-03 / 8.960e-03 |
| full | 8 x 1 | block-sparse | python | True | 7.498e-03 / 7.499e-03 | 8.871e-03 / 8.873e-03 | 9.112e-03 / 8.960e-03 |
| causal | 1 x 4 | block-sparse | online | True | 7.529e-03 / 7.529e-03 | 8.942e-03 / 8.942e-03 | 9.442e-03 / 9.064e-03 |
| mixed | 1 x 4 | block-sparse | online | True | 7.600e-03 / 7.600e-03 | 9.101e-03 / 9.100e-03 | 9.546e-03 / 9.194e-03 |
| mixed | 8 x 1 | block-sparse | online | False | 7.602e-03 / 7.600e-03 | 9.109e-03 / 9.100e-03 | 9.349e-03 / 9.194e-03 |
| causal | 2 x 4 | native | online | True | 7.528e-03 / 7.529e-03 | 8.944e-03 / 8.942e-03 | 9.216e-03 / 9.064e-03 |
| causal | 2 x 4 | native | flash | True | 7.532e-03 / 7.529e-03 | 8.944e-03 / 8.942e-03 | 9.221e-03 / 9.064e-03 |
| causal | 2 x 4 | native | python | True | 7.537e-03 / 7.529e-03 | 8.948e-03 / 8.942e-03 | 9.225e-03 / 9.064e-03 |
| causal | 4 x 2 | native | online | True | 7.529e-03 / 7.529e-03 | 8.944e-03 / 8.942e-03 | 9.215e-03 / 9.064e-03 |
| causal | 4 x 2 | native | flash | True | 7.537e-03 / 7.529e-03 | 8.948e-03 / 8.942e-03 | 9.227e-03 / 9.064e-03 |
| causal | 4 x 2 | native | python | True | 7.535e-03 / 7.529e-03 | 8.947e-03 / 8.942e-03 | 9.222e-03 / 9.064e-03 |
| causal | 8 x 1 | native | online | True | 7.530e-03 / 7.529e-03 | 8.943e-03 / 8.942e-03 | 9.215e-03 / 9.064e-03 |
| causal | 8 x 1 | native | flash | True | 7.540e-03 / 7.529e-03 | 8.955e-03 / 8.942e-03 | 9.238e-03 / 9.064e-03 |
| causal | 8 x 1 | native | python | True | 7.532e-03 / 7.529e-03 | 8.946e-03 / 8.942e-03 | 9.220e-03 / 9.064e-03 |
| full | 2 x 4 | native | online | True | 7.499e-03 / 7.499e-03 | 8.873e-03 / 8.873e-03 | 9.111e-03 / 8.960e-03 |
| full | 2 x 4 | native | flash | True | 7.499e-03 / 7.499e-03 | 8.872e-03 / 8.873e-03 | 9.113e-03 / 8.960e-03 |
| full | 2 x 4 | native | python | True | 7.501e-03 / 7.499e-03 | 8.874e-03 / 8.873e-03 | 9.114e-03 / 8.960e-03 |
| full | 4 x 2 | native | online | True | 7.499e-03 / 7.499e-03 | 8.872e-03 / 8.873e-03 | 9.111e-03 / 8.960e-03 |
| full | 4 x 2 | native | flash | True | 7.501e-03 / 7.499e-03 | 8.872e-03 / 8.873e-03 | 9.114e-03 / 8.960e-03 |
| full | 4 x 2 | native | python | True | 7.501e-03 / 7.499e-03 | 8.870e-03 / 8.873e-03 | 9.114e-03 / 8.960e-03 |
| full | 8 x 1 | native | online | True | 7.499e-03 / 7.499e-03 | 8.869e-03 / 8.873e-03 | 9.111e-03 / 8.960e-03 |
| full | 8 x 1 | native | flash | True | 7.502e-03 / 7.499e-03 | 8.872e-03 / 8.873e-03 | 9.116e-03 / 8.960e-03 |
| full | 8 x 1 | native | python | True | 7.500e-03 / 7.499e-03 | 8.871e-03 / 8.873e-03 | 9.112e-03 / 8.960e-03 |

## Native FA4 mask for plain layouts

One plain causal / full document, 8x H100, 32 heads; both columns measured in the same session. With a ring, the native path stripes a causal document over the ring ranks so every step is one causal or strictly causal FA4 call.

| scenario | block-sparse mask ms | native mask ms | change |
|---|---|---|---|
| causal_128k_ulysses8 | 113.9 | 106.3 | -6.7% |
| causal_128k_usp_r2u4_online_ptrr | 115.8 | 113.9 | -1.7% |
| causal_128k_ring8_online_ptrr | 136.7 | 136.1 | -0.5% |
| causal_512k_ulysses8 | 1727.1 | 1637.1 | -5.2% |
| causal_512k_usp_r2u4_online_ptrr | 1767.0 | 1726.8 | -2.3% |
| causal_512k_ring8_online_ptrr | 1784.3 | 1779.4 | -0.3% |
| full_128k_ulysses8 | 214.0 | 204.7 | -4.3% |
| full_128k_usp_r2u4_online_ptrr | 221.2 | 209.6 | -5.2% |
| full_128k_ring8_online_ptrr | 238.9 | 231.8 | -3.0% |
| full_512k_ulysses8 | 3454.4 | 3261.9 | -5.6% |
| full_512k_usp_r2u4_online_ptrr | 3465.2 | 3331.1 | -3.9% |
| full_512k_ring8_online_ptrr | 3460.9 | 3331.2 | -3.7% |

## Native masks: accuracy vs the block-sparse path

Online combine, 16k tokens, mean relative error vs fp32 (out / dq / dk / dv).

| layout | ring x ulysses | block-sparse path | native path |
|---|---|---|---|
| causal | 2 x 4 | 2.17e-03 / 1.69e-03 / 2.39e-03 / 2.39e-03 | 2.16e-03 / 1.69e-03 / 2.40e-03 / 2.39e-03 |
| causal | 4 x 2 | 2.17e-03 / 1.69e-03 / 2.34e-03 / 2.33e-03 | 2.16e-03 / 1.69e-03 / 2.34e-03 / 2.34e-03 |
| causal | 8 x 1 | 2.17e-03 / 1.69e-03 / 2.33e-03 / 2.33e-03 | 2.16e-03 / 1.69e-03 / 2.34e-03 / 2.33e-03 |
| full | 2 x 4 | 2.23e-03 / 1.69e-03 / 2.40e-03 / 2.40e-03 | 2.23e-03 / 1.69e-03 / 2.40e-03 / 2.40e-03 |
| full | 4 x 2 | 2.23e-03 / 1.69e-03 / 2.34e-03 / 2.34e-03 | 2.23e-03 / 1.69e-03 / 2.34e-03 / 2.34e-03 |
| full | 8 x 1 | 2.23e-03 / 1.69e-03 / 2.34e-03 / 2.34e-03 | 2.23e-03 / 1.69e-03 / 2.34e-03 / 2.34e-03 |

## FA3 vs FA4 kernels on one GPU

One H100, 4 heads x head dim 128 over the whole sequence (the per-GPU work of Ulysses-8 with 32 heads), native causal / full masks, median ms.

| layout | seq | FA3 fwd | FA4 fwd | FA3 fwd+bwd | FA4 fwd+bwd | FA4 vs FA3 |
|---|---|---|---|---|---|---|
| causal | 128k | 24.7 | 27.2 | 96.3 | 97.7 | +1.4% |
| causal | 512k | 415.6 | 448.8 | 1572.8 | 1620.8 | +3.1% |
| full | 128k | 50.1 | 51.1 | 199.1 | 193.9 | -2.6% |
| full | 512k | 867.0 | 854.4 | 3325.0 | 3223.6 | -3.0% |

## Convergence: ~200M LM, 64k packed tokens per step, 8 GPUs

Same model init and data order; the Ulysses run computes attention for the whole sequence in one FA4 call per head subset, so it is the single-GPU numerics. `ulysses_ref_repeat` is a rerun of that reference: FA4's backward is not bitwise deterministic, so it gives the run-to-run noise floor. Loss columns are means over the last 200 steps; the gap is the largest difference of the 50-step moving averages after step 200.

| run | steps | final loss | delta vs ref | max smoothed gap | mean step s |
|---|---|---|---|---|---|
| ulysses_ref | 4000 | 3.6376 | +0.0000 | 0.0000 | 0.081 |
| ring_flash_bf16grads | 4000 | 3.6377 | +0.0001 | 0.0163 | 0.196 |
| ring_online_bf16grads | 4000 | 3.6375 | -0.0001 | 0.0093 | 0.181 |
| ring_online_fp32grads | 4000 | 3.6353 | -0.0023 | 0.0127 | 0.176 |
| ring_python_bf16grads | 4000 | 3.6368 | -0.0008 | 0.0097 | 0.192 |
| ulysses_ref_repeat | 4000 | 3.6359 | -0.0017 | 0.0125 | 0.081 |

## GQA head layouts on 8 GPUs

### qwen7b_28q4kv_causal_1024k (world 8, heads 28/4, mask density 0.500)

| config | median ms | p10 / p90 | TFLOP/s/GPU | vs naive ring | ring imbalance (contiguous) | peak GiB |
|---|---|---|---|---|---|---|
| usp_r2u4_online_ptrr | 5874.1 | 5850.0 / 5920.8 | 587 | 1.57x | 1.000 (1.500) | 14.1 |
| ring8_online_ptrr | 5973.6 | 5964.4 / 5992.9 | 577 | 1.54x | 1.000 (1.875) | 11.7 |
| usp_r2u4_python_contig | 9193.3 | 9161.6 / 9245.8 | 375 | 1.00x | 1.500 (1.500) | 14.6 |
| ring8_python_contig | 11221.2 | 11188.4 / 11788.4 | 307 | 0.82x | 1.875 (1.875) | 11.3 |

### qwen7b_28q4kv_causal_256k (world 8, heads 28/4, mask density 0.500)

| config | median ms | p10 / p90 | TFLOP/s/GPU | vs naive ring | ring imbalance (contiguous) | peak GiB |
|---|---|---|---|---|---|---|
| usp_r2u4_online_ptrr | 376.6 | 373.0 / 383.8 | 572 | 1.48x | 1.000 (1.500) | 3.5 |
| ring8_online_ptrr | 386.1 | 378.7 / 396.0 | 558 | 1.44x | 1.000 (1.875) | 2.9 |
| usp_r2u4_python_contig | 557.9 | 547.2 / 566.6 | 386 | 1.00x | 1.500 (1.500) | 3.6 |
| ring8_python_contig | 704.4 | 700.1 / 712.2 | 306 | 0.79x | 1.875 (1.875) | 2.8 |

### qwen7b_28q4kv_mixed_1024k (world 8, heads 28/4, mask density 0.192)

| config | median ms | p10 / p90 | TFLOP/s/GPU | vs naive ring | ring imbalance (contiguous) | peak GiB |
|---|---|---|---|---|---|---|
| usp_r2u4_online_ptrr | 2335.5 | 2327.0 / 2349.4 | 567 | 1.08x | 1.000 (1.067) | 14.6 |
| ring8_online_ptrr | 2407.6 | 2401.0 / 2418.6 | 550 | 1.05x | 1.001 (2.011) | 11.4 |
| usp_r2u4_python_contig | 2522.3 | 2506.2 / 2543.7 | 525 | 1.00x | 1.067 (1.067) | 14.6 |
| ring8_python_contig | 5014.3 | 4993.5 / 5033.0 | 264 | 0.50x | 2.011 (2.011) | 11.4 |

### qwen7b_28q4kv_mixed_256k (world 8, heads 28/4, mask density 0.192)

| config | median ms | p10 / p90 | TFLOP/s/GPU | vs naive ring | ring imbalance (contiguous) | peak GiB |
|---|---|---|---|---|---|---|
| usp_r2u4_online_ptrr | 159.2 | 155.2 / 165.3 | 519 | 1.11x | 1.001 (1.081) | 3.6 |
| ring8_online_ptrr | 176.8 | 172.1 / 182.8 | 467 | 1.00x | 1.003 (2.029) | 2.8 |
| usp_r2u4_python_contig | 176.8 | 174.8 / 179.1 | 467 | 1.00x | 1.081 (1.081) | 3.6 |
| ring8_python_contig | 329.5 | 325.2 / 334.4 | 251 | 0.54x | 2.029 (2.029) | 2.8 |

### llama_32q8kv_causal_1024k (world 8, heads 32/8, mask density 0.500)

| config | median ms | p10 / p90 | TFLOP/s/GPU | vs naive ring | ring imbalance (contiguous) | peak GiB |
|---|---|---|---|---|---|---|
| ulysses8 | 6538.1 | 6528.0 / 6542.6 | 603 | 2.00x | - | 11.5 |
| usp_r2u4_online_ptrr | 6724.0 | 6693.2 / 6766.6 | 586 | 1.95x | 1.000 (1.500) | 19.1 |
| ring8_online_ptrr | 6852.7 | 6833.1 / 6877.9 | 575 | 1.91x | 1.000 (1.875) | 16.6 |
| ring8_python_contig | 13093.1 | 13053.0 / 13141.3 | 301 | 1.00x | 1.875 (1.875) | 15.7 |

### llama_32q8kv_causal_256k (world 8, heads 32/8, mask density 0.500)

| config | median ms | p10 / p90 | TFLOP/s/GPU | vs naive ring | ring imbalance (contiguous) | peak GiB |
|---|---|---|---|---|---|---|
| ulysses8 | 414.0 | 406.1 / 429.3 | 595 | 1.96x | - | 2.9 |
| usp_r2u4_online_ptrr | 429.5 | 424.8 / 437.8 | 573 | 1.89x | 1.000 (1.500) | 4.8 |
| ring8_online_ptrr | 448.2 | 444.0 / 461.1 | 550 | 1.81x | 1.000 (1.875) | 4.1 |
| ring8_python_contig | 811.2 | 806.8 / 819.9 | 304 | 1.00x | 1.875 (1.875) | 3.9 |

### llama_32q8kv_mixed_1024k (world 8, heads 32/8, mask density 0.192)

| config | median ms | p10 / p90 | TFLOP/s/GPU | vs naive ring | ring imbalance (contiguous) | peak GiB |
|---|---|---|---|---|---|---|
| ulysses8 | 2647.2 | 2642.7 / 2655.2 | 572 | 2.20x | - | 12.6 |
| usp_r2u4_online_ptrr | 2687.8 | 2670.9 / 2696.2 | 563 | 2.17x | 1.000 (1.067) | 19.6 |
| ring8_online_ptrr | 2768.4 | 2762.7 / 2777.5 | 547 | 2.11x | 1.001 (2.011) | 15.8 |
| ring8_python_contig | 5829.9 | 5806.1 / 5869.8 | 260 | 1.00x | 2.011 (2.011) | 15.8 |

### llama_32q8kv_mixed_256k (world 8, heads 32/8, mask density 0.192)

| config | median ms | p10 / p90 | TFLOP/s/GPU | vs naive ring | ring imbalance (contiguous) | peak GiB |
|---|---|---|---|---|---|---|
| ulysses8 | 172.7 | 170.0 / 177.6 | 547 | 2.24x | - | 3.0 |
| usp_r2u4_online_ptrr | 183.2 | 179.0 / 190.5 | 516 | 2.11x | 1.001 (1.081) | 4.8 |
| ring8_online_ptrr | 211.5 | 206.0 / 214.1 | 447 | 1.83x | 1.003 (2.029) | 3.9 |
| ring8_python_contig | 386.8 | 382.4 / 390.6 | 244 | 1.00x | 2.029 (2.029) | 3.9 |

## Ulysses-16 vs USP across two nodes (profiled pairs)

| config | median ms |
|---|---|
| llama_32q8kv_docs_256k_ulysses16 | 58.0 |
| llama_32q8kv_docs_256k_usp_r2u8_online_ptrr | 59.6 |
| llama_32q8kv_mixed_256k_ulysses16 | 96.6 |
| llama_32q8kv_mixed_256k_usp_r2u8_online_ptrr | 97.4 |
| mha_32q_mixed_256k_ulysses16 | 100.4 |
| mha_32q_mixed_256k_usp_r2u8_online_ptrr | 111.4 |

## Where the GPU time goes

GPU time in one profiled forward + backward on rank 0 (ms). `nccl` counts NCCL kernels, which occupy SMs; `memcpy` is copy-engine traffic.

| trace | attention | nccl | attention overlapping nccl | memcpy | other | GPU span |
|---|---|---|---|---|---|---|
| mixed_512k_r8u1_nccl4py | 650.4 | 8.8 | 0.0 | 27.8 | 25.4 | 711.1 |
| mixed_512k_r8u1_nccl | 650.3 | 116.8 | 70.9 | 0.0 | 25.8 | 706.9 |
| mixed_512k_r8u1_nccl_serial | 649.9 | 148.6 | 71.3 | 0.0 | 25.7 | 738.0 |
| mixed_512k_r8u1_symm_mem_direct | 645.1 | 5.0 | 0.0 | 28.8 | 68.3 | 724.2 |
| mixed_512k_r8u1_symm_mem | 644.1 | 8.4 | 0.0 | 28.5 | 62.6 | 720.9 |
| llama_32q8kv_docs_256k_ulysses16 | 42.6 | 11.8 | 0.0 | 0.0 | 2.0 | 56.7 |
| llama_32q8kv_docs_256k_usp_r2u8_online_ptrr | 48.8 | 59.4 | 6.6 | 0.0 | 2.9 | 104.9 |
| llama_32q8kv_mixed_256k_ulysses16 | 80.2 | 11.5 | 0.0 | 0.0 | 2.0 | 94.0 |
| llama_32q8kv_mixed_256k_usp_r2u8_online_ptrr | 85.9 | 19.0 | 7.5 | 0.0 | 3.0 | 100.7 |
| mha_32q_mixed_256k_ulysses16 | 77.2 | 30.6 | 0.0 | 0.0 | 2.1 | 110.3 |
| mha_32q_mixed_256k_usp_r2u8_online_ptrr | 83.6 | 282.9 | 20.2 | 0.0 | 5.4 | 352.1 |
| mixed_1024k_r16u1_nccl4py | 1335.3 | 61.1 | 0.0 | 47.5 | 51.2 | 1477.5 |
| mixed_1024k_r16u1_nccl | 1324.9 | 732.9 | 350.0 | 0.0 | 52.1 | 1715.8 |
| mixed_1024k_r16u1_nccl_serial | 1317.3 | 529.9 | 260.5 | 0.0 | 52.1 | 1595.1 |
| mixed_1024k_r2u8_nccl4py | 1286.9 | 31.0 | 0.0 | 1.8 | 12.2 | 1354.0 |
| mixed_1024k_r2u8_nccl | 1281.3 | 124.7 | 60.1 | 0.0 | 12.1 | 1358.1 |
| mixed_1024k_r2u8_nccl_serial | 1284.7 | 120.7 | 53.8 | 0.0 | 12.1 | 1364.0 |

## Online FA4 kernel vs chunked FA4

Single H100, forward only, 16k query tokens, KV split into chunks of 16k, 32 heads, dense attention. Time relative to one FA4 call over the whole KV; output error vs fp32.

| chunks | one FA4 call ms | online | FA4 per chunk + flash combine | FA4 per chunk + PyTorch combine | error: one call / online / flash / PyTorch |
|---|---|---|---|---|---|
| 1 | 6.52 | 6.78 (1.040x) | 6.66 (1.022x) | 6.41 (0.984x) | 2.24e-03 / 2.24e-03 / 2.24e-03 / 2.24e-03 |
| 2 | 13.44 | 13.12 (0.975x) | 13.99 (1.041x) | 13.40 (0.997x) | 2.25e-03 / 2.25e-03 / 2.73e-03 / 2.78e-03 |
| 4 | 25.52 | 26.56 (1.041x) | 29.60 (1.160x) | 29.32 (1.149x) | 2.25e-03 / 2.25e-03 / 3.23e-03 / 2.57e-03 |
| 8 | 52.16 | 54.84 (1.051x) | 59.90 (1.148x) | 58.79 (1.127x) | 2.26e-03 / 2.26e-03 / 3.90e-03 / 2.44e-03 |
| 16 | 105.12 | 108.68 (1.034x) | 123.26 (1.173x) | 117.80 (1.121x) | 2.28e-03 / 2.28e-03 / 4.93e-03 / 2.38e-03 |

## 7B multimodal MoT decoder, 28 layers, 8 GPUs

| config | step s | tokens/s/GPU | MFU | attention FLOP share | peak GiB |
|---|---|---|---|---|---|
| mm7b_mixed_128k_ring8_python_contig | 5.10 | 3213 | 24.4% | 48% | 14.4 |
| mm7b_mixed_128k_ring8_online_ptrr | 3.73 | 4388 | 33.3% | 48% | 13.4 |
| mm7b_mixed_128k_usp_r4u2_online_ptrr | 3.52 | 4657 | 35.3% | 48% | 13.4 |
| mm7b_mixed_128k_usp_r2u4_online_ptrr | 3.48 | 4706 | 35.7% | 48% | 14.0 |
| mm7b_mixed_128k_usp_r2u4_python_contig | 3.68 | 4452 | 33.8% | 48% | 14.3 |
| mm7b_mixed_256k_ring8_python_contig | 15.18 | 2159 | 24.0% | 64% | 22.5 |
| mm7b_mixed_256k_ring8_online_ptrr | 9.85 | 3326 | 36.9% | 64% | 21.0 |
| mm7b_mixed_256k_usp_r4u2_online_ptrr | 9.39 | 3488 | 38.7% | 64% | 21.0 |
| mm7b_mixed_256k_usp_r2u4_online_ptrr | 9.27 | 3537 | 39.3% | 64% | 21.7 |
| mm7b_mixed_256k_usp_r2u4_python_contig | 9.83 | 3335 | 37.0% | 64% | 22.2 |
| mm7b_mixed_512k_ring8_python_contig | 50.93 | 1287 | 23.6% | 78% | 38.6 |
| mm7b_mixed_512k_ring8_online_ptrr | 29.53 | 2219 | 40.6% | 78% | 35.9 |
| mm7b_mixed_512k_usp_r4u2_online_ptrr | 28.61 | 2291 | 41.9% | 78% | 36.3 |
| mm7b_mixed_512k_usp_r2u4_online_ptrr | 28.10 | 2332 | 42.7% | 78% | 37.1 |
| mm7b_mixed_512k_usp_r2u4_python_contig | 30.43 | 2154 | 39.4% | 78% | 38.2 |
| mm7b_mixed_1024k_ring8_python_contig | exit=1 | | | | |
| mm7b_mixed_1024k_ring8_online_ptrr | exit=1 | | | | |
| mm7b_mixed_1024k_usp_r4u2_online_ptrr | exit=1 | | | | |
| mm7b_mixed_1024k_usp_r2u4_online_ptrr | exit=1 | | | | |
| mm7b_mixed_1024k_usp_r2u4_python_contig | exit=1 | | | | |
| mm7b_mixed_16k_single_flex | 2.19 | 7492 | 33.1% | 10% | 13.4 |
| mm7b_mixed_16k_single_fa4 | 2.09 | 7846 | 34.6% | 10% | 13.4 |
| mm7b_mixed_32k_single_flex | 4.76 | 6880 | 34.2% | 20% | 21.0 |
| mm7b_mixed_32k_single_fa4 | 4.38 | 7473 | 37.1% | 20% | 21.0 |
| mm7b_mixed_64k_single_flex | 11.08 | 5917 | 34.5% | 32% | 35.9 |
| mm7b_mixed_64k_single_fa4 | 9.74 | 6728 | 39.2% | 32% | 35.9 |

## 7B multimodal MoT decoder, 28 layers, 16 GPUs, 2 nodes

| config | step s | tokens/s/GPU | MFU | attention FLOP share | peak GiB |
|---|---|---|---|---|---|
| mm7b_mixed_512k_ring16_python_contig | 30.13 | 1088 | 19.9% | 78% | 20.9 |
| mm7b_mixed_512k_ring16_online_ptrr | 16.25 | 2016 | 36.9% | 78% | 19.3 |
| mm7b_mixed_512k_usp_r8u2_online_ptrr | 15.29 | 2143 | 39.2% | 78% | 19.5 |
| mm7b_mixed_512k_usp_r4u4_online_ptrr | 14.87 | 2204 | 40.4% | 78% | 20.2 |
| mm7b_mixed_512k_usp_r4u4_online_ptrr_nccl4py | 15.03 | 2180 | 39.9% | 78% | 20.4 |
| mm7b_mixed_1024k_ring16_python_contig | 108.83 | 602 | 19.6% | 88% | 37.1 |
| mm7b_mixed_1024k_ring16_online_ptrr | 52.36 | 1252 | 40.8% | 88% | 34.4 |
| mm7b_mixed_1024k_usp_r8u2_online_ptrr | 50.36 | 1301 | 42.4% | 88% | 34.8 |
| mm7b_mixed_1024k_usp_r4u4_online_ptrr | 49.12 | 1334 | 43.5% | 88% | 35.7 |
| mm7b_mixed_1024k_usp_r4u4_online_ptrr_nccl4py | 49.03 | 1337 | 43.5% | 88% | 36.2 |

## Collectives (2 nodes)

```
{"bench": "collectives", "world": 16, "mib": 512, "all_reduce_world_busbw_gbps": 139.1696402996166, "all_to_all_intra_node_busbw_gbps": 310.98865531925543, "all_to_all_inter_node_busbw_gbps": 16.156437185486475, "ring_p2p_inter_node_gbps_per_rank": 20.268017175028515}
```
