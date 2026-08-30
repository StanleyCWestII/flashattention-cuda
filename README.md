# flashattention-cuda

A FlashAttention kernel written from scratch in CUDA for the RTX 4090. It incorporates tiled keys and values and online softmax with no materialized attention matrix.

This is a personal project I wrote to understand what FlashAttention is and sharpen my memory management skills. Because of this, it is important to note it is fast against a naive baseline. See section 6 for more details.

## 1. Overview

| File | What It Is |
|---|---|
| `flash_cpu.cpp` | CPU attention, written out step by step. The reference point and explains the math behind the kernel |
| `basekernel.cu`| The naive GPU attention. Materializes the full T x T score matrix in global memory |
|`flashattention.cu` | The actual kernel. Tiled, fused, online softmax |
|`test_flash.cu`| Tests the correctness of the FlashAttention kernel |
| `bench_flash.cu` | The benchmark for the kernel |

## 2. Quick Start

```
make test       # correctness across 16 sequence lengths
make bench      # naive vs flash at T = 512 to 4096
```

Note: this requires CUDA and an sm_89 card to run.

## 3. The Idea

Standard attention computes S = QK^T / sqrt(D), softmaxes it, and then multiplies by V. The issue is that S is T x T, and at T = 4096, that is 64 MB written to global memory and read back, for a result that is thrown away.

FlashAttention does the same math but in a different order:

1. **Tile the keys and values.** Take one block of BC keys at a time and compute only those scores for the query rows this block owns.

2. **Consume the block immediately.** Accumulate the scores into a running output and discard them. S is never stored.

3. **Online softmax.** You cannot normalize without the row max, and you do not have it yet. Carry two scalars per row: `m`, the largest score seen so far, and `l`, the running sum of exp. Normalize against the current `m`. When a later block contains a bigger score, `m` was wrong, so correct all the accumulated work with one multiply:
`l = l * correction + new`, `O_acc = O_acc * correction + new`

4. **Normalize once, at the end.** Divide `O_acc` by the final `l`.

The result is identical to standard attention to within float rounding. Worst observed absolute value error against the CPU reference is 1.4e-07.

## 4. Configuration and Shape 

```
D = 64      # head dimension
BR = 64     # query rows per block, one thread per row
BC = 32     # keys and values staged in shared memory per trip
```

One block owns 64 query rows. Each thread owns exactly one row, holds that row's Q vector and its 64-element output accumulator in registers, and walks the whole key sequence in tiles of 32. Shared memory holds `Kds[32][64]` and `Vds[32][64]`, which is 16KB per block.

`ptxas` reports 252 registers per thread, 0 bytes spilled, 16384 bytes of shared memory. At 64 threads per block that is 4 blocks resident per SM, or about 17% occupancy. The kernel is register-limited.

## 5. Results

Measured on an RTX 4090, CUDA 13.3, driver 610.57.04. Both kernels are checked against the CPU reference before anything is timed. 

Correctness at T = 512, worst absolute error:

| Kernel | Error |
|---|---|
| `flashattention.cu` | 1.416e-07 |
| `basekernel.cu` | 5.588e-08 |

Throughput:

| T | Naive ms | Flash ms | Speedup | S Matrix The Naive Kernel Writes | Flash GFLOP/s |
|---|---|---|---|---|---|
| 512 | 14.871 | 0.121 | 123.07x | 1.0MB | 555.4 |
| 1024 | 29.828 | 0.235 | 126.79x | 4.0MB | 1141.0 |
| 2048 | 59.766 | 0.462 | 129.41x | 16.0MB | 2325.0 |
| 4096 | 120.078 | 0.921 | 130.44x | 64.0MB | 4665.5 |

Correctness sweep: 16 sequence lengths from T = 1 to T = 512, chosen to land on and around every tile boundary (1, 2, 31, 32, 33, 63, 64, 65, 96, 127, 128, 129, 200, 256, 384, 512). All match the CPU reference.

## 6. Read The Speedup Honestly

The 130x is against my own naive kernel, not against a real implementation. The naive kernel is deliberately unoptimized: one thread per query row, no shared memory, and a full T x T round trip through global memory. The speedup is real, but there is still a lot left on the table.

At T = 4096, the kernel sustains ~4.7 TFLOP/s. The 4090 performs ~83 TFLOP/s, so `flashattention.cu` sits at ~6% of peak. Real FlashAttention implements tensor cores and half precision, which I did not touch. These techniques are outside of my current knowledge base, but I do want to learn them soon.

## 7. Limitations

- **No 2D tiling.** One thread per query row is the single biggest limiter. Each thread carries a 64-float accumulator, which is what drives registers to 252 and occupancy to 17%. 
- **FP32 only, no tensor cores.**
- **D is fixed at 64** by `#define`, not a runtime parameter.
- **Forward pass only.** No backward pass, so this cannot train anything.
- **No causal masking**, no dropout, no attention bias, no multi-head batching. Single head, full bidirectional attention.
- **No `cudaMemcpyAsync` or stream overlap.** Timing is kernel time only.
- Arch is hardcoded to sm_89 in the Makefile.

## 8. Authorship and References

| Mine | AI-Assisted: Tooling, Instrumentation, Measurement |
|---|---|
| `flash_cpu.cpp` | `test_flash.cu` |
| `basekernel.cu` | `bench_flash.cu` |
| `flashattention.cu` | `Makefile` |
| | This README |

Every kernel in this repo is mine, completely. The tests and benchmarks were built with AI assistance.

- Hwu, Kirk, and El Hajj, *Programming Massively Parallel Processors*, 5th Edition.
- Dao, Fu, Ermon, Rudra, and Ré, *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022).
- Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (2023).
