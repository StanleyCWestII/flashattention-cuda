# flashattention-cuda

A FlashAttention forward pass written from scratch in CUDA for the RTX 4090.
Tiled keys and values, online softmax, no materialized attention matrix.

This is v1. It is correct and it is fast against a naive baseline, but it is
early: see section 6 before reading anything into the speedup number.

## 1. Overview

| File | What it is |
|---|---|
| `flash_cpu.cpp` | CPU attention, written out step by step. The reference and the explanation of the math. |
| `basekernel.cu` | Naive GPU attention. Materializes the full T x T score matrix in global memory. |
| `flashattention.cu` | The kernel. Tiled, fused, online softmax, never writes S. |
| `test_flash.cu` | Correctness harness. Not part of the kernel. |
| `bench_flash.cu` | Benchmark harness. Not part of the kernel. |

## 2. Quick Start

Requires CUDA and an sm_89 card. The arch is set in the Makefile.

```
make test     # correctness across 16 sequence lengths
make bench    # naive vs flash at T = 512 to 4096
```

## 3. The Idea

Standard attention computes S = QK^T / sqrt(D), softmaxes it, then multiplies by
V. The problem is S: it is T x T, and at T = 4096 that is 64 MB written to
global memory and read back, for a result that is thrown away immediately.

FlashAttention does the same math in a different order:

1. **Tile the keys and values.** Take one block of BC keys at a time and compute
   only those scores for the query rows this block owns.
2. **Consume the block immediately.** Fold the scores into a running output
   accumulator and discard them. S is never stored.
3. **Online softmax.** You cannot normalize without the row max, and you do not
   have it yet. Carry two scalars per row: `m`, the largest score seen so far,
   and `l`, the running sum of exp. Normalize against the current `m`. When a
   later block contains a bigger score, `m` was wrong, so correct all the
   accumulated work with one multiply:
   `l = l * correction + new`, `O_acc = O_acc * correction + new`.
4. **Normalize once, at the end.** Divide `O_acc` by the final `l`.

The result is identical to standard attention to within float rounding. Worst
observed absolute error against the CPU reference is 1.4e-07.

## 4. Configuration and Shape

```
D  = 64    head dimension
BR = 64    query rows per block, one thread per row
BC = 32    keys and values staged in shared memory per trip
```

One block owns 64 query rows. Each thread owns exactly one row, holds that
row's Q vector and its 64-element output accumulator in registers, and walks
the whole key sequence in tiles of 32. Shared memory holds `Kds[32][64]` and
`Vds[32][64]`, which is 16 KB per block.

`ptxas` reports 252 registers per thread, 0 bytes spilled, 16384 bytes of
shared memory. At 64 threads per block that is 4 blocks resident per SM, or
about 17% occupancy. The kernel is register-limited, and that low occupancy is
the main thing standing between v1 and v2.

## 5. Results

Measured on an RTX 4090, CUDA 13.3, driver 610.57.04. Both kernels are checked
against the CPU reference before anything is timed.

Correctness at T = 512, worst absolute error:

| Kernel | Error |
|---|---|
| `flashattention.cu` | 1.416e-07 |
| `basekernel.cu` | 5.588e-08 |

Throughput:

| T | naive ms | flash ms | speedup | S matrix the naive kernel writes | flash GFLOP/s |
|---|---|---|---|---|---|
| 512 | 14.871 | 0.121 | 123.07x | 1.0 MB | 555.4 |
| 1024 | 29.828 | 0.235 | 126.79x | 4.0 MB | 1141.0 |
| 2048 | 59.766 | 0.462 | 129.41x | 16.0 MB | 2325.0 |
| 4096 | 120.078 | 0.921 | 130.44x | 64.0 MB | 4665.5 |

Correctness sweep: 16 sequence lengths from T = 1 to T = 512, chosen to land on
and around every tile boundary (1, 2, 31, 32, 33, 63, 64, 65, 96, 127, 128,
129, 200, 256, 384, 512). All match the CPU reference.

## 6. Read The Speedup Honestly

**The 130x is against my own naive kernel, not against a real implementation.**
`basekernel.cu` is deliberately unoptimized: one thread per query row, no shared
memory, and a full T x T round trip through global memory. Beating it by two
orders of magnitude demonstrates that the fusion works. It does not demonstrate
that this kernel is fast in absolute terms.

The absolute number is the one to watch. At T = 4096 the kernel sustains about
4.7 TFLOP/s. The 4090 does roughly 83 TFLOP/s of FP32, so this is around 6% of
peak. Real FlashAttention implementations also use tensor cores and half
precision, which this does not touch at all.

So: the algorithm is right, the memory traffic win is real, and the kernel
itself has most of its performance still ahead of it.

## 7. Limitations and What Is Next

- **No 2D tiling.** One thread per query row is the single biggest limiter. Each
  thread carries a 64-float accumulator, which is what drives registers to 252
  and occupancy to 17%. Giving each thread a patch of rows and columns instead
  is v2.
- **FP32 only, no tensor cores.**
- **D is fixed at 64** by `#define`, not a runtime parameter.
- **Forward pass only.** No backward pass, so this cannot train anything.
- **No causal masking**, no dropout, no attention bias, no multi-head batching.
  Single head, full bidirectional attention.
- **No `cudaMemcpyAsync` or stream overlap**; timing is kernel time only.
- Arch is hardcoded to sm_89 in the Makefile.

## 8. Authorship and References

| Mine | Tooling, built with AI assistance |
|---|---|
| flash_cpu.cpp | test_flash.cu |
| basekernel.cu | bench_flash.cu |
| flashattention.cu | Makefile |
| | This README |

Every kernel in this repo is mine, written line by line. The harnesses grade
them and time them, and were built with AI assistance so that the kernels
themselves stayed my work.

- Dao, Fu, Ermon, Rudra, and Ré, *FlashAttention: Fast and Memory-Efficient
  Exact Attention with IO-Awareness* (2022).
- Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work
  Partitioning* (2023).
- Hwu, Kirk, and El Hajj, *Programming Massively Parallel Processors*, 5th edition.
