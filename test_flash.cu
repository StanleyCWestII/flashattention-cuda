// Test harness for flashattention.cu. Not part of the kernel.
// Generates Q/K/V for any T and D, builds a CPU reference, compares row by row.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#include "flashattention.cu"

// deterministic filler so runs are reproducible
static void fill(float* p, int n, unsigned seed)
{
    unsigned s = seed;
    for (int i = 0; i < n; ++i)
    {
        s = s * 1664525u + 1013904223u;
        p[i] = ((float)((s >> 8) & 0xFFFF) / 32768.0f) - 1.0f;   // [-1, 1)
    }
}

// ------------------------------------------------------------- cpu reference
// Plain attention, materializes the whole T x T score table. Ground truth.
static void reference(const float* Q, const float* K, const float* V,
                      float* out, int T)
{
    float* S = (float*)malloc((size_t)T * T * sizeof(float));

    for (int i = 0; i < T; ++i)
    {
        for (int j = 0; j < T; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < D; ++k) sum += Q[i*D + k] * K[j*D + k];
            S[i*T + j] = sum / sqrtf(D);
        }

        float m = S[i*T];
        for (int j = 0; j < T; ++j) if (S[i*T + j] > m) m = S[i*T + j];

        float l = 0.0f;
        for (int j = 0; j < T; ++j) l += expf(S[i*T + j] - m);

        for (int k = 0; k < D; ++k) out[i*D + k] = 0.0f;
        for (int j = 0; j < T; ++j)
        {
            float w = expf(S[i*T + j] - m) / l;
            for (int k = 0; k < D; ++k) out[i*D + k] += w * V[j*D + k];
        }
    }
    free(S);
}

// Returns failing row count for one sequence length.
static int run(int T, int verbose)
{
    size_t n = (size_t)T * D, bytes = n * sizeof(float);

    float *hQ = (float*)malloc(bytes), *hK = (float*)malloc(bytes);
    float *hV = (float*)malloc(bytes), *hO = (float*)malloc(bytes);
    float *ref = (float*)malloc(bytes);
    fill(hQ, n, 1); fill(hK, n, 2); fill(hV, n, 3);

    float *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, bytes); cudaMalloc(&dK, bytes);
    cudaMalloc(&dV, bytes); cudaMalloc(&dO, bytes);
    cudaMemcpy(dQ, hQ, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV, bytes, cudaMemcpyHostToDevice);

    flashAttention<<<(T + BR - 1) / BR, BR>>>(dQ, dK, dV, dO, T);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("launch failed: %s\n", cudaGetErrorString(err)); return T; }
    cudaDeviceSynchronize();

    cudaMemcpy(hO, dO, bytes, cudaMemcpyDeviceToHost);
    reference(hQ, hK, hV, ref, T);

    int fails = 0;
    float worstAll = 0.0f;
    for (int i = 0; i < T; ++i)
    {
        float worst = 0.0f;
        for (int k = 0; k < D; ++k)
        {
            float d = fabsf(hO[i*D + k] - ref[i*D + k]);
            if (d > worst) worst = d;
        }
        if (worst > worstAll) worstAll = worst;
        if (!(worst < 1e-4f)) ++fails;

        if (verbose && i < 4)
        {
            printf("row %-4d gpu[%7.4f %7.4f ...]  cpu[%7.4f %7.4f ...]  %.2e  %s\n",
                   i, hO[i*D], hO[i*D+1], ref[i*D], ref[i*D+1], worst,
                   worst < 1e-4f ? "ok" : "FAIL");
        }
    }
    if (verbose) printf("worst diff over all %d rows: %.2e\n", T, worstAll);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    free(hQ); free(hK); free(hV); free(hO); free(ref);
    return fails;
}

int main()
{
    printf("BR=%d  BC=%d  D=%d\n", BR, BC, D);
    printf("shared/block = %d bytes\n\n", (int)(2 * BC * D * sizeof(float)));

    // sizes chosen to straddle both boundaries: BR (query blocks), BC (key tiles)
    int sizes[] = {1, 2, 31, 32, 33, 63, 64, 65, 96, 127, 128, 129, 200, 256, 384, 512};
    int bad = 0;

    printf("T     blocks  lastblk  keytiles  result\n");
    for (unsigned s = 0; s < sizeof(sizes)/sizeof(int); ++s)
    {
        int T = sizes[s];
        int fails = run(T, 0);
        if (fails) ++bad;
        printf("%-5d %-7d %-8d %-9d %s\n", T, (T + BR - 1)/BR,
               T - ((T + BR - 1)/BR - 1) * BR, (T + BC - 1)/BC,
               fails ? "FAIL" : "ok");
    }

    printf("\n--- detail at T=200 ---\n");
    run(200, 1);

    printf("\n%s\n", bad ? "BOUNDS BROKEN" : "all sizes match");
    return bad != 0;
}
