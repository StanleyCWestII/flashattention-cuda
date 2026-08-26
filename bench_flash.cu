// Benchmark harness. Not part of either kernel.
// Times flashattention.cu against basekernel.cu on identical inputs,
// and checks both against a CPU reference.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#include "flashattention.cu"   // defines D, BR, BC
#include "basekernel.cu"       // same D

#define BASE_TPB 256           // threads per block for the naive kernel

static void fill(float* p, size_t n, unsigned seed)
{
    unsigned s = seed;
    for (size_t i = 0; i < n; ++i)
    {
        s = s * 1664525u + 1013904223u;
        p[i] = ((float)((s >> 8) & 0xFFFF) / 32768.0f) - 1.0f;
    }
}

static void reference(const float* Q, const float* K, const float* V, float* out, int T)
{
    float* row = (float*)malloc((size_t)T * sizeof(float));
    for (int i = 0; i < T; ++i)
    {
        float m = -INFINITY;
        for (int j = 0; j < T; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < D; ++k) sum += Q[(size_t)i*D + k] * K[(size_t)j*D + k];
            row[j] = sum / sqrtf(D);
            if (row[j] > m) m = row[j];
        }
        float l = 0.0f;
        for (int j = 0; j < T; ++j) l += expf(row[j] - m);
        for (int k = 0; k < D; ++k) out[(size_t)i*D + k] = 0.0f;
        for (int j = 0; j < T; ++j)
        {
            float w = expf(row[j] - m) / l;
            for (int k = 0; k < D; ++k) out[(size_t)i*D + k] += w * V[(size_t)j*D + k];
        }
    }
    free(row);
}

static float worstDiff(const float* a, const float* b, size_t n)
{
    float w = 0.0f;
    for (size_t i = 0; i < n; ++i)
    {
        float d = fabsf(a[i] - b[i]);
        if (d > w) w = d;
    }
    return w;
}

// median-of-reps timing, in milliseconds
static float timeKernel(void (*launch)(int), int T, int reps)
{
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 3; ++i) launch(T);          // warmup
    cudaDeviceSynchronize();

    float best = 1e30f;
    for (int r = 0; r < reps; ++r)
    {
        cudaEventRecord(a);
        launch(T);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    return best;
}

static float *dQ, *dK, *dV, *dS, *dOf, *dOb;

static void launchFlash(int T) { flashAttention<<<(T + BR - 1)/BR, BR>>>(dQ, dK, dV, dOf, T); }
static void launchBase (int T) { gpuKernel<<<(T + BASE_TPB - 1)/BASE_TPB, BASE_TPB>>>(dQ, dK, dV, dS, dOb, T); }

int main()
{
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("%s   BR=%d BC=%d D=%d\n\n", prop.name, BR, BC, D);

    int sizes[] = {512, 1024, 2048, 4096};
    const int NS = sizeof(sizes)/sizeof(int);

    // ---- correctness at the smallest size, against the CPU ----
    {
        int T = sizes[0];
        size_t n = (size_t)T * D;
        float *hQ=(float*)malloc(n*4), *hK=(float*)malloc(n*4), *hV=(float*)malloc(n*4);
        float *hF=(float*)malloc(n*4), *hB=(float*)malloc(n*4), *ref=(float*)malloc(n*4);
        fill(hQ,n,1); fill(hK,n,2); fill(hV,n,3);

        cudaMalloc(&dQ,n*4); cudaMalloc(&dK,n*4); cudaMalloc(&dV,n*4);
        cudaMalloc(&dOf,n*4); cudaMalloc(&dOb,n*4);
        cudaMalloc(&dS,(size_t)T*T*4);
        cudaMemcpy(dQ,hQ,n*4,cudaMemcpyHostToDevice);
        cudaMemcpy(dK,hK,n*4,cudaMemcpyHostToDevice);
        cudaMemcpy(dV,hV,n*4,cudaMemcpyHostToDevice);
        cudaMemset(dOf,0,n*4); cudaMemset(dOb,0,n*4);

        launchFlash(T); launchBase(T);
        cudaDeviceSynchronize();
        cudaMemcpy(hF,dOf,n*4,cudaMemcpyDeviceToHost);
        cudaMemcpy(hB,dOb,n*4,cudaMemcpyDeviceToHost);
        reference(hQ,hK,hV,ref,T);

        printf("correctness at T=%d, worst absolute error vs CPU\n", T);
        printf("  flashattention.cu   %.3e   %s\n", worstDiff(hF,ref,n),
               worstDiff(hF,ref,n) < 1e-4f ? "ok" : "FAIL");
        printf("  basekernel.cu       %.3e   %s\n\n", worstDiff(hB,ref,n),
               worstDiff(hB,ref,n) < 1e-4f ? "ok" : "FAIL");

        cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dS);cudaFree(dOf);cudaFree(dOb);
        free(hQ);free(hK);free(hV);free(hF);free(hB);free(ref);
    }

    // ---- timing ----
    printf("%-6s %10s %10s %9s %10s %10s\n",
           "T", "naive ms", "flash ms", "speedup", "S size", "flash GF/s");
    for (int s = 0; s < NS; ++s)
    {
        int T = sizes[s];
        size_t n = (size_t)T * D, sBytes = (size_t)T * T * 4;

        float* h=(float*)malloc(n*4);
        cudaMalloc(&dQ,n*4); cudaMalloc(&dK,n*4); cudaMalloc(&dV,n*4);
        cudaMalloc(&dOf,n*4); cudaMalloc(&dOb,n*4);
        if (cudaMalloc(&dS,sBytes) != cudaSuccess) { printf("T=%d: S alloc failed\n", T); return 1; }

        fill(h,n,1); cudaMemcpy(dQ,h,n*4,cudaMemcpyHostToDevice);
        fill(h,n,2); cudaMemcpy(dK,h,n*4,cudaMemcpyHostToDevice);
        fill(h,n,3); cudaMemcpy(dV,h,n*4,cudaMemcpyHostToDevice);
        cudaMemset(dOf,0,n*4); cudaMemset(dOb,0,n*4);

        float mb = timeKernel(launchBase,  T, 20);
        float mf = timeKernel(launchFlash, T, 20);

        double flops = 4.0 * (double)T * T * D;          // QK^T and PV
        printf("%-6d %10.3f %10.3f %8.2fx %8.1f MB %10.1f\n",
               T, mb, mf, mb/mf, sBytes/1048576.0, flops/(mf*1e-3)/1e9);

        cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dS);cudaFree(dOf);cudaFree(dOb);
        free(h);
    }
    return 0;
}
