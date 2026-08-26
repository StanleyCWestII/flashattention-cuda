#include <cuda_runtime.h>
#include <cmath>

#define D (64)

__global__

void gpuKernel(float* Q, float* K, float* V, float* S, float* O, int T)
{
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread >= T)
    {
        return;
    }

    for (int i = 0; i < T; ++i)
    {
        float sum = 0.0f;
        for (int j = 0; j < D; ++j)
        {
            sum += Q[thread * D + j] * K[i * D + j];
        }

        S[thread * T + i] = sum / sqrtf(D);
    }

    float max = S[thread * T];
    for (int i = 0; i < T; ++i)
    {
        float maxtmp = S[thread * T + i];
        if (maxtmp > max)
        {
            max = maxtmp;
        }
    }

    float l = 0.0f;
    for (int i = 0; i < T; ++i)
    {
        l += expf(S[thread * T + i] - max);
    }

    for (int i = 0; i < T; ++i)
    {
        S[thread * T + i] = expf(S[thread * T + i] - max) / l;
    }

    for (int i = 0; i < T; ++i)
    {
        for (int j = 0; j < D; ++j)
        {
            O[thread * D + j] += S[thread * T + i] * V[i * D + j];
        }
    }
}
