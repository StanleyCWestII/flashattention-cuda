#include <cuda_runtime.h>
#include <cmath>

// What is FlashAttention?
//
// FlashAttention is similar to Attention. The math, from steps 1-4, are identical.
// The thing that changes is when each step runs, and what gets stored.
//
// Change 1: Tile the keys and values.
// Instead of computing all of S, take the first block of keys. Compute only those
// scores for the query row you are working on.
//
// Change 2: Consume the block immediately, then throw it away.
// Do not store the scores. Fold them straight into a running output accumular O_acc
// and discard.
//
// Change 3: Online softmax, to make change 2 legal.
// You cannot normalize without the row max, and you do not have it yet. So carry
// two extra scalars per row:
// m = biggest score seen so far
// l = running sum of exp
// Normalize against m provisionally. When a later block contains a bigger score,
// m was wrong, so fix all the accumulated work with one multiply:
// l = l * correction + new terms
// O_acc = O_acc * correction + new terms
//
// Change 4: Normalize once, at the end.
// After the last block, divide O_acc by the final l. That is your output row.

#define D (64) // numbers in one vector
#define BR (64) // threads in the block, so query rows per block
#define BC (32) // keys staged in shared per trip
#define FIT (BC * D) // floats in one tile

__global__

// Q, K, and V are input matrices. O is the output matrix
// Q holds query vectors
// K holds key vectors
// V holds value vectors
// T is total tokens, so keys total and query rows total
void flashAttention(float* Q, float* K, float* V, float* O, int T)
{
    __shared__ float Kds[BC][D]; // shared memory for K of BC x D size
    __shared__ float Vds[BC][D]; // shared memory for V of BC x D size

    int bx = blockIdx.x; // which query block this block owns
    int tx = threadIdx.x; // which query row inside the block. also the address

    float max = -INFINITY; // records the highest value measured
    float l = 0.0f; // stores the sum of e^(score - m) scores

    float QS[D]; // shared memory float for Q matrix

    // loads Q into shared memory array
    if ((bx*BR + tx) < T)
    {
        for (int i = 0; i < D; ++i)
        {
            QS[i] = Q[(bx * BR + tx) * D + i];
        }
    }

    float acc[D] {0.0f}; // stores the two outputs
    float score[BC]; // stores scores

    // iterates over all batches
    for (int i = 0; i < ((T + BC - 1) / BC); ++i)
    {
        // loads values into shared memory, strides by thread count and starts at
        // the specific thread
        for (int j = tx; j < FIT; j += BR)
        {
            int row = j / D;
            int col = j % D;

            if ((i*BC + row) < T)
            {
                Kds[row][col] = K[(i*BC + row) * D + col]; // key vectors
                Vds[row][col] = V[(i*BC + row) * D + col]; // value vectors
            }
        }

        __syncthreads();

        for (int j = 0; j < BC; ++j)
        {
            float sum = 0.0f;
                for (int k = 0; k < D; ++k)
                {
                    sum += QS[k] * Kds[j][k]; // scores are formed from Q * K
                }

            score[j] = sum / sqrtf(D); // stores sum in score container
            if ((i * BC + j) >= T)
            {
                score[j] = -INFINITY;
            }
        }

        float newmax = max; // assigns the maximum value seen to newmax
        for (int j = 0; j < BC; ++j)
        {
            newmax = fmaxf(newmax, score[j]); // compares the max to new values, picks the higher one
        }

        float correction = expf(max - newmax);
        l = l * correction;
        max = newmax;
        for (int j = 0; j < D; ++j)
        {
            acc[j] = acc[j] * correction;
        }

        for (int j = 0; j < BC; ++j)
        {
            score[j] = expf(score[j] - newmax); // calculate weight of each key
            l += score[j];

            for (int k = 0; k < D; ++k)
            {
                acc[k] += score[j] * Vds[j][k];
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < D; ++i)
    {
        if ((bx*BR + tx) < T)
        {
            O[(bx*BR + tx) * D + i] = acc[i] / l;
        }
    }
}
