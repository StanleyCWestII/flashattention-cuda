#include <cmath>

// What is Attention?
//
// You have N tokens. Each one starts with a vector. Multiply each token by three
// learned weight matrices to get a query, a key, and a value. Stack them:
// Q = x * W_Q (query)
// K = x * W_K (key)
// V = x * W_V (value)
//
// There are four steps to Attention:
//
// Step 1: Score every pair.
// Dot query i against key j for all i, j. That is Q K^T.
// S = Q K^T    N x N
// S[i][j] = how strongly token i should listen to token j.
//
// Step 2: Scale.
// Divide S by the square root of d. Purely to stop the numbers growing as d gets
// bigger.
//
// Step 3: Softmax each row.
// Turn one row of raw scores into weights that are positive and sum to 1. Three
// passes over the row:
// a. find the max of the row
// b. sum exp(score - max) over the row
// c. each weight = exp(score - max) / sum
// Result is P, also N x N.
//
// Step 4: Blend the values.
// Token i's output is its weights applied to the value vectors.
// O = P V    N x d

#define T 3 // number of tokens
#define D 2 // numbers per vector

void attention()
{
    // Q, K, and V are input matrices. S keeps track of how much token i should listen
    // to token j, and O is the output matrix.
    float Q[T][D] = { {1,0}, {0, 1}, {1, 1} };
    float K[T][D] = { {1, 2}, {2, 3}, {0, 3} };
    float V[T][D] = { {3, 4}, {1, 3}, {0, 2} };
    float S[T][T] = {};
    float O[T][D] = {};

    // iterates over tokens
    for (int i = 0; i < T; ++i)
    {
        // iterates over tokens
        for (int j = 0; j < T; ++j)
        {
            // accumulates the computations
            float sum = 0.0f;
            // iterates over numbers per vector
            for (int k = 0; k < D; ++k)
            {
                sum += Q[i][k] * K[j][k];
            }
            // fills the S slot with the sum
            S[i][j] = sum / sqrtf(D);
        }

        // finds the maximum value
        float max = S[i][0];
        for (int j = 0; j < T; ++j)
        {
            float maxtmp = S[i][j];
            if (maxtmp > max)
            {
                max = maxtmp;
            }
        }

        float l = 0.0f;
        for (int j = 0; j < T; ++j)
        {
            l += (expf(S[i][j] - max));
        }

        for (int j = 0; j < T; ++j)
        {
            S[i][j] = expf(S[i][j] - max) / l;
        }

        for (int j = 0; j < T; ++j)
        {
            for (int k = 0; k < D; ++k)
            {
                O[i][k] += S[i][j] * V[j][k];
            }
        }
    }
}

int main()
{
    attention();
    return 0;
}
