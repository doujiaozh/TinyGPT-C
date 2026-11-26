#include "tinygpt.h"

#include <math.h>
#include <string.h>

void tgpt_matmul(const float *a, const float *b, float *c, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        float *row = c + (size_t)i * n;
        for (int j = 0; j < n; j++) {
            row[j] = 0.0f;
        }
    }
    for (int i = 0; i < m; i++) {
        const float *arow = a + (size_t)i * k;
        float *crow = c + (size_t)i * n;
        for (int p = 0; p < k; p++) {
            float value = arow[p];
            const float *brow = b + (size_t)p * n;
            for (int j = 0; j < n; j++) {
                crow[j] += value * brow[j];
            }
        }
    }
}

void tgpt_matmul_nt(const float *a, const float *b, float *c, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        const float *arow = a + (size_t)i * k;
        for (int j = 0; j < n; j++) {
            const float *brow = b + (size_t)j * k;
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += arow[p] * brow[p];
            }
            c[(size_t)i * n + j] = sum;
        }
    }
}

void tgpt_softmax(float *x, int n) {
    if (n <= 0) {
        return;
    }
    float maximum = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > maximum) {
            maximum = x[i];
        }
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        float value = expf(x[i] - maximum);
        x[i] = value;
        sum += value;
    }
    if (sum <= 0.0) {
        float uniform = 1.0f / (float)n;
        for (int i = 0; i < n; i++) {
            x[i] = uniform;
        }
        return;
    }
    float inverse = (float)(1.0 / sum);
    for (int i = 0; i < n; i++) {
        x[i] *= inverse;
    }
}

void tgpt_layernorm(float *out, const float *x, const float *w, const float *b, int n) {
    double mean = 0.0;
    for (int i = 0; i < n; i++) {
        mean += x[i];
    }
    mean /= (double)n;
    double variance = 0.0;
    for (int i = 0; i < n; i++) {
        double delta = (double)x[i] - mean;
        variance += delta * delta;
    }
    variance /= (double)n;
    float inverse = (float)(1.0 / sqrt(variance + 1e-5));
    float meanf = (float)mean;
    for (int i = 0; i < n; i++) {
        out[i] = (x[i] - meanf) * inverse * w[i] + b[i];
    }
}

void tgpt_gelu(float *x, int n) {
    const float coefficient = 0.7978845608028654f;
    for (int i = 0; i < n; i++) {
        float value = x[i];
        float cube = value * value * value;
        x[i] = 0.5f * value * (1.0f + tanhf(coefficient * (value + 0.044715f * cube)));
    }
}
