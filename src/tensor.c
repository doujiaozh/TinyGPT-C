#include "tinygpt.h"

#include <math.h>
#include <omp.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define TGPT_PARALLEL_WORK 1500000

static float half_to_float(uint16_t value) {
    uint32_t sign = ((uint32_t)value & 0x8000u) << 16;
    uint32_t exponent = ((uint32_t)value >> 10) & 0x1Fu;
    uint32_t mantissa = (uint32_t)value & 0x3FFu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                shift++;
            }
            mantissa &= 0x3FFu;
            uint32_t new_exponent = 113u - (uint32_t)shift;
            bits = sign | (new_exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    float result = 0.0f;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

float tgpt_half_to_float(uint16_t value) {
    return half_to_float(value);
}

#if !defined(__AVX2__)
static void gemv_scalar_range(const float *a, const float *b, float *c, int k, int n, int begin, int end) {
    for (int p = 0; p < k; p++) {
        float value = a[p];
        const float *row = b + (size_t)p * n;
        for (int j = begin; j < end; j++) {
            c[j] += value * row[j];
        }
    }
}
#endif

#if defined(__AVX2__)
static void gemv_avx2_range(const float *a, const float *b, float *c, int k, int n, int begin, int end) {
    for (int p = 0; p < k; p++) {
        __m256 value = _mm256_set1_ps(a[p]);
        const float *row = b + (size_t)p * n;
        int j = begin;
        for (; j + 32 <= end; j += 32) {
            __m256 c0 = _mm256_loadu_ps(c + j);
            __m256 c1 = _mm256_loadu_ps(c + j + 8);
            __m256 c2 = _mm256_loadu_ps(c + j + 16);
            __m256 c3 = _mm256_loadu_ps(c + j + 24);
            c0 = _mm256_fmadd_ps(value, _mm256_loadu_ps(row + j), c0);
            c1 = _mm256_fmadd_ps(value, _mm256_loadu_ps(row + j + 8), c1);
            c2 = _mm256_fmadd_ps(value, _mm256_loadu_ps(row + j + 16), c2);
            c3 = _mm256_fmadd_ps(value, _mm256_loadu_ps(row + j + 24), c3);
            _mm256_storeu_ps(c + j, c0);
            _mm256_storeu_ps(c + j + 8, c1);
            _mm256_storeu_ps(c + j + 16, c2);
            _mm256_storeu_ps(c + j + 24, c3);
        }
        for (; j + 8 <= end; j += 8) {
            __m256 c0 = _mm256_loadu_ps(c + j);
            c0 = _mm256_fmadd_ps(value, _mm256_loadu_ps(row + j), c0);
            _mm256_storeu_ps(c + j, c0);
        }
        for (; j < end; j++) {
            c[j] += a[p] * row[j];
        }
    }
}
#endif

static void gemv_range(const float *a, const float *b, float *c, int k, int n, int begin, int end) {
#if defined(__AVX2__)
    gemv_avx2_range(a, b, c, k, n, begin, end);
#else
    (void)n;
    gemv_scalar_range(a, b, c, k, n, begin, end);
#endif
}

#if !defined(__AVX2__)
static float dot_scalar(const float *a, const float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}
#endif

#if defined(__AVX2__)
static float dot_avx2(const float *a, const float *b, int n) {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    __m256 total = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    __m128 low = _mm256_castps256_ps128(total);
    __m128 high = _mm256_extractf128_ps(total, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float result = _mm_cvtss_f32(sum);
    for (; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}
#endif

static float dot_product(const float *a, const float *b, int n) {
#if defined(__AVX2__)
    return dot_avx2(a, b, n);
#else
    return dot_scalar(a, b, n);
#endif
}

#if !defined(__AVX2__) || !defined(__F16C__)
static float dot_f16_scalar(const float *a, const uint16_t *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * half_to_float(b[i]);
    }
    return sum;
}
#endif

#if defined(__AVX2__) && defined(__F16C__)

static float dot_f16_avx2(const float *a, const uint16_t *b, int n) {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(b + i))), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),
                              _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(b + i + 8))), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16),
                              _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(b + i + 16))), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24),
                              _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(b + i + 24))), s3);
    }
    __m256 total = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    __m128 low = _mm256_castps256_ps128(total);
    __m128 high = _mm256_extractf128_ps(total, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float result = _mm_cvtss_f32(sum);
    for (; i < n; i++) {
        result += a[i] * half_to_float(b[i]);
    }
    return result;
}
#endif

static float dot_f16(const float *a, const uint16_t *b, int n) {
#if defined(__AVX2__) && defined(__F16C__)
    return dot_f16_avx2(a, b, n);
#else
    return dot_f16_scalar(a, b, n);
#endif
}

void tgpt_matmul(const float *a, const float *b, float *c, int m, int k, int n) {
    if (m == 1 && k > 0 && n > 0) {
        memset(c, 0, (size_t)n * sizeof(float));
        int threads = omp_get_max_threads();
        int parallel = threads > 1 && (long long)n * k >= TGPT_PARALLEL_WORK;
        if (parallel) {
#pragma omp parallel
            {
                int thread = omp_get_thread_num();
                int count = omp_get_num_threads();
                int chunk = (n + count - 1) / count;
                chunk = (chunk + 31) & ~31;
                int begin = thread * chunk;
                int end = begin + chunk;
                if (end > n) {
                    end = n;
                }
                if (begin < end) {
                    gemv_range(a, b, c, k, n, begin, end);
                }
            }
        } else {
            gemv_range(a, b, c, k, n, 0, n);
        }
        return;
    }
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
    if (m == 1 && k > 0 && n > 0) {
        int threads = omp_get_max_threads();
        int parallel = threads > 1 && (long long)n * k >= TGPT_PARALLEL_WORK;
        if (parallel) {
#pragma omp parallel for schedule(static)
            for (int row = 0; row < n; row++) {
                c[row] = dot_product(a, b + (size_t)row * k, k);
            }
        } else {
            for (int row = 0; row < n; row++) {
                c[row] = dot_product(a, b + (size_t)row * k, k);
            }
        }
        return;
    }
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

void tgpt_matmul_f16(const float *a, const uint16_t *b, float *c, int m, int k, int n) {
    if (m == 1 && k > 0 && n > 0) {
        int threads = omp_get_max_threads();
        int parallel = threads > 1 && (long long)n * k >= TGPT_PARALLEL_WORK;
        if (parallel) {
#pragma omp parallel for schedule(static)
            for (int row = 0; row < n; row++) {
                c[row] = dot_f16(a, b + (size_t)row * k, k);
            }
        } else {
            for (int row = 0; row < n; row++) {
                c[row] = dot_f16(a, b + (size_t)row * k, k);
            }
        }
        return;
    }
    for (int i = 0; i < m; i++) {
        const float *arow = a + (size_t)i * k;
        for (int j = 0; j < n; j++) {
            const uint16_t *brow = b + (size_t)j * k;
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += arow[p] * half_to_float(brow[p]);
            }
            c[(size_t)i * n + j] = sum;
        }
    }
}

void tgpt_matmul_nt_f16(const float *a, const uint16_t *b, float *c, int m, int k, int n) {
    tgpt_matmul_f16(a, b, c, m, k, n);
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
