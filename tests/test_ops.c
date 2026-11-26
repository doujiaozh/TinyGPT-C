#include "tinygpt.h"

#include <math.h>
#include <stdio.h>

static int failures = 0;

static void check(const char *name, int condition) {
    if (condition) {
        printf("%s PASS\n", name);
    } else {
        printf("%s FAIL\n", name);
        failures++;
    }
}

static int close_enough(float a, float b, float tolerance) {
    return fabsf(a - b) <= tolerance;
}

static void test_matmul(void) {
    float a[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float b[6] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    tgpt_matmul(a, b, c, 2, 3, 2);
    check("matmul_shape", close_enough(c[0], 58.0f, 1e-4f) && close_enough(c[1], 64.0f, 1e-4f) &&
                              close_enough(c[2], 139.0f, 1e-4f) && close_enough(c[3], 154.0f, 1e-4f));
}

static void test_matmul_nt(void) {
    float a[3] = {1.0f, 2.0f, 3.0f};
    float b[6] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    float c[2] = {0.0f, 0.0f};
    tgpt_matmul_nt(a, b, c, 1, 3, 2);
    check("matmul_nt", close_enough(c[0], 1.0f, 1e-4f) && close_enough(c[1], 2.0f, 1e-4f));
}

static void test_softmax(void) {
    float x[3] = {1.0f, 2.0f, 3.0f};
    tgpt_softmax(x, 3);
    float sum = x[0] + x[1] + x[2];
    check("softmax_sum", close_enough(sum, 1.0f, 1e-5f));
    check("softmax_order", x[0] < x[1] && x[1] < x[2]);
    float expected = expf(3.0f - 3.0f) / (expf(1.0f - 3.0f) + expf(2.0f - 3.0f) + expf(3.0f - 3.0f));
    check("softmax_value", close_enough(x[2], expected, 1e-5f));
}

static void test_layernorm(void) {
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float b[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    tgpt_layernorm(out, x, w, b, 4);
    float mean = (out[0] + out[1] + out[2] + out[3]) / 4.0f;
    float second = (out[0] * out[0] + out[1] * out[1] + out[2] * out[2] + out[3] * out[3]) / 4.0f;
    check("layernorm_mean", fabsf(mean) < 1e-5f);
    check("layernorm_variance", fabsf(second - 1.0f) < 1e-3f);
}

static void test_gelu(void) {
    float x[2] = {0.0f, 1.0f};
    tgpt_gelu(x, 2);
    check("gelu_zero", fabsf(x[0]) < 1e-6f);
    check("gelu_one", close_enough(x[1], 0.841192f, 1e-4f));
}

int main(void) {
    test_matmul();
    test_matmul_nt();
    test_softmax();
    test_layernorm();
    test_gelu();
    if (failures == 0) {
        printf("all operator tests passed\n");
        return 0;
    }
    printf("operator tests failed: %d\n", failures);
    return 1;
}
