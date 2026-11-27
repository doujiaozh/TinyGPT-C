#include "userlib.h"

typedef float v8f __attribute__((vector_size(32)));

static const char simd_pass[] = "tinygpt zos: SIMD PASS\n";
static const char scalar_pass[] = "tinygpt zos: SIMD unavailable, scalar PASS\n";
static const char failure[] = "tinygpt zos: FAIL\n";

static volatile float one_value = 1.0f;
static volatile float two_value = 2.0f;

static int simd_available(void) {
    unsigned int eax = 1;
    unsigned int ebx = 0;
    unsigned int ecx = 0;
    unsigned int edx = 0;
    __asm__ volatile("cpuid" : "+a"(eax), "=b"(ebx), "+c"(ecx), "=d"(edx));
    int osxsave = (ecx >> 27) & 1;
    int avx = (ecx >> 28) & 1;
    int fma = (ecx >> 12) & 1;
    if (!(osxsave && avx && fma)) {
        return 0;
    }
    eax = 7;
    ecx = 0;
    __asm__ volatile("cpuid" : "+a"(eax), "=b"(ebx), "+c"(ecx), "=d"(edx));
    if (((ebx >> 5) & 1) == 0) {
        return 0;
    }
    unsigned int xcr0_low = 0;
    unsigned int xcr0_high = 0;
    __asm__ volatile("xgetbv" : "=a"(xcr0_low), "=d"(xcr0_high) : "c"(0));
    (void)xcr0_high;
    return (xcr0_low & 6) == 6;
}

static int scalar_test(void) {
    float a[16];
    float b[16];
    for (int i = 0; i < 16; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
    float sum = 0.0f;
    for (int i = 0; i < 16; i++) {
        sum += a[i] * b[i];
    }
    return sum == 32.0f;
}

__attribute__((target("avx2,fma"), noinline)) static int simd_test(void) {
    float a[16] __attribute__((aligned(32)));
    float b[16] __attribute__((aligned(32)));
    for (int i = 0; i < 16; i++) {
        a[i] = one_value;
        b[i] = two_value;
    }
    v8f va0 = *(const v8f *)a;
    v8f vb0 = *(const v8f *)b;
    v8f va1 = *(const v8f *)(a + 8);
    v8f vb1 = *(const v8f *)(b + 8);
    v8f total = va0 * vb0 + va1 * vb1;
    float lanes[8] __attribute__((aligned(32)));
    *(v8f *)lanes = total;
    float sum = lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];
    return sum == 32.0f;
}

__attribute__((noreturn)) void _start(void) {
    int available = simd_available();
    int passed = available ? simd_test() : scalar_test();
    if (!passed) {
        zwrite(failure);
        zexit(1);
    }
    if (available) {
        zwrite(simd_pass);
    } else {
        zwrite(scalar_pass);
    }
    zexit(0);
}
