#include "tinygpt.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

static double now(void) {
    return omp_get_wtime();
}

int main(void) {
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    omp_set_num_threads(2);
    TGPTModel model;
    if (!tgpt_load_model(&model, "assets/model.bin")) {
        return 1;
    }
    tgpt_reset_cache(&model);
    tgpt_forward(&model, 464, 0);
    double start = now();
    for (int i = 0; i < 20; i++) {
        tgpt_forward(&model, 464, i % 32);
    }
    double finish = now();
    printf("forward_ms %.3f\n", (finish - start) * 1000.0 / 20.0);
    start = now();
    for (int i = 0; i < 100; i++) {
        tgpt_matmul_nt_f16(model.h, model.wte, model.logits, 1, model.cfg.n_embd, model.cfg.vocab_size);
    }
    finish = now();
    printf("logits_ms %.3f\n", (finish - start) * 1000.0 / 100.0);
    start = now();
    for (int i = 0; i < 500; i++) {
        tgpt_matmul_f16(model.h, model.layers[0].qkv_w, model.qkv, 1, model.cfg.n_embd, 3 * model.cfg.n_embd);
    }
    finish = now();
    printf("qkv_ms %.3f\n", (finish - start) * 1000.0 / 500.0);
    start = now();
    for (int i = 0; i < 500; i++) {
        tgpt_matmul_f16(model.h, model.layers[0].fc_w, model.ff, 1, model.cfg.n_embd, model.cfg.n_inner);
    }
    finish = now();
    printf("fc_ms %.3f\n", (finish - start) * 1000.0 / 500.0);
    start = now();
    for (int i = 0; i < 500; i++) {
        tgpt_matmul_f16(model.ff, model.layers[0].ff_proj_w, model.h, 1, model.cfg.n_inner, model.cfg.n_embd);
    }
    finish = now();
    printf("proj_ms %.3f\n", (finish - start) * 1000.0 / 500.0);
    tgpt_free_model(&model);
    return 0;
}
