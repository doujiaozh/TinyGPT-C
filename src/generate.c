#include "tinygpt.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static uint64_t random_next(TGPTModel *model) {
    uint64_t value = model->rng;
    if (value == 0) {
        value = 0x9e3779b97f4a7c15ULL;
    }
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    model->rng = value;
    return value * 2685821657736338717ULL;
}

static double random_double(TGPTModel *model) {
    uint64_t value = random_next(model);
    return (double)(value >> 11) * (1.0 / 9007199254740992.0);
}

static int argmax_logits(const float *values, int count) {
    int best = 0;
    for (int i = 1; i < count; i++) {
        if (values[i] > values[best]) {
            best = i;
        }
    }
    return best;
}

int tgpt_forward(TGPTModel *model, int token, int pos) {
    if (token < 0 || token >= model->cfg.vocab_size || pos < 0 || pos >= model->cfg.n_ctx) {
        return 0;
    }
    tgpt_embed(model, token, pos);
    for (int layer = 0; layer < model->cfg.n_layer; layer++) {
        tgpt_attention(model, layer, pos);
        tgpt_mlp(model, layer);
    }
    tgpt_layernorm(model->h, model->x, model->lnf_w, model->lnf_b, model->cfg.n_embd);
    tgpt_matmul_nt(model->h, model->wte, model->logits, 1, model->cfg.n_embd, model->cfg.vocab_size);
    return 1;
}

int tgpt_sample(TGPTModel *model, float temperature, int top_k, float top_p) {
    int vocab = model->cfg.vocab_size;
    if (vocab <= 0) {
        return -1;
    }
    if (temperature <= 0.0f) {
        return argmax_logits(model->logits, vocab);
    }
    int limit = top_k;
    if (limit <= 0 || limit > vocab) {
        limit = vocab;
    }
    float *values = (float *)malloc((size_t)vocab * sizeof(float));
    int *indices = (int *)malloc((size_t)vocab * sizeof(int));
    if (!values || !indices) {
        free(values);
        free(indices);
        return argmax_logits(model->logits, vocab);
    }
    float inverse_temperature = 1.0f / temperature;
    for (int i = 0; i < vocab; i++) {
        values[i] = model->logits[i] * inverse_temperature;
        indices[i] = i;
    }
    for (int i = 0; i < limit; i++) {
        int best = i;
        for (int j = i + 1; j < vocab; j++) {
            if (values[j] > values[best]) {
                best = j;
            }
        }
        if (best != i) {
            float temporary = values[i];
            values[i] = values[best];
            values[best] = temporary;
            int temporary_index = indices[i];
            indices[i] = indices[best];
            indices[best] = temporary_index;
        }
    }
    float maximum = values[0];
    double sum = 0.0;
    for (int i = 0; i < limit; i++) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    if (sum <= 0.0) {
        int result = indices[0];
        free(values);
        free(indices);
        return result;
    }
    float threshold_p = top_p;
    if (!(threshold_p > 0.0f) || threshold_p > 1.0f) {
        threshold_p = 1.0f;
    }
    double threshold = (double)threshold_p * sum;
    double cumulative = 0.0;
    int keep = limit;
    for (int i = 0; i < limit; i++) {
        cumulative += values[i];
        if (cumulative >= threshold) {
            keep = i + 1;
            break;
        }
    }
    if (keep < 1) {
        keep = 1;
    }
    double kept_sum = 0.0;
    for (int i = 0; i < keep; i++) {
        kept_sum += values[i];
    }
    double target = random_double(model) * kept_sum;
    double running = 0.0;
    int result = indices[keep - 1];
    for (int i = 0; i < keep; i++) {
        running += values[i];
        if (running >= target) {
            result = indices[i];
            break;
        }
    }
    free(values);
    free(indices);
    return result;
}
