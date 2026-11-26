#include "tinygpt.h"

#include <math.h>
#include <string.h>

void tgpt_attention(TGPTModel *model, int layer, int pos) {
    const int embd = model->cfg.n_embd;
    const int heads = model->cfg.n_head;
    const int dim = model->cfg.head_dim;
    const int ctx = model->cfg.n_ctx;
    TGPTLayer *params = &model->layers[layer];
    tgpt_layernorm(model->h, model->x, params->ln1_w, params->ln1_b, embd);
    tgpt_matmul(model->h, params->qkv_w, model->qkv, 1, embd, 3 * embd);
    for (int i = 0; i < 3 * embd; i++) {
        model->qkv[i] += params->qkv_b[i];
    }
    float *query = model->qkv;
    float *key = model->qkv + embd;
    float *value = model->qkv + 2 * embd;
    tgpt_cache_store(model, layer, pos, key, value);
    float scale = 1.0f / sqrtf((float)dim);
    for (int head = 0; head < heads; head++) {
        const float *qh = query + head * dim;
        for (int t = 0; t <= pos; t++) {
            const float *kh = model->k_cache + (((size_t)layer * ctx + t) * embd) + head * dim;
            float dot = 0.0f;
            for (int d = 0; d < dim; d++) {
                dot += qh[d] * kh[d];
            }
            model->scores[t] = dot * scale;
        }
        tgpt_softmax(model->scores, pos + 1);
        float *out = model->attn + head * dim;
        memset(out, 0, (size_t)dim * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            float weight = model->scores[t];
            const float *vh = model->v_cache + (((size_t)layer * ctx + t) * embd) + head * dim;
            for (int d = 0; d < dim; d++) {
                out[d] += weight * vh[d];
            }
        }
    }
    tgpt_matmul(model->attn, params->attn_proj_w, model->h, 1, embd, embd);
    for (int i = 0; i < embd; i++) {
        model->x[i] += model->h[i] + params->attn_proj_b[i];
    }
}
