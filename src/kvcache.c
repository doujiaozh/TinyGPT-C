#include "tinygpt.h"

#include <string.h>

void tgpt_reset_cache(TGPTModel *model) {
    model->cache_len = 0;
}

void tgpt_cache_store(TGPTModel *model, int layer, int pos, const float *k, const float *v) {
    int embd = model->cfg.n_embd;
    size_t offset = ((size_t)layer * model->cfg.n_ctx + pos) * embd;
    memcpy(model->k_cache + offset, k, (size_t)embd * sizeof(float));
    memcpy(model->v_cache + offset, v, (size_t)embd * sizeof(float));
    if (pos + 1 > model->cache_len) {
        model->cache_len = pos + 1;
    }
}
