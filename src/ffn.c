#include "tinygpt.h"

void tgpt_mlp(TGPTModel *model, int layer) {
    const int embd = model->cfg.n_embd;
    const int inner = model->cfg.n_inner;
    TGPTLayer *params = &model->layers[layer];
    tgpt_layernorm(model->h, model->x, params->ln2_w, params->ln2_b, embd);
    tgpt_matmul(model->h, params->fc_w, model->ff, 1, embd, inner);
    for (int i = 0; i < inner; i++) {
        model->ff[i] += params->fc_b[i];
    }
    tgpt_gelu(model->ff, inner);
    tgpt_matmul(model->ff, params->ff_proj_w, model->h, 1, inner, embd);
    for (int i = 0; i < embd; i++) {
        model->x[i] += model->h[i] + params->ff_proj_b[i];
    }
}
