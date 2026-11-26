#include "tinygpt.h"

#include <string.h>

void tgpt_embed(TGPTModel *model, int token, int pos) {
    int embd = model->cfg.n_embd;
    const float *token_row = model->wte + (size_t)token * embd;
    const float *pos_row = model->wpe + (size_t)pos * embd;
    for (int i = 0; i < embd; i++) {
        model->x[i] = token_row[i] + pos_row[i];
    }
}
