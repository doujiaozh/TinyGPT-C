#include "tinygpt.h"

#include <immintrin.h>

void tgpt_embed(TGPTModel *model, int token, int pos) {
    int embd = model->cfg.n_embd;
    const uint16_t *token_row = model->wte + (size_t)token * embd;
    const float *pos_row = model->wpe + (size_t)pos * embd;
    int i = 0;
    for (; i + 8 <= embd; i += 8) {
        __m128i half = _mm_loadu_si128((const __m128i *)(token_row + i));
        __m256 token_values = _mm256_cvtph_ps(half);
        __m256 position_values = _mm256_loadu_ps(pos_row + i);
        _mm256_storeu_ps(model->x + i, _mm256_add_ps(token_values, position_values));
    }
    for (; i < embd; i++) {
        model->x[i] = tgpt_half_to_float(token_row[i]) + pos_row[i];
    }
}
