#include "tinygpt.h"

#include <errno.h>
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

static int read_bytes(FILE *file, void *buffer, size_t size) {
    if (size == 0) {
        return 1;
    }
    return fread(buffer, 1, size, file) == size;
}

static int read_u32(FILE *file, uint32_t *value) {
    unsigned char bytes[4];
    if (!read_bytes(file, bytes, 4)) {
        return 0;
    }
    *value = (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) | ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
    return 1;
}

static int read_u64(FILE *file, uint64_t *value) {
    unsigned char bytes[8];
    if (!read_bytes(file, bytes, 8)) {
        return 0;
    }
    *value = (uint64_t)bytes[0] | ((uint64_t)bytes[1] << 8) | ((uint64_t)bytes[2] << 16) | ((uint64_t)bytes[3] << 24) |
             ((uint64_t)bytes[4] << 32) | ((uint64_t)bytes[5] << 40) | ((uint64_t)bytes[6] << 48) | ((uint64_t)bytes[7] << 56);
    return 1;
}

static int is_half_tensor(const char *name) {
    if (strcmp(name, "wte.weight") == 0) {
        return 1;
    }
    if (strstr(name, ".attn.") && strstr(name, ".weight")) {
        return 1;
    }
    if (strstr(name, ".mlp.") && strstr(name, ".weight")) {
        return 1;
    }
    return 0;
}

static uint16_t float_to_half(float value) {
    __m128 input = _mm_set_ss(value);
    __m128i half = _mm_cvtps_ph(input, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return (uint16_t)_mm_extract_epi16(half, 0);
}

static void convert_f32_to_f16(const float *source, uint16_t *destination, uint64_t count) {
    uint64_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 values = _mm256_loadu_ps(source + i);
        __m128i half = _mm256_cvtps_ph(values, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i *)(destination + i), half);
    }
    for (; i < count; i++) {
        destination[i] = float_to_half(source[i]);
    }
}

static void transpose_f32_to_f16(const float *source, uint16_t *destination, int rows, int columns) {
    for (int i0 = 0; i0 < rows; i0 += 32) {
        int i1 = i0 + 32 < rows ? i0 + 32 : rows;
        for (int j0 = 0; j0 < columns; j0 += 32) {
            int j1 = j0 + 32 < columns ? j0 + 32 : columns;
            for (int i = i0; i < i1; i++) {
                const float *row = source + (size_t)i * columns;
                for (int j = j0; j < j1; j++) {
                    destination[(size_t)j * rows + i] = float_to_half(row[j]);
                }
            }
        }
    }
}

static int resolve_tensor(TGPTModel *model, const char *name, void **slot, uint32_t *ndim, uint64_t *d0, uint64_t *d1) {
    const TGPTConfig *cfg = &model->cfg;
    if (strcmp(name, "wte.weight") == 0) {
        *slot = &model->wte;
        *ndim = 2;
        *d0 = (uint64_t)cfg->vocab_size;
        *d1 = (uint64_t)cfg->n_embd;
        return 1;
    }
    if (strcmp(name, "wpe.weight") == 0) {
        *slot = &model->wpe;
        *ndim = 2;
        *d0 = (uint64_t)cfg->n_ctx;
        *d1 = (uint64_t)cfg->n_embd;
        return 1;
    }
    if (strcmp(name, "ln_f.weight") == 0) {
        *slot = &model->lnf_w;
        *ndim = 1;
        *d0 = (uint64_t)cfg->n_embd;
        *d1 = 0;
        return 1;
    }
    if (strcmp(name, "ln_f.bias") == 0) {
        *slot = &model->lnf_b;
        *ndim = 1;
        *d0 = (uint64_t)cfg->n_embd;
        *d1 = 0;
        return 1;
    }
    for (int layer = 0; layer < cfg->n_layer; layer++) {
        TGPTLayer *params = &model->layers[layer];
        char expected[128];
        snprintf(expected, sizeof(expected), "h.%d.ln_1.weight", layer);
        if (strcmp(name, expected) == 0) {
            *slot = &params->ln1_w;
            *ndim = 1;
            *d0 = (uint64_t)cfg->n_embd;
            *d1 = 0;
            return 1;
        }
        snprintf(expected, sizeof(expected), "h.%d.ln_1.bias", layer);
        if (strcmp(name, expected) == 0) {
            *slot = &params->ln1_b;
            *ndim = 1;
            *d0 = (uint64_t)cfg->n_embd;
            *d1 = 0;
            return 1;
        }
        snprintf(expected, sizeof(expected), "h.%d.attn.c_attn.weight", layer);
        if (strcmp(name, expected) == 0) {
            *slot = &params->qkv_w;
            *ndim = 2;
            *d0 = (uint64_t)cfg->n_embd;
            *d1 = (uint64_t)(3 * cfg->n_embd);
            return 1;
        }
        snprintf(expected, sizeof(expected), "h.%d.attn.c_attn.bias", layer);
        if (strcmp(name, expected) == 0) {
            *slot = &params->qkv_b;
            *ndim = 1;
            *d0 = (uint64_t)(3 * cfg->n_embd);
            *d1 = 0;
            return 1;
        }
        snprintf(expected, sizeof(expected), "h.%d.attn.c_proj.weight", layer);
        if (strcmp(name, expected) == 0) {
            *slot = &params->attn_proj_w;
            *ndim = 2;
            *d0 = (uint64_t)cfg->n_embd;
            *d1 = (uint64_t)cfg->n_embd;
            return 1;
        }
        snprintf(expected, sizeof(expected), "h.%d.attn.c_proj.bias", layer);
        if (strcmp(name, expected) == 0) {
            *slot = &params->attn_proj_b;
            *ndim = 1;
            *d0 = (uint64_t)cfg->n_embd;
            *d1 = 0;
            return 1;
        }
        snprintf(expected, sizeof(expected), "h.%d.ln_2.weight", layer);
        if (strcmp(name, expected) == 0) {
            *slot = &params->ln2_w;
            *ndim = 1;
            *d0 = (uint64_t)cfg->n_embd;
            *d1 = 0;
            return 1;
        }
        snprintf(expected, sizeof(expected), "h.%d.ln_2.bias", layer);
        if (strcmp(name, expected) == 0) {
            *slot = &params->ln2_b;
            *ndim = 1;
            *d0 = (uint64_t)cfg->n_embd;
            *d1 = 0;
            return 1;
        }
        snprintf(expected, sizeof(expected), "h.%d.mlp.c_fc.weight", layer);
        if (strcmp(name, expected) == 0) {
            *slot = &params->fc_w;
            *ndim = 2;
            *d0 = (uint64_t)cfg->n_embd;
            *d1 = (uint64_t)cfg->n_inner;
            return 1;
        }
        snprintf(expected, sizeof(expected), "h.%d.mlp.c_fc.bias", layer);
        if (strcmp(name, expected) == 0) {
            *slot = &params->fc_b;
            *ndim = 1;
            *d0 = (uint64_t)cfg->n_inner;
            *d1 = 0;
            return 1;
        }
        snprintf(expected, sizeof(expected), "h.%d.mlp.c_proj.weight", layer);
        if (strcmp(name, expected) == 0) {
            *slot = &params->ff_proj_w;
            *ndim = 2;
            *d0 = (uint64_t)cfg->n_inner;
            *d1 = (uint64_t)cfg->n_embd;
            return 1;
        }
        snprintf(expected, sizeof(expected), "h.%d.mlp.c_proj.bias", layer);
        if (strcmp(name, expected) == 0) {
            *slot = &params->ff_proj_b;
            *ndim = 1;
            *d0 = (uint64_t)cfg->n_embd;
            *d1 = 0;
            return 1;
        }
    }
    return 0;
}

static int all_weights_present(const TGPTModel *model) {
    if (!model->wte || !model->wpe || !model->lnf_w || !model->lnf_b) {
        return 0;
    }
    for (int layer = 0; layer < model->cfg.n_layer; layer++) {
        const TGPTLayer *params = &model->layers[layer];
        if (!params->ln1_w || !params->ln1_b || !params->qkv_w || !params->qkv_b || !params->attn_proj_w ||
            !params->attn_proj_b || !params->ln2_w || !params->ln2_b || !params->fc_w || !params->fc_b ||
            !params->ff_proj_w || !params->ff_proj_b) {
            return 0;
        }
    }
    return 1;
}

void tgpt_free_model(TGPTModel *model) {
    if (!model) {
        return;
    }
    free(model->wte);
    free(model->wpe);
    free(model->lnf_w);
    free(model->lnf_b);
    for (int layer = 0; layer < TGPT_MAX_LAYERS; layer++) {
        TGPTLayer *params = &model->layers[layer];
        free(params->ln1_w);
        free(params->ln1_b);
        free(params->qkv_w);
        free(params->qkv_b);
        free(params->attn_proj_w);
        free(params->attn_proj_b);
        free(params->ln2_w);
        free(params->ln2_b);
        free(params->fc_w);
        free(params->fc_b);
        free(params->ff_proj_w);
        free(params->ff_proj_b);
    }
    free(model->x);
    free(model->h);
    free(model->qkv);
    free(model->attn);
    free(model->ff);
    free(model->logits);
    free(model->scores);
    free(model->k_cache);
    free(model->v_cache);
    memset(model, 0, sizeof(*model));
}

int tgpt_load_model(TGPTModel *model, const char *path) {
    memset(model, 0, sizeof(*model));
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "error: cannot open model %s: %s\n", path, strerror(errno));
        return 0;
    }
    char magic[8];
    uint32_t version = 0;
    uint32_t layers = 0;
    uint32_t heads = 0;
    uint32_t embd = 0;
    uint32_t ctx = 0;
    uint32_t vocab = 0;
    uint32_t inner = 0;
    uint32_t tensor_count = 0;
    if (!read_bytes(file, magic, 8) || memcmp(magic, "TGPTBIN1", 8) != 0 || !read_u32(file, &version) ||
        version != 1 || !read_u32(file, &layers) || !read_u32(file, &heads) || !read_u32(file, &embd) ||
        !read_u32(file, &ctx) || !read_u32(file, &vocab) || !read_u32(file, &inner) ||
        !read_u32(file, &tensor_count)) {
        fprintf(stderr, "error: invalid model header\n");
        fclose(file);
        return 0;
    }
    if (layers < 1 || layers > TGPT_MAX_LAYERS || heads < 1 || heads > TGPT_MAX_HEADS || embd < 1 ||
        embd > TGPT_MAX_DIM || ctx < 1 || ctx > TGPT_MAX_CTX || vocab < 1 || vocab > 200000 || inner < 1 ||
        inner > TGPT_MAX_INNER || embd % heads != 0 || tensor_count > 4096) {
        fprintf(stderr, "error: unsupported model configuration\n");
        fclose(file);
        return 0;
    }
    model->cfg.n_layer = (int)layers;
    model->cfg.n_head = (int)heads;
    model->cfg.n_embd = (int)embd;
    model->cfg.n_ctx = (int)ctx;
    model->cfg.vocab_size = (int)vocab;
    model->cfg.n_inner = (int)inner;
    model->cfg.head_dim = (int)embd / (int)heads;
    for (uint32_t index = 0; index < tensor_count; index++) {
        uint32_t name_length = 0;
        if (!read_u32(file, &name_length) || name_length == 0 || name_length > 255) {
            fprintf(stderr, "error: invalid tensor name\n");
            fclose(file);
            tgpt_free_model(model);
            return 0;
        }
        char name[256];
        if (!read_bytes(file, name, name_length)) {
            fprintf(stderr, "error: truncated tensor name\n");
            fclose(file);
            tgpt_free_model(model);
            return 0;
        }
        name[name_length] = 0;
        uint32_t ndim = 0;
        if (!read_u32(file, &ndim) || ndim < 1 || ndim > 2) {
            fprintf(stderr, "error: invalid tensor rank\n");
            fclose(file);
            tgpt_free_model(model);
            return 0;
        }
        uint64_t dims[2] = {0, 0};
        for (uint32_t d = 0; d < ndim; d++) {
            if (!read_u64(file, &dims[d])) {
                fprintf(stderr, "error: truncated tensor shape\n");
                fclose(file);
                tgpt_free_model(model);
                return 0;
            }
        }
        uint32_t dtype = 0;
        uint64_t nbytes = 0;
        if (!read_u32(file, &dtype) || !read_u64(file, &nbytes)) {
            fprintf(stderr, "error: truncated tensor metadata\n");
            fclose(file);
            tgpt_free_model(model);
            return 0;
        }
        void *slot = NULL;
        uint32_t expected_ndim = 0;
        uint64_t expected0 = 0;
        uint64_t expected1 = 0;
        int known = resolve_tensor(model, name, &slot, &expected_ndim, &expected0, &expected1);
        if (!known) {
            if (fseek(file, (long)nbytes, SEEK_CUR) != 0) {
                fprintf(stderr, "error: cannot skip tensor %s\n", name);
                fclose(file);
                tgpt_free_model(model);
                return 0;
            }
            continue;
        }
        if (dtype != 0 || ndim != expected_ndim || dims[0] != expected0 || (ndim == 2 && dims[1] != expected1)) {
            fprintf(stderr, "error: shape or dtype mismatch for %s\n", name);
            fclose(file);
            tgpt_free_model(model);
            return 0;
        }
        if (!slot || *(void **)slot) {
            fprintf(stderr, "error: unresolved or duplicate tensor slot for %s\n", name);
            fclose(file);
            tgpt_free_model(model);
            return 0;
        }
        uint64_t count = dims[0];
        if (ndim == 2) {
            count *= dims[1];
        }
        if (count == 0 || count > (uint64_t)1 << 31 || nbytes != count * sizeof(float)) {
            fprintf(stderr, "error: invalid tensor size for %s\n", name);
            fclose(file);
            tgpt_free_model(model);
            return 0;
        }
        int half = is_half_tensor(name);
        if (half) {
            float *source = (float *)malloc((size_t)nbytes);
            if (!source) {
                fprintf(stderr, "error: out of memory for %s\n", name);
                fclose(file);
                tgpt_free_model(model);
                return 0;
            }
            if (!read_bytes(file, source, (size_t)nbytes)) {
                fprintf(stderr, "error: truncated tensor data for %s\n", name);
                free(source);
                fclose(file);
                tgpt_free_model(model);
                return 0;
            }
            uint16_t *destination = (uint16_t *)malloc((size_t)count * sizeof(uint16_t));
            if (!destination) {
                fprintf(stderr, "error: out of memory for %s\n", name);
                free(source);
                fclose(file);
                tgpt_free_model(model);
                return 0;
            }
            if (strcmp(name, "wte.weight") == 0) {
                convert_f32_to_f16(source, destination, count);
            } else {
                transpose_f32_to_f16(source, destination, (int)dims[0], (int)dims[1]);
            }
            free(source);
            *(uint16_t **)slot = destination;
        } else {
            float *data = (float *)malloc((size_t)nbytes);
            if (!data) {
                fprintf(stderr, "error: out of memory for %s\n", name);
                fclose(file);
                tgpt_free_model(model);
                return 0;
            }
            if (!read_bytes(file, data, (size_t)nbytes)) {
                fprintf(stderr, "error: truncated tensor data for %s\n", name);
                free(data);
                fclose(file);
                tgpt_free_model(model);
                return 0;
            }
            *(float **)slot = data;
        }
    }
    fclose(file);
    if (!all_weights_present(model)) {
        fprintf(stderr, "error: model file is missing required tensors\n");
        tgpt_free_model(model);
        return 0;
    }
    size_t embd_count = (size_t)embd;
    size_t inner_count = (size_t)inner;
    size_t vocab_count = (size_t)vocab;
    model->x = (float *)calloc(embd_count, sizeof(float));
    model->h = (float *)calloc(embd_count, sizeof(float));
    model->qkv = (float *)calloc(embd_count * 3, sizeof(float));
    model->attn = (float *)calloc(embd_count, sizeof(float));
    model->ff = (float *)calloc(inner_count, sizeof(float));
    model->logits = (float *)calloc(vocab_count, sizeof(float));
    model->scores = (float *)calloc((size_t)ctx * (size_t)heads, sizeof(float));
    size_t cache_count = (size_t)layers * (size_t)ctx * embd_count;
    model->k_cache = (float *)calloc(cache_count, sizeof(float));
    model->v_cache = (float *)calloc(cache_count, sizeof(float));
    if (!model->x || !model->h || !model->qkv || !model->attn || !model->ff || !model->logits || !model->scores ||
        !model->k_cache || !model->v_cache) {
        fprintf(stderr, "error: out of memory for activations\n");
        tgpt_free_model(model);
        return 0;
    }
    model->cache_len = 0;
    model->rng = 0x9e3779b97f4a7c15ULL;
    printf("model loaded: layers=%d heads=%d embd=%d ctx=%d vocab=%d inner=%d\n", model->cfg.n_layer,
           model->cfg.n_head, model->cfg.n_embd, model->cfg.n_ctx, model->cfg.vocab_size, model->cfg.n_inner);
    return 1;
}
