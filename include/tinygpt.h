#ifndef TINYGPT_H
#define TINYGPT_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define TGPT_MAX_LAYERS 48
#define TGPT_MAX_HEADS 32
#define TGPT_MAX_DIM 2048
#define TGPT_MAX_INNER 8192
#define TGPT_MAX_CTX 4096
#define TGPT_EOT 50256

typedef struct {
    int n_layer;
    int n_head;
    int n_embd;
    int n_ctx;
    int vocab_size;
    int n_inner;
    int head_dim;
} TGPTConfig;

typedef struct {
    float *ln1_w;
    float *ln1_b;
    float *qkv_w;
    float *qkv_b;
    float *attn_proj_w;
    float *attn_proj_b;
    float *ln2_w;
    float *ln2_b;
    float *fc_w;
    float *fc_b;
    float *ff_proj_w;
    float *ff_proj_b;
} TGPTLayer;

typedef struct {
    TGPTConfig cfg;
    float *wte;
    float *wpe;
    float *lnf_w;
    float *lnf_b;
    TGPTLayer layers[TGPT_MAX_LAYERS];
    float *x;
    float *h;
    float *qkv;
    float *attn;
    float *ff;
    float *logits;
    float *scores;
    float *k_cache;
    float *v_cache;
    int cache_len;
    uint64_t rng;
} TGPTModel;

typedef struct {
    int vocab_size;
    int n_merges;
    int max_token_len;
    char **token_bytes;
    int *token_len;
    int byte_to_id[256];
    uint64_t *vocab_keys;
    int *vocab_ids;
    int vocab_cap;
    uint64_t *pair_keys;
    int *pair_rank;
    int *pair_result;
    int pair_cap;
} TGPTTokenizer;

int tgpt_load_model(TGPTModel *model, const char *path);
void tgpt_free_model(TGPTModel *model);
int tgpt_forward(TGPTModel *model, int token, int pos);
int tgpt_sample(TGPTModel *model, float temperature, int top_k, float top_p);
void tgpt_reset_cache(TGPTModel *model);

void tgpt_matmul(const float *a, const float *b, float *c, int m, int k, int n);
void tgpt_matmul_nt(const float *a, const float *b, float *c, int m, int k, int n);
void tgpt_softmax(float *x, int n);
void tgpt_layernorm(float *out, const float *x, const float *w, const float *b, int n);
void tgpt_gelu(float *x, int n);

void tgpt_embed(TGPTModel *model, int token, int pos);
void tgpt_attention(TGPTModel *model, int layer, int pos);
void tgpt_mlp(TGPTModel *model, int layer);
void tgpt_cache_store(TGPTModel *model, int layer, int pos, const float *k, const float *v);

int tgpt_tokenizer_load(TGPTTokenizer *tokenizer, const char *vocab_path, const char *merges_path);
void tgpt_tokenizer_free(TGPTTokenizer *tokenizer);
int tgpt_encode(const TGPTTokenizer *tokenizer, const char *text, int *ids, int max_ids);
int tgpt_decode(const TGPTTokenizer *tokenizer, const int *ids, int n, char *out, int max_out);

#endif
