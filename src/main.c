#include "tinygpt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void print_usage(const char *program) {
    printf("usage: %s [options]\n", program);
    printf("  --model PATH      model binary, default assets/model.bin\n");
    printf("  --vocab PATH      vocab json, default assets/vocab.json\n");
    printf("  --merges PATH     merges text, default assets/merges.txt\n");
    printf("  --prompt TEXT     input prompt\n");
    printf("  --tokens N        generated tokens, default 48\n");
    printf("  --temp F          sampling temperature, default 0.8\n");
    printf("  --top-k N         top-k sampling, default 40\n");
    printf("  --top-p F         nucleus sampling, default 0.95\n");
    printf("  --seed N          random seed, default 1234\n");
    printf("  --greedy          deterministic argmax decoding\n");
    printf("  --trace           print prompt and generated token ids\n");
}

int main(int argc, char **argv) {
    const char *model_path = "assets/model.bin";
    const char *vocab_path = "assets/vocab.json";
    const char *merges_path = "assets/merges.txt";
    const char *prompt = "The future of artificial intelligence is";
    int generate_tokens = 48;
    float temperature = 0.8f;
    int top_k = 40;
    float top_p = 0.95f;
    unsigned long long seed = 1234;
    int trace = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) {
            vocab_path = argv[++i];
        } else if (strcmp(argv[i], "--merges") == 0 && i + 1 < argc) {
            merges_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            generate_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            top_p = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--greedy") == 0) {
            temperature = 0.0f;
        } else if (strcmp(argv[i], "--trace") == 0) {
            trace = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "error: unknown or incomplete option %s\n", argv[i]);
            print_usage(argv[0]);
            return 2;
        }
    }
    if (generate_tokens < 1) {
        generate_tokens = 1;
    }
    printf("tinygpt-c: pure C GPT-2 inference\n");
    TGPTTokenizer tokenizer;
    if (!tgpt_tokenizer_load(&tokenizer, vocab_path, merges_path)) {
        fprintf(stderr, "error: cannot load tokenizer\n");
        return 1;
    }
    printf("tokenizer loaded: vocab=%d merges=%d\n", tokenizer.vocab_size, tokenizer.n_merges);
    TGPTModel model;
    if (!tgpt_load_model(&model, model_path)) {
        tgpt_tokenizer_free(&tokenizer);
        return 1;
    }
    model.rng = (uint64_t)seed;
    int prompt_capacity = (int)strlen(prompt) + 8;
    int *ids = (int *)malloc((size_t)(prompt_capacity + generate_tokens + 1) * sizeof(int));
    if (!ids) {
        fprintf(stderr, "error: out of memory\n");
        tgpt_free_model(&model);
        tgpt_tokenizer_free(&tokenizer);
        return 1;
    }
    int prompt_tokens = tgpt_encode(&tokenizer, prompt, ids, prompt_capacity);
    if (prompt_tokens < 0) {
        fprintf(stderr, "error: prompt tokenization failed\n");
        free(ids);
        tgpt_free_model(&model);
        tgpt_tokenizer_free(&tokenizer);
        return 1;
    }
    if (prompt_tokens == 0) {
        ids[0] = TGPT_EOT;
        prompt_tokens = 1;
    }
    if (prompt_tokens >= model.cfg.n_ctx) {
        fprintf(stderr, "error: prompt exceeds context window\n");
        free(ids);
        tgpt_free_model(&model);
        tgpt_tokenizer_free(&tokenizer);
        return 1;
    }
    int room = model.cfg.n_ctx - prompt_tokens;
    if (generate_tokens > room) {
        generate_tokens = room;
    }
    if (generate_tokens < 1) {
        fprintf(stderr, "error: no room for generation\n");
        free(ids);
        tgpt_free_model(&model);
        tgpt_tokenizer_free(&tokenizer);
        return 1;
    }
    int *extended = (int *)realloc(ids, (size_t)(prompt_tokens + generate_tokens + 1) * sizeof(int));
    if (!extended) {
        fprintf(stderr, "error: out of memory\n");
        free(ids);
        tgpt_free_model(&model);
        tgpt_tokenizer_free(&tokenizer);
        return 1;
    }
    ids = extended;
    printf("prompt: %s\n", prompt);
    printf("prompt tokens: %d\n", prompt_tokens);
    tgpt_reset_cache(&model);
    clock_t start = clock();
    for (int i = 0; i < prompt_tokens; i++) {
        if (!tgpt_forward(&model, ids[i], i)) {
            fprintf(stderr, "error: forward pass failed\n");
            free(ids);
            tgpt_free_model(&model);
            tgpt_tokenizer_free(&tokenizer);
            return 1;
        }
    }
    int produced = 0;
    for (int step = 0; step < generate_tokens; step++) {
        int next = tgpt_sample(&model, temperature, top_k, top_p);
        if (next < 0 || next >= model.cfg.vocab_size) {
            fprintf(stderr, "error: sampling failed\n");
            break;
        }
        if (next == TGPT_EOT) {
            break;
        }
        ids[prompt_tokens + produced] = next;
        produced++;
        if (step + 1 >= generate_tokens) {
            break;
        }
        if (!tgpt_forward(&model, next, prompt_tokens + produced - 1)) {
            fprintf(stderr, "error: forward pass failed\n");
            break;
        }
    }
    clock_t finish = clock();
    double elapsed = (double)(finish - start) / (double)CLOCKS_PER_SEC;
    int total = prompt_tokens + produced;
    size_t output_capacity = (size_t)total * (size_t)(tokenizer.max_token_len > 0 ? tokenizer.max_token_len : 8) + 1;
    char *output = (char *)malloc(output_capacity);
    if (!output) {
        fprintf(stderr, "error: out of memory\n");
        free(ids);
        tgpt_free_model(&model);
        tgpt_tokenizer_free(&tokenizer);
        return 1;
    }
    int output_length = tgpt_decode(&tokenizer, ids, total, output, (int)output_capacity);
    if (output_length < 0) {
        fprintf(stderr, "error: decode failed\n");
        free(output);
        free(ids);
        tgpt_free_model(&model);
        tgpt_tokenizer_free(&tokenizer);
        return 1;
    }
    printf("output:\n");
    fwrite(output, 1, (size_t)output_length, stdout);
    printf("\n");
    if (trace) {
        printf("token ids:");
        for (int i = 0; i < total; i++) {
            printf(" %d", ids[i]);
        }
        printf("\n");
    }
    printf("generated tokens: %d\n", produced);
    printf("context used: %d\n", total);
    printf("elapsed seconds: %.3f\n", elapsed);
    if (elapsed > 0.0 && produced > 0) {
        printf("tokens per second: %.2f\n", (double)produced / elapsed);
    }
    free(output);
    free(ids);
    tgpt_free_model(&model);
    tgpt_tokenizer_free(&tokenizer);
    return 0;
}
