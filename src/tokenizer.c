#include "tinygpt.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#define TGPT_VOCAB_LIMIT 200000
#define TGPT_PAIR_CAP 262144

static uint64_t hash_bytes(const unsigned char *data, int length) {
    uint64_t hash = 1469598103934665603ULL;
    for (int i = 0; i < length; i++) {
        hash ^= (uint64_t)data[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static int next_power_of_two(int value) {
    int result = 1;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

static int unicode_to_byte(uint32_t codepoint) {
    int extra = 0;
    for (int byte = 0; byte < 256; byte++) {
        if ((byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || (byte >= 174 && byte <= 255)) {
            if ((uint32_t)byte == codepoint) {
                return byte;
            }
        } else {
            if ((uint32_t)(256 + extra) == codepoint) {
                return byte;
            }
            extra++;
        }
    }
    return -1;
}

static int utf8_decode(const unsigned char *text, int length, int *position, uint32_t *codepoint) {
    int index = *position;
    if (index >= length) {
        return 0;
    }
    unsigned char first = text[index];
    if (first < 0x80) {
        *codepoint = first;
        *position = index + 1;
        return 1;
    }
    int extra = 0;
    uint32_t value = 0;
    if ((first & 0xE0) == 0xC0) {
        extra = 1;
        value = first & 0x1F;
    } else if ((first & 0xF0) == 0xE0) {
        extra = 2;
        value = first & 0x0F;
    } else if ((first & 0xF8) == 0xF0) {
        extra = 3;
        value = first & 0x07;
    } else {
        return 0;
    }
    if (index + extra >= length) {
        return 0;
    }
    for (int i = 1; i <= extra; i++) {
        unsigned char next = text[index + i];
        if ((next & 0xC0) != 0x80) {
            return 0;
        }
        value = (value << 6) | (next & 0x3F);
    }
    *codepoint = value;
    *position = index + extra + 1;
    return 1;
}

static int utf8_encode(uint32_t codepoint, char *output) {
    if (codepoint < 0x80) {
        output[0] = (char)codepoint;
        return 1;
    }
    if (codepoint < 0x800) {
        output[0] = (char)(0xC0 | (codepoint >> 6));
        output[1] = (char)(0x80 | (codepoint & 0x3F));
        return 2;
    }
    if (codepoint < 0x10000) {
        output[0] = (char)(0xE0 | (codepoint >> 12));
        output[1] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
        output[2] = (char)(0x80 | (codepoint & 0x3F));
        return 3;
    }
    output[0] = (char)(0xF0 | (codepoint >> 18));
    output[1] = (char)(0x80 | ((codepoint >> 12) & 0x3F));
    output[2] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
    output[3] = (char)(0x80 | (codepoint & 0x3F));
    return 4;
}

static int token_to_bytes(const char *text, unsigned char **output, int *output_length) {
    int length = (int)strlen(text);
    unsigned char *buffer = (unsigned char *)malloc((size_t)(length > 0 ? length : 1));
    if (!buffer) {
        return 0;
    }
    int position = 0;
    int written = 0;
    while (position < length) {
        uint32_t codepoint = 0;
        if (!utf8_decode((const unsigned char *)text, length, &position, &codepoint)) {
            free(buffer);
            return 0;
        }
        int byte = unicode_to_byte(codepoint);
        if (byte < 0) {
            free(buffer);
            return 0;
        }
        buffer[written++] = (unsigned char)byte;
    }
    *output = buffer;
    *output_length = written;
    return 1;
}

static int read_hex4(FILE *file, uint32_t *value) {
    uint32_t result = 0;
    for (int i = 0; i < 4; i++) {
        int c = fgetc(file);
        int digit = -1;
        if (c >= '0' && c <= '9') {
            digit = c - '0';
        } else if (c >= 'a' && c <= 'f') {
            digit = c - 'a' + 10;
        } else if (c >= 'A' && c <= 'F') {
            digit = c - 'A' + 10;
        }
        if (digit < 0) {
            return 0;
        }
        result = (result << 4) | (uint32_t)digit;
    }
    *value = result;
    return 1;
}

static int read_json_string(FILE *file, char *output, int capacity) {
    int c = fgetc(file);
    if (c != '"') {
        return 0;
    }
    int written = 0;
    while ((c = fgetc(file)) != EOF) {
        if (c == '"') {
            if (written >= capacity) {
                return 0;
            }
            output[written] = 0;
            return 1;
        }
        if (c == '\\') {
            int escaped = fgetc(file);
            unsigned char simple = 0;
            int is_simple = 1;
            switch (escaped) {
                case '"':
                    simple = '"';
                    break;
                case '\\':
                    simple = '\\';
                    break;
                case '/':
                    simple = '/';
                    break;
                case 'b':
                    simple = 8;
                    break;
                case 'f':
                    simple = 12;
                    break;
                case 'n':
                    simple = 10;
                    break;
                case 'r':
                    simple = 13;
                    break;
                case 't':
                    simple = 9;
                    break;
                default:
                    is_simple = 0;
                    break;
            }
            if (is_simple) {
                if (written + 1 >= capacity) {
                    return 0;
                }
                output[written++] = (char)simple;
                continue;
            }
            if (escaped != 'u') {
                return 0;
            }
            uint32_t codepoint = 0;
            if (!read_hex4(file, &codepoint)) {
                return 0;
            }
            if (codepoint >= 0xD800 && codepoint <= 0xDBFF) {
                int slash = fgetc(file);
                int letter = fgetc(file);
                if (slash != '\\' || letter != 'u') {
                    return 0;
                }
                uint32_t low = 0;
                if (!read_hex4(file, &low) || low < 0xDC00 || low > 0xDFFF) {
                    return 0;
                }
                codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00);
            }
            if (written + 4 >= capacity) {
                return 0;
            }
            written += utf8_encode(codepoint, output + written);
            continue;
        }
        if (written + 1 >= capacity) {
            return 0;
        }
        output[written++] = (char)c;
    }
    return 0;
}

static int skip_space(FILE *file) {
    int c = 0;
    do {
        c = fgetc(file);
    } while (c == ' ' || c == '\t' || c == '\n' || c == '\r');
    return c;
}

static int parse_integer(FILE *file, int *value) {
    int c = fgetc(file);
    while (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        c = fgetc(file);
    }
    if (c < '0' || c > '9') {
        return 0;
    }
    long result = 0;
    while (c >= '0' && c <= '9') {
        result = result * 10 + (c - '0');
        if (result > INT_MAX) {
            return 0;
        }
        c = fgetc(file);
    }
    if (c != EOF) {
        ungetc(c, file);
    }
    *value = (int)result;
    return 1;
}

static void vocab_insert(TGPTTokenizer *tokenizer, int id) {
    uint64_t hash = hash_bytes((const unsigned char *)tokenizer->token_bytes[id], tokenizer->token_len[id]);
    int mask = tokenizer->vocab_cap - 1;
    int slot = (int)(hash & (uint64_t)mask);
    while (tokenizer->vocab_ids[slot] >= 0) {
        int other = tokenizer->vocab_ids[slot];
        if (tokenizer->vocab_keys[slot] == hash && tokenizer->token_len[other] == tokenizer->token_len[id] &&
            memcmp(tokenizer->token_bytes[other], tokenizer->token_bytes[id], (size_t)tokenizer->token_len[id]) == 0) {
            return;
        }
        slot = (slot + 1) & mask;
    }
    tokenizer->vocab_keys[slot] = hash;
    tokenizer->vocab_ids[slot] = id;
}

static int vocab_find(const TGPTTokenizer *tokenizer, const unsigned char *bytes, int length) {
    uint64_t hash = hash_bytes(bytes, length);
    int mask = tokenizer->vocab_cap - 1;
    int slot = (int)(hash & (uint64_t)mask);
    while (tokenizer->vocab_ids[slot] >= 0) {
        int id = tokenizer->vocab_ids[slot];
        if (tokenizer->vocab_keys[slot] == hash && tokenizer->token_len[id] == length &&
            memcmp(tokenizer->token_bytes[id], bytes, (size_t)length) == 0) {
            return id;
        }
        slot = (slot + 1) & mask;
    }
    return -1;
}

static int pair_find(const TGPTTokenizer *tokenizer, uint64_t key, int *rank, int *result) {
    int mask = tokenizer->pair_cap - 1;
    int slot = (int)((key * 11400714819323198485ULL) & (uint64_t)mask);
    while (tokenizer->pair_keys[slot] != UINT64_MAX) {
        if (tokenizer->pair_keys[slot] == key) {
            *rank = tokenizer->pair_rank[slot];
            *result = tokenizer->pair_result[slot];
            return 1;
        }
        slot = (slot + 1) & mask;
    }
    return 0;
}

static int pair_insert(TGPTTokenizer *tokenizer, uint64_t key, int rank, int result) {
    int mask = tokenizer->pair_cap - 1;
    int slot = (int)((key * 11400714819323198485ULL) & (uint64_t)mask);
    while (tokenizer->pair_keys[slot] != UINT64_MAX) {
        if (tokenizer->pair_keys[slot] == key) {
            if (rank < tokenizer->pair_rank[slot]) {
                tokenizer->pair_rank[slot] = rank;
                tokenizer->pair_result[slot] = result;
            }
            return 1;
        }
        slot = (slot + 1) & mask;
    }
    tokenizer->pair_keys[slot] = key;
    tokenizer->pair_rank[slot] = rank;
    tokenizer->pair_result[slot] = result;
    return 1;
}

static int load_vocab(TGPTTokenizer *tokenizer, const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        return 0;
    }
    tokenizer->token_bytes = (char **)calloc(TGPT_VOCAB_LIMIT, sizeof(char *));
    tokenizer->token_len = (int *)calloc(TGPT_VOCAB_LIMIT, sizeof(int));
    if (!tokenizer->token_bytes || !tokenizer->token_len) {
        fclose(file);
        return 0;
    }
    int c = skip_space(file);
    if (c != '{') {
        fclose(file);
        return 0;
    }
    int maximum = -1;
    while (1) {
        c = skip_space(file);
        if (c == '}') {
            break;
        }
        if (c != '"') {
            fclose(file);
            return 0;
        }
        ungetc(c, file);
        char key[1024];
        if (!read_json_string(file, key, (int)sizeof(key))) {
            fclose(file);
            return 0;
        }
        c = skip_space(file);
        if (c != ':') {
            fclose(file);
            return 0;
        }
        int id = 0;
        if (!parse_integer(file, &id) || id < 0 || id >= TGPT_VOCAB_LIMIT) {
            fclose(file);
            return 0;
        }
        unsigned char *bytes = NULL;
        int length = 0;
        if (!token_to_bytes(key, &bytes, &length)) {
            fclose(file);
            return 0;
        }
        if (tokenizer->token_bytes[id]) {
            free(bytes);
        } else {
            if (length > tokenizer->max_token_len) {
                tokenizer->max_token_len = length;
            }
            tokenizer->token_bytes[id] = (char *)bytes;
            tokenizer->token_len[id] = length;
        }
        if (id > maximum) {
            maximum = id;
        }
        c = skip_space(file);
        if (c == ',') {
            continue;
        }
        if (c == '}') {
            break;
        }
        fclose(file);
        return 0;
    }
    fclose(file);
    if (maximum < 0) {
        return 0;
    }
    tokenizer->vocab_size = maximum + 1;
    for (int i = 0; i < 256; i++) {
        tokenizer->byte_to_id[i] = -1;
    }
    for (int id = 0; id < tokenizer->vocab_size; id++) {
        if (!tokenizer->token_bytes[id] || tokenizer->token_len[id] != 1) {
            continue;
        }
        int byte = (unsigned char)tokenizer->token_bytes[id][0];
        if (tokenizer->byte_to_id[byte] < 0) {
            tokenizer->byte_to_id[byte] = id;
        }
    }
    for (int i = 0; i < 256; i++) {
        if (tokenizer->byte_to_id[i] < 0) {
            return 0;
        }
    }
    tokenizer->vocab_cap = next_power_of_two(tokenizer->vocab_size * 2);
    tokenizer->vocab_keys = (uint64_t *)calloc((size_t)tokenizer->vocab_cap, sizeof(uint64_t));
    tokenizer->vocab_ids = (int *)malloc((size_t)tokenizer->vocab_cap * sizeof(int));
    if (!tokenizer->vocab_keys || !tokenizer->vocab_ids) {
        return 0;
    }
    for (int i = 0; i < tokenizer->vocab_cap; i++) {
        tokenizer->vocab_ids[i] = -1;
    }
    for (int id = 0; id < tokenizer->vocab_size; id++) {
        if (tokenizer->token_bytes[id]) {
            vocab_insert(tokenizer, id);
        }
    }
    return 1;
}

static int load_merges(TGPTTokenizer *tokenizer, const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        return 0;
    }
    tokenizer->pair_cap = TGPT_PAIR_CAP;
    tokenizer->pair_keys = (uint64_t *)malloc((size_t)tokenizer->pair_cap * sizeof(uint64_t));
    tokenizer->pair_rank = (int *)calloc((size_t)tokenizer->pair_cap, sizeof(int));
    tokenizer->pair_result = (int *)calloc((size_t)tokenizer->pair_cap, sizeof(int));
    if (!tokenizer->pair_keys || !tokenizer->pair_rank || !tokenizer->pair_result) {
        fclose(file);
        return 0;
    }
    for (int i = 0; i < tokenizer->pair_cap; i++) {
        tokenizer->pair_keys[i] = UINT64_MAX;
    }
    char line[1024];
    int rank = 0;
    while (fgets(line, (int)sizeof(line), file)) {
        size_t length = strlen(line);
        while (length > 0 && (line[length - 1] == '\n' || line[length - 1] == '\r')) {
            line[--length] = 0;
        }
        if (length == 0 || line[0] == '#') {
            continue;
        }
        char *separator = strchr(line, ' ');
        if (!separator) {
            fclose(file);
            return 0;
        }
        *separator = 0;
        const char *left_text = line;
        const char *right_text = separator + 1;
        unsigned char *left_bytes = NULL;
        unsigned char *right_bytes = NULL;
        int left_length = 0;
        int right_length = 0;
        if (!token_to_bytes(left_text, &left_bytes, &left_length) ||
            !token_to_bytes(right_text, &right_bytes, &right_length)) {
            free(left_bytes);
            free(right_bytes);
            fclose(file);
            return 0;
        }
        int left_id = vocab_find(tokenizer, left_bytes, left_length);
        int right_id = vocab_find(tokenizer, right_bytes, right_length);
        if (left_id < 0 || right_id < 0) {
            free(left_bytes);
            free(right_bytes);
            fclose(file);
            return 0;
        }
        unsigned char combined[1024];
        if (left_length + right_length > (int)sizeof(combined)) {
            free(left_bytes);
            free(right_bytes);
            fclose(file);
            return 0;
        }
        memcpy(combined, left_bytes, (size_t)left_length);
        memcpy(combined + left_length, right_bytes, (size_t)right_length);
        int result_id = vocab_find(tokenizer, combined, left_length + right_length);
        free(left_bytes);
        free(right_bytes);
        if (result_id < 0) {
            fclose(file);
            return 0;
        }
        uint64_t key = ((uint64_t)(uint32_t)left_id << 32) | (uint32_t)right_id;
        pair_insert(tokenizer, key, rank, result_id);
        rank++;
    }
    fclose(file);
    tokenizer->n_merges = rank;
    return 1;
}

int tgpt_tokenizer_load(TGPTTokenizer *tokenizer, const char *vocab_path, const char *merges_path) {
    memset(tokenizer, 0, sizeof(*tokenizer));
    if (!load_vocab(tokenizer, vocab_path)) {
        fprintf(stderr, "error: vocab load failed\n");
        tgpt_tokenizer_free(tokenizer);
        return 0;
    }
    if (!load_merges(tokenizer, merges_path)) {
        fprintf(stderr, "error: merges load failed\n");
        tgpt_tokenizer_free(tokenizer);
        return 0;
    }
    return 1;
}

void tgpt_tokenizer_free(TGPTTokenizer *tokenizer) {
    if (!tokenizer) {
        return;
    }
    if (tokenizer->token_bytes) {
        for (int i = 0; i < TGPT_VOCAB_LIMIT; i++) {
            free(tokenizer->token_bytes[i]);
        }
    }
    free(tokenizer->token_bytes);
    free(tokenizer->token_len);
    free(tokenizer->vocab_keys);
    free(tokenizer->vocab_ids);
    free(tokenizer->pair_keys);
    free(tokenizer->pair_rank);
    free(tokenizer->pair_result);
    memset(tokenizer, 0, sizeof(*tokenizer));
}

static int contraction_length(const char *text, int available) {
    if (available < 2 || text[0] != '\'') {
        return 0;
    }
    if (text[1] == 's' || text[1] == 't' || text[1] == 'm' || text[1] == 'd') {
        return 2;
    }
    if (available >= 3 && text[1] == 'r' && text[2] == 'e') {
        return 3;
    }
    if (available >= 3 && text[1] == 'v' && text[2] == 'e') {
        return 3;
    }
    if (available >= 3 && text[1] == 'l' && text[2] == 'l') {
        return 3;
    }
    return 0;
}

static int is_letter_byte(unsigned char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c >= 128;
}

static int is_digit_byte(unsigned char c) {
    return c >= '0' && c <= '9';
}

static int is_space_byte(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

static int bpe_encode(const TGPTTokenizer *tokenizer, const unsigned char *word, int length, int *output, int capacity) {
    if (length <= 0) {
        return 0;
    }
    int *symbols = (int *)malloc((size_t)length * sizeof(int));
    if (!symbols) {
        return -1;
    }
    int count = length;
    for (int i = 0; i < length; i++) {
        int id = tokenizer->byte_to_id[word[i]];
        if (id < 0) {
            free(symbols);
            return -1;
        }
        symbols[i] = id;
    }
    while (count > 1) {
        int best_rank = INT_MAX;
        int best_left = -1;
        int best_right = -1;
        int best_result = -1;
        for (int i = 0; i + 1 < count; i++) {
            uint64_t key = ((uint64_t)(uint32_t)symbols[i] << 32) | (uint32_t)symbols[i + 1];
            int rank = 0;
            int result = 0;
            if (pair_find(tokenizer, key, &rank, &result) && rank < best_rank) {
                best_rank = rank;
                best_left = symbols[i];
                best_right = symbols[i + 1];
                best_result = result;
            }
        }
        if (best_rank == INT_MAX) {
            break;
        }
        int *merged = (int *)malloc((size_t)count * sizeof(int));
        if (!merged) {
            free(symbols);
            return -1;
        }
        int written = 0;
        for (int i = 0; i < count;) {
            if (i + 1 < count && symbols[i] == best_left && symbols[i + 1] == best_right) {
                merged[written++] = best_result;
                i += 2;
            } else {
                merged[written++] = symbols[i];
                i++;
            }
        }
        free(symbols);
        symbols = merged;
        count = written;
    }
    if (count > capacity) {
        free(symbols);
        return -1;
    }
    memcpy(output, symbols, (size_t)count * sizeof(int));
    free(symbols);
    return count;
}

int tgpt_encode(const TGPTTokenizer *tokenizer, const char *text, int *ids, int max_ids) {
    int length = (int)strlen(text);
    int position = 0;
    int produced = 0;
    while (position < length) {
        int start = position;
        int end = position;
        int contraction = contraction_length(text + position, length - position);
        if (contraction > 0) {
            end = position + contraction;
        } else {
            int cursor = position;
            if (text[cursor] == ' ' && cursor + 1 < length) {
                cursor++;
            }
            if (cursor < length && is_letter_byte((unsigned char)text[cursor])) {
                int scan = cursor + 1;
                while (scan < length && is_letter_byte((unsigned char)text[scan])) {
                    scan++;
                }
                end = scan;
            } else if (cursor < length && is_digit_byte((unsigned char)text[cursor])) {
                int scan = cursor + 1;
                while (scan < length && is_digit_byte((unsigned char)text[scan])) {
                    scan++;
                }
                end = scan;
            } else if (cursor < length && !is_space_byte((unsigned char)text[cursor])) {
                int scan = cursor + 1;
                while (scan < length && !is_space_byte((unsigned char)text[scan]) &&
                       !is_letter_byte((unsigned char)text[scan]) && !is_digit_byte((unsigned char)text[scan])) {
                    scan++;
                }
                end = scan;
            } else {
                int scan = position;
                while (scan < length && is_space_byte((unsigned char)text[scan])) {
                    scan++;
                }
                end = scan > position ? scan : position + 1;
            }
        }
        int required = end - start;
        if (produced + required > max_ids) {
            return -1;
        }
        int encoded = bpe_encode(tokenizer, (const unsigned char *)text + start, required, ids + produced, max_ids - produced);
        if (encoded < 0) {
            return -1;
        }
        produced += encoded;
        position = end;
    }
    return produced;
}

int tgpt_decode(const TGPTTokenizer *tokenizer, const int *ids, int n, char *out, int max_out) {
    int written = 0;
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        if (id < 0 || id >= tokenizer->vocab_size || !tokenizer->token_bytes[id]) {
            return -1;
        }
        int length = tokenizer->token_len[id];
        if (written + length >= max_out) {
            return -1;
        }
        memcpy(out + written, tokenizer->token_bytes[id], (size_t)length);
        written += length;
    }
    if (max_out > 0) {
        out[written] = 0;
    }
    return written;
}
