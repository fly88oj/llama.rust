/* State-kinds dumper: the remaining `llama_state_seq_get_data` /
 * `llama_state_get_data` byte formats @ bd4f514db1 on two synthetic models —
 *
 *   * deepseek32 (DSA) — the dsa pair's lid half
 *     (llama-kv-cache-dsa.cpp:164-172): the K-only MLA base serialization
 *     followed by the K-only indexer-key half;
 *   * minimax-m3 (MSA) — the idx half (llama-kv-cache-msa.cpp:160-168): the
 *     plain base serialization followed by the idx K rows and the
 *     never-written zero V rows;
 *   * both — the whole-context model-info header of llama_state_get_data
 *     (llama-context.cpp:3341-3357: write_string(llm_arch_name(arch)) around
 *     memory->state_write(io) = seq_id -1).
 *
 * The context mirrors the port's tests/state_kinds_e2e.rs exactly: n_ctx 512,
 * n_seq_max 1, n_ubatch 512, flash attention ON (v_trans = !flash_attn — only
 * the v_trans = 0 layout matches the port's always-!v_trans caches), the same
 * fixed 16-token prefill and the same 8 fixed tail steps.
 *
 * Build + run:
 *
 *   gcc -O2 parity/ref_state_kinds.c -o /tmp/ref_state_kinds \
 *     -I /home/jeffrey/llm/llama.cpp-pinned/include \
 *     -I /home/jeffrey/llm/llama.cpp-pinned/ggml/include \
 *     -L /home/jeffrey/llm/llama.cpp/build-rust-ref/bin -lllama \
 *     -Wl,-rpath,/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
 *   /tmp/ref_state_kinds <model.gguf> <out-prefix> [n_tail]
 *
 * Output: <out-prefix>-seq.bin  = 'SQST' u32 len + the seq-0 blob
 *         <out-prefix>-full.bin = 'FULL' u32 len + the whole-context blob
 */
#include "llama.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const llama_token toks[16] = {3, 17, 42, 9, 21, 5, 8, 30, 11, 29, 2, 16, 4, 13, 25, 7};
static const int n_tok = 16;

static int dump_blob(const char * path, const char * magic, const uint8_t * buf, size_t size) {
    FILE * out = fopen(path, "wb");
    if (!out) {
        fprintf(stderr, "cannot open %s\n", path);
        return 1;
    }
    fwrite(magic, 1, 4, out);
    uint32_t len = (uint32_t) size;
    fwrite(&len, sizeof(uint32_t), 1, out);
    fwrite(buf, 1, size, out);
    fclose(out);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <out-prefix> [n_tail]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * out_prefix = argv[2];
    int n_tail = 0;
    if (argc > 3) {
        n_tail = atoi(argv[3]);
    }

    llama_backend_init();

    const struct llama_model_params mparams = llama_model_default_params();
    struct llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "model load failed: %s\n", model_path);
        return 1;
    }

    /* the port's test geometry (state_kinds_e2e.rs): FA on, one sequence */
    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 512;
    cparams.n_batch         = 512;
    cparams.n_ubatch        = 512;
    cparams.n_seq_max       = 1;
    cparams.n_threads       = 8;
    cparams.n_threads_batch = 8;
    cparams.no_perf         = true;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "context init failed\n");
        return 1;
    }

    struct llama_batch batch = llama_batch_init(n_tok, 0, 1);
    for (int i = 0; i < n_tok; i++) {
        batch.token[i]     = toks[i];
        batch.pos[i]       = (llama_pos) i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 0;
    }
    batch.logits[n_tok - 1] = 1;
    batch.n_tokens = n_tok;
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "prefill decode failed\n");
        return 1;
    }

    /* the port dumper's fixed tail: token toks[s % n] at position n + s */
    for (int s = 0; s < n_tail; s++) {
        struct llama_batch db = llama_batch_init(1, 0, 1);
        db.n_tokens      = 1;
        db.token[0]      = toks[s % n_tok];
        db.pos[0]        = (llama_pos) (n_tok + s);
        db.n_seq_id[0]   = 1;
        db.seq_id[0][0]  = 0;
        db.logits[0]     = 1;
        if (llama_decode(ctx, db) != 0) {
            fprintf(stderr, "tail decode %d failed\n", s);
            return 1;
        }
        llama_batch_free(db);
    }

    /* the sequence-0 blob */
    {
        const size_t size = llama_state_seq_get_size(ctx, 0);
        uint8_t * buf = (uint8_t *) malloc(size);
        if (!buf) {
            fprintf(stderr, "oom (%zu bytes)\n", size);
            return 1;
        }
        const size_t got = llama_state_seq_get_data(ctx, buf, size, 0);
        if (got != size) {
            fprintf(stderr, "seq state size mismatch: %zu vs %zu\n", got, size);
            return 1;
        }
        char path[512];
        snprintf(path, sizeof(path), "%s-seq.bin", out_prefix);
        if (dump_blob(path, "SQST", buf, size) != 0) {
            return 1;
        }
        printf("seq state: %zu bytes -> %s (tail %d)\n", size, path, n_tail);
        free(buf);
    }

    /* the whole-context blob (the arch-string header + memory->state_write) */
    {
        const size_t size = llama_state_get_size(ctx);
        uint8_t * buf = (uint8_t *) malloc(size);
        if (!buf) {
            fprintf(stderr, "oom (%zu bytes)\n", size);
            return 1;
        }
        const size_t got = llama_state_get_data(ctx, buf, size);
        if (got != size) {
            fprintf(stderr, "full state size mismatch: %zu vs %zu\n", got, size);
            return 1;
        }
        char path[512];
        snprintf(path, sizeof(path), "%s-full.bin", out_prefix);
        if (dump_blob(path, "FULL", buf, size) != 0) {
            return 1;
        }
        printf("full state: %zu bytes -> %s\n", size, path);
        free(buf);
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
