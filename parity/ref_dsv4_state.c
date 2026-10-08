/* DSV4 sequence-state dumper: loads the synthetic deepseek4 model, decodes a
 * fixed 16-token prompt on sequence 0, then writes the raw
 * `llama_state_seq_get_data` blob (the exact bytes /slots save-restore and
 * the speculative checkpointing exchange) to parity/dsv4_state_ref.bin — the
 * byte-format oracle for the Rust port's `DecodeContext::state_seq_get_data`
 * (crates/llama/src/kv_cache.rs's `llama_state_seq_*` section + context.rs).
 *
 * The context mirrors the port's test driver exactly (dsv4_state_e2e.rs):
 * n_ctx 512, n_seq_max 1 (kv_unified, one compressed stream), n_ubatch 512
 * and flash attention ON — the dsv4 raw pair is built with
 * `v_trans = !flash_attn` (llama-model.cpp:2480), and only the v_trans = 0
 * layout matches the port's always-!v_trans caches.
 *
 * Build + run (the reference tree is pinned at bd4f514db1):
 *
 *   gcc -O2 parity/ref_dsv4_state.c -o /tmp/ref_dsv4_state \
 *     -I /home/jeffrey/llm/llama.cpp-pinned/include \
 *     -I /home/jeffrey/llm/llama.cpp-pinned/ggml/include \
 *     -L /home/jeffrey/llm/llama.cpp/build-rust-ref/bin -lllama \
 *     -Wl,-rpath,/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
 *   /tmp/ref_dsv4_state <model.gguf> <out.bin> [n_decode_tail]
 *
 * Output layout: u32 magic 'DS4S', u32 blob_len, then the blob bytes.
 */
#include "llama.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <out.bin> [n_decode_tail]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * out_path   = argv[2];
    int n_decode_tail = 0;
    if (argc > 3) {
        n_decode_tail = atoi(argv[3]);
    }

    llama_backend_init();

    const struct llama_model_params mparams = llama_model_default_params();
    struct llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "model load failed: %s\n", model_path);
        return 1;
    }

    /* the port's test driver geometry (dsv4_state_e2e.rs driver_for):
     * n_ctx 512, n_ubatch 512, one sequence, FA on */
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

    /* the same fixed token stream the port decodes (no tokenizer involved) */
    const llama_token toks[16] = {3, 17, 42, 9, 21, 5, 8, 30, 11, 29, 2, 16, 4, 13, 25, 7};
    const int n_tok = 16;

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

    /* optional tail steps (single-token batches) — the state then carries
     * rolled compressor rings, past block boundaries. The fed token is
     * FIXED (the prompt's (s mod n)-th id), matching the port dumper. */
    for (int s = 0; s < n_decode_tail; s++) {
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

    /* the sequence-0 state blob */
    const size_t size = llama_state_seq_get_size(ctx, 0);
    uint8_t * buf = (uint8_t *) malloc(size);
    if (!buf) {
        fprintf(stderr, "oom (%zu bytes)\n", size);
        return 1;
    }
    const size_t got = llama_state_seq_get_data(ctx, buf, size, 0);
    if (got != size) {
        fprintf(stderr, "state size mismatch: %zu vs %zu\n", got, size);
        return 1;
    }

    FILE * out = fopen(out_path, "wb");
    if (!out) {
        fprintf(stderr, "cannot open %s\n", out_path);
        return 1;
    }
    fwrite("DS4S", 1, 4, out);
    uint32_t len = (uint32_t) size;
    fwrite(&len, sizeof(uint32_t), 1, out);
    fwrite(buf, 1, size, out);
    fclose(out);

    printf("dsv4 seq state: %zu bytes -> %s (tail %d)\n", size, out_path, n_decode_tail);

    free(buf);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
