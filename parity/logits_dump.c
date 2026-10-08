/* Logit dumper: decodes a prompt + fixed continuation with the reference
 * llama API and prints top-3 logits per step — ground truth for Rust parity.
 *
 * gcc logits_dump.c -I pinned/include -I build-rust-ref/include -o logits_dump \
 *   -L build-rust-ref/bin -lllama -lggml -Wl,-rpath,... (see repo PARITY.md)
 */
#include "llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *model = argv[1];
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    struct llama_model *m = llama_model_load_from_file(model, mp);
    if (!m) { fprintf(stderr, "load failed\n"); return 1; }

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_threads = 8;
    cp.n_threads_batch = 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; // match the Rust non-FA path
    struct llama_context *ctx = llama_init_from_model(m, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    // reference tokenizer crashes on raw strings at this commit
    // (tokenizer_st_partition/get_token_data) — use ids verified via
    // llama-tokenize: "The capital of France is"
    llama_token tokbuf[64] = {785, 6722, 315, 9625, 374};
    const int n_prompt = 5;
    llama_token *toks = tokbuf;

    int n_past = 0;
    // prefill
    llama_batch b = llama_batch_get_one(toks, n_prompt);
    if (llama_decode(ctx, b)) { fprintf(stderr, "prefill failed\n"); return 1; }
    n_past = n_prompt;

    for (int step = 0; step < 8; step++) {
        float *logits = llama_get_logits_ith(ctx, -1);
        int n_vocab = llama_n_vocab(m);
        int top[3] = {-1,-1,-1};
        float topv[3] = {-1e30f,-1e30f,-1e30f};
        for (int i = 0; i < n_vocab; i++) {
            if (logits[i] > topv[0]) {
                topv[2]=topv[1]; top[2]=top[1];
                topv[1]=topv[0]; top[1]=top[0];
                topv[0]=logits[i]; top[0]=i;
            } else if (logits[i] > topv[1]) {
                topv[2]=topv[1]; top[2]=top[1];
                topv[1]=logits[i]; top[1]=i;
            } else if (logits[i] > topv[2]) {
                topv[2]=logits[i]; top[2]=i;
            }
        }
        printf("step %d: top3 (%d, %.6f) (%d, %.6f) (%d, %.6f)\n",
               step, top[0], topv[0], top[1], topv[1], top[2], topv[2]);
        llama_token next = top[0];
        printf("  -> %d\n", next);
        if (llama_token_is_eog(m, next)) break;
        llama_batch nb = llama_batch_get_one(&next, 1);
        if (llama_decode(ctx, nb)) break;
        n_past++;
    }
    llama_free(ctx);
    llama_model_free(m);
    return 0;
}
