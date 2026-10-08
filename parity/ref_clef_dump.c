// ref_clef_dump.c — the clef decision model's whole-output dump from the NEW
// reference (a7b94df2c), for tests/clef_e2e.rs.
//
// Drives the same public path the reference server's /v1/systemone uses
// (tools/server/server-decision.cpp): a llama_batch_ext with
// llama_batch_ext_set_decision_order marks, one llama_process(DECODER) call
// over the whole batch (the arch creates no memory, llama-model.cpp
// create_memory → nullptr for LLM_ARCH_CLEF), then llama_get_embeddings()
// returns res->t_embd — the [1, n_tokens] decision scores (row i = option i,
// the pad rows beyond n_options, + the NaN status addend).
//
// Every entry is marked output=true so n_outputs == n_tokens and the
// embeddings buffer carries the full tensor.
//
// build (from the repo root; C++ — see the staging-API note below):
//   g++ -O2 -x c++ -I/home/jeffrey/llm/llama.cpp-next/include \
//       -I/home/jeffrey/llm/llama.cpp-next/ggml/include \
//       parity/ref_clef_dump.c -o /tmp/s2m-clef/ref_clef_dump \
//       -L/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin \
//       -lllama -lggml -lggml-base -lggml-cpu -lm \
//       -Wl,-rpath,/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin
// run:
//   ./ref_clef_dump <model.gguf> <out.bin> [--threads N] [--fa on|off]
//
// File format (little-endian):
//   char magic[8] = "CLEFSC1\0"
//   u32  n_tokens
//   u32  n_options
//   f32  scores[n_tokens]

#include "llama.h"

// the decision-order enum + its setter are staging-API symbols
// (src/llama-ext.h:107-116), not part of the installed llama.h yet, and the
// reference .so exports the setter with C++ linkage — so this probe
// compiles as C++ (g++ -x c++) to link the mangled symbol
enum llama_decision_order {
    LLAMA_DECISION_ORDER_NONE            = 0,
    LLAMA_DECISION_ORDER_QUESTION_NOUL   = 1,
    LLAMA_DECISION_ORDER_QUESTION_CHOICE = 2,
    LLAMA_DECISION_ORDER_QUESTION_SCORE  = 3,
    LLAMA_DECISION_ORDER_OPTION          = 4,
};
bool llama_batch_ext_set_decision_order(struct llama_batch_ext * batch, int32_t idx, enum llama_decision_order order);

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char ** argv) {
    (void) argc; (void) argv;
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <out.bin> [--threads N] [--fa on|off]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * out_path   = argv[2];
    int n_threads = 4;
    int fa_on     = 0;

    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--fa") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "on")) {
                fa_on = 1;
            } else {
                fa_on = 0;
            }
        } else {
            fprintf(stderr, "unknown argument %s\n", argv[i]);
            return 1;
        }
    }

    // the probe batch: [NONE x3, QUESTION_CHOICE x3, OPTION x3,
    // QUESTION_SCORE x2, OPTION x1] — two questions, two options (the
    // canonical shape of tests/clef_e2e.rs)
    const int tokens[12] = { 11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 51 };
    const enum llama_decision_order orders_c[12] = {
        LLAMA_DECISION_ORDER_NONE,     LLAMA_DECISION_ORDER_NONE,     LLAMA_DECISION_ORDER_NONE,
        LLAMA_DECISION_ORDER_QUESTION_CHOICE, LLAMA_DECISION_ORDER_QUESTION_CHOICE, LLAMA_DECISION_ORDER_QUESTION_CHOICE,
        LLAMA_DECISION_ORDER_OPTION,   LLAMA_DECISION_ORDER_OPTION,   LLAMA_DECISION_ORDER_OPTION,
        LLAMA_DECISION_ORDER_QUESTION_SCORE,  LLAMA_DECISION_ORDER_QUESTION_SCORE,
        LLAMA_DECISION_ORDER_OPTION,
    };
    const int n_tokens = 12;

    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    struct llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "model load failed\n");
        return 1;
    }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 512;
    cparams.n_batch         = 2048;
    cparams.n_ubatch        = 2048;
    cparams.n_seq_max       = 1;
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.embeddings      = true;
    cparams.pooling_type    = LLAMA_POOLING_TYPE_NONE;
    cparams.no_perf         = true;
    if (fa_on) {
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    } else {
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "context init failed\n");
        return 1;
    }

    // the decision-ordered batch (the /v1/systemone path)
    struct llama_batch_ext * batch = llama_batch_ext_init(ctx);
    for (int i = 0; i < n_tokens; i++) {
        const int32_t idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
        if (idx != i) {
            fprintf(stderr, "add_token(%d) = %d\n", i, idx);
            return 1;
        }
        llama_pos pos[1] = { i };
        if (!llama_batch_ext_set_pos(batch, idx, pos)) {
            fprintf(stderr, "set_pos(%d) failed\n", i);
            return 1;
        }
        if (!llama_batch_ext_set_output_embd(batch, idx, true)) {
            fprintf(stderr, "set_output_embd(%d) failed\n", i);
            return 1;
        }
        if (!llama_batch_ext_set_decision_order(batch, idx, orders_c[i])) {
            fprintf(stderr, "set_decision_order(%d) failed\n", i);
            return 1;
        }
    }

    const int32_t rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
    if (rc != 0) {
        fprintf(stderr, "llama_process = %d\n", rc);
        return 1;
    }

    const float * embd = llama_get_embeddings(ctx);
    if (!embd) {
        fprintf(stderr, "llama_get_embeddings returned NULL\n");
        return 1;
    }

    // n_options = the number of OPTION spans (clef_get_spans' rule)
    int orders[12];
    for (int i = 0; i < n_tokens; i++) {
        orders[i] = (int) orders_c[i];
    }
    int n_options = 0;
    for (int i = 0; i < n_tokens; i++) {
        if (orders[i] == (int) LLAMA_DECISION_ORDER_OPTION &&
            (i == 0 || orders[i - 1] != (int) LLAMA_DECISION_ORDER_OPTION)) {
            n_options++;
        }
    }

    FILE * out = fopen(out_path, "wb");
    if (!out) {
        fprintf(stderr, "fopen(%s) failed\n", out_path);
        return 1;
    }
    fwrite("CLEFSC1\0", 1, 8, out);
    uint32_t n32 = (uint32_t) n_tokens;
    fwrite(&n32, sizeof(uint32_t), 1, out);
    n32 = (uint32_t) n_options;
    fwrite(&n32, sizeof(uint32_t), 1, out);
    // t_embd is [n_embd_out=1, n_tokens]: one score row per entry
    fwrite(embd, sizeof(float), (size_t) n_tokens, out);
    fclose(out);

    llama_batch_ext_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    fprintf(stderr, "wrote %s: %d tokens, %d options\n", out_path, n_tokens, n_options);
    return 0;
}
