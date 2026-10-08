// ref_decode_dump.c — stream-dump EVERY graph node of one reference decoder
// prefill (bd4f514db1), for node-by-node bisection of the port's multi-token
// prefill divergence (PARITY.md "decoder -fe per-token embeddings" case).
//
// The probe drives libllama exactly like `llama-server --embeddings
// --pooling none` does for one /embedding request (server-context.cpp:2864 +
// 2152-2192): llama_decode of all tokens in a single ubatch, every token an
// output row, cparams.embeddings = true. The cb_eval callback
// (ggml-backend.cpp:1798-1835) then hands us every computed node in graph
// order; each node's elements are written through a strided walk (views have
// parent strides, a flat memcpy would mix rows) as f32 — F16 -> F32 is exact.
//
// build (from the repo root):
//   gcc -O2 -I/home/jeffrey/llm/llama.cpp-pinned/include \
//       -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
//       parity/ref_decode_dump.c -o parity/ref_decode_dump \
//       -L/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
//       -lllama -lggml -lggml-base -lggml-cpu -lm \
//       -Wl,-rpath,/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
// run:
//   ./parity/ref_decode_dump <model.gguf> <out.bin> [--text <prompt>]
//                            [--threads N] [--fa on|off]
//
// File format (little-endian; consumed by parity/decode_dump_cmp.py and the
// Rust mirror in crates/llama/tests/qwen3_prefill_dump.rs):
//   char  magic[8] = "DECDMP1\0"
//   u32   n_tokens
//   i32   tokens[n_tokens]
//   u32   n_nodes
//   per node:
//     u8  op_len,  char  op[op_len]     // ggml_op_desc(t), e.g. "MUL_MAT"
//     u8  name_len,char  name[name_len] // graph tensor name (may be empty)
//     u8  type_len,char  type[type_len] // ggml_type_name(t->type)
//     i64 ne[4]
//     u64 n_elems
//     f32 elems[n_elems]   // strided walk, F16 exact — ONLY when
//                          // n_elems <= ELEM_CAP; the KV-cache set_rows views
//                          // (whole-cache shapes) exceed it and are skipped by
//                          // this rule on BOTH sides, so the streams align.
//
// No string literals inside ternaries (repo scanner rule); if/else only.

#include "llama.h"
#include "ggml.h"
#include "ggml-cpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static FILE * g_out   = NULL;
static int    g_nodes = 0;
static int    g_active = 0;

// nodes larger than this carry no element payload in the stream — the cap sits
// between the per-ubatch intermediates (<= 8x1024 here) and the whole-layer
// KV-cache views (524288) / the tied lm head (1.2M); shape-deterministic, so
// the Rust mirror omits exactly the same nodes
#define ELEM_CAP (1ull << 19)

static void put_str(const char * s) {
    const size_t len = strlen(s);
    fputc(len > 255 ? 255 : (int) len, g_out);
    fwrite(s, 1, len > 255 ? 255 : len, g_out);
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    (void) user_data;
    if (!g_active) {
        return false; // ignore graph_reserve / warmup passes
    }
    if (ask) {
        return true; // want every node
    }

    const int64_t ne[4] = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
    int64_t n = 1;
    for (int d = 0; d < 4; d++) {
        n *= ne[d];
    }

    // header first (so the comparator can seek), then the strided walk
    g_nodes++;
    put_str(ggml_op_desc(t));
    put_str(t->name ? t->name : "");
    put_str(ggml_type_name(t->type));
    fwrite(ne, sizeof(int64_t), 4, g_out);
    uint64_t nu = (uint64_t) n;
    fwrite(&nu, sizeof(uint64_t), 1, g_out);

    if ((t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16) && (uint64_t) n < ELEM_CAP) {
        // strided element walk: for a view, nb[] strides belong to the parent
        for (int64_t flat = 0; flat < n; flat++) {
            int64_t rem = flat;
            size_t  off = 0;
            for (int d = 0; d < 4; d++) {
                const int64_t idx = rem % ne[d];
                rem /= ne[d];
                off += (size_t) idx * t->nb[d];
            }
            const char * p = (const char *) t->data + off;
            float v;
            if (t->type == GGML_TYPE_F32) {
                memcpy(&v, p, sizeof(float));
            } else {
                ggml_fp16_t h;
                memcpy(&h, p, sizeof(h));
                v = ggml_fp16_to_fp32(h);
            }
            fwrite(&v, sizeof(float), 1, g_out);
        }
    } else if ((uint64_t) n < ELEM_CAP) {
        // non-float node (never happens in the decoder forward today): the
        // comparator reports it from the shape/type header alone
        float v = 0.0f;
        for (int64_t i = 0; i < n; i++) {
            fwrite(&v, sizeof(float), 1, g_out);
        }
    }
    // over-cap nodes (KV-cache views): no payload at all
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <out.bin> [--text <p>] [--threads N] [--fa on|off]\n",
                argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * out_path   = argv[2];
    const char * prompt     = "The capital of France is Paris.";
    int  n_threads = 8;
    int  fa_on     = 0;

    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--text") && i + 1 < argc) {
            prompt = argv[++i];
        } else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
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

    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    struct llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "model load failed\n");
        return 1;
    }
    const struct llama_vocab * vocab = llama_model_get_vocab(model);

    // /embedding tokenizes with add_special = true (server-context.cpp:tokenize)
    const size_t plen = strlen(prompt);
    int nt = llama_tokenize(vocab, prompt, (int) plen, NULL, 0, true, true);
    if (nt <= 0) {
        nt = -nt; // negative return = required capacity
    }
    if (nt <= 0) {
        fprintf(stderr, "tokenize failed: %d\n", nt);
        return 1;
    }
    llama_token * toks = (llama_token *) malloc(sizeof(llama_token) * (size_t) nt);
    nt = llama_tokenize(vocab, prompt, (int) plen, toks, nt, true, true);
    if (nt <= 0) {
        fprintf(stderr, "tokenize(2) failed: %d\n", nt);
        return 1;
    }

    // the server's context (parity/embd_rows_probe.sh): -c 512 -t 8, batch
    // defaults (n_batch 2048 >= n_ubatch 512 >= 8 tokens -> one ubatch)
    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx            = 512;
    cparams.n_batch          = nt; // sized to the prompt — see the header note
    cparams.n_ubatch         = nt;
    cparams.n_seq_max        = 1;
    cparams.n_threads        = n_threads;
    cparams.n_threads_batch  = n_threads;
    if (getenv("DSV4_NOEMB") != NULL) {
        cparams.embeddings = false; // plain decode (logits only)
    } else {
        cparams.embeddings       = true;      // --embeddings
    }
    cparams.pooling_type     = LLAMA_POOLING_TYPE_NONE; // --pooling none
    cparams.no_perf          = true;
    if (fa_on) {
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    } else {
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
    if (getenv("DSV4_NOCB") == NULL) {
        cparams.cb_eval           = cb_eval;
        cparams.cb_eval_user_data = NULL;
    }

    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "context init failed\n");
        return 1;
    }

    g_out = fopen(out_path, "wb");
    if (!g_out) {
        fprintf(stderr, "cannot open %s\n", out_path);
        return 1;
    }
    fwrite("DECDMP1\0", 1, 8, g_out);
    uint32_t ntu = (uint32_t) nt;
    fwrite(&ntu, sizeof(uint32_t), 1, g_out);
    fwrite(toks, sizeof(llama_token), (size_t) nt, g_out);
    const uint32_t nodes_zero = 0;
    fwrite(&nodes_zero, sizeof(uint32_t), 1, g_out); // patched after the run

    struct llama_batch batch = llama_batch_init(nt, 0, 1);
    for (int i = 0; i < nt; i++) {
        batch.token[i]     = toks[i];
        batch.pos[i]       = (llama_pos) i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1; // every token an output row (output_all)
    }
    batch.n_tokens = nt;

    g_active = 1;
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }
    g_active = 0;

    // patch the node count into the header
    long cur = ftell(g_out);
    fseek(g_out, (long) (8 + 4 + 4 * (size_t) nt), SEEK_SET);
    uint32_t nn = (uint32_t) g_nodes;
    fwrite(&nn, sizeof(uint32_t), 1, g_out);
    fseek(g_out, cur, SEEK_SET);
    fclose(g_out);

    fprintf(stderr, "ref_decode_dump: %s tokens=%d nodes=%d -> %s\n",
            model_path, nt, g_nodes, out_path);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
