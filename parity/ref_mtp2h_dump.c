// ref_mtp2_dump.c — drive the pinned reference's own MTP draft context
// (bd4f514db1) over the mtp2 batch's synthetic nextn files, for bit-level
// parity against the port's graph-level driver (tests/mtp2_e2e.rs).
//
// The construction mirrors common/speculative.cpp:2545-2549 (the draft-mtp
// context creation): mparams.load_mtp = true (what --spec-type draft-mtp sets
// through common.cpp:1713) + cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP. The
// step shape is the draft loop's non-chained, non-mem-shared arm
// (speculative.cpp:1616-1767): one token + one F32 h row per decode, the
// previous step's argmax + t_h_nextn row feeding the next — the identical
// chain the port's Mtp2Driver replays.
//
// build (from the repo root):
//   g++ -O2 -I/home/jeffrey/llm/llama.cpp-pinned/include \
//       -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
//       -I/home/jeffrey/llm/llama.cpp-pinned/src \
//       parity/ref_mtp2_dump.c -o parity/ref_mtp2_dump \
//       -L/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
//       -lllama -lggml -lggml-base -lggml-cpu -lm \
//       -Wl,-rpath,/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
// run:
//   ./parity/ref_mtp2h_dump <arch.gguf> <out.bin> [--steps N] [--threads N]
//
// File format (little-endian, mirrored byte-for-byte by the port):
//   char magic[8] = "MTP2P\0\0\0"
//   u32  n_steps
//   u32  n_vocab
//   u32  n_embd
//   per step:
//     i32  fed_token
//     f32  logits[n_vocab]   — the draft's t_logits row
//     f32  h_next[n_embd]    — the draft's t_h_nextn row

#include "llama.h"
#include "llama-ext.h"
#include "ggml.h"
#include "ggml-cpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static FILE * g_out   = NULL;
static FILE * g_nodes = NULL;
static int    g_active = 0;

// DECDMP1 rules (the gemma4 probe's cb_eval): named nodes carry payloads,
// over-cap shapes are shape-only
static bool cb_eval(struct ggml_tensor * t, bool ask, void * /*user_data*/) {
    if (ask) {
        return true;
    }
    if (!g_active || !g_nodes) {
        return true;
    }
    const char * name = ggml_get_name(t);
    uint8_t l = (uint8_t) (strlen(ggml_op_desc(t)) & 0xff);
    fwrite(&l, 1, 1, g_nodes); fwrite(ggml_op_desc(t), 1, l, g_nodes);
    l = (uint8_t) (strlen(name) & 0xff);
    fwrite(&l, 1, 1, g_nodes); fwrite(name, 1, l, g_nodes);
    l = (uint8_t) (strlen(ggml_type_name(t->type)) & 0xff);
    fwrite(&l, 1, 1, g_nodes); fwrite(ggml_type_name(t->type), 1, l, g_nodes);
    fwrite(t->ne, sizeof(int64_t), 4, g_nodes);
    const int64_t nel = ggml_nelements(t);
    fwrite(&nel, sizeof(int64_t), 1, g_nodes);
    if (nel > 0 && nel <= (1 << 19) && (t->type == GGML_TYPE_F32)) {
        const int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
        const size_t nb0 = t->nb[0], nb1 = t->nb[1], nb2 = t->nb[2], nb3 = t->nb[3];
        const char * base = (const char *) t->data;
        for (int64_t i3 = 0; i3 < ne3; i3++)
        for (int64_t i2 = 0; i2 < ne2; i2++)
        for (int64_t i1 = 0; i1 < ne1; i1++) {
            const char * row = base + i3*nb3 + i2*nb2 + i1*nb1;
            for (int64_t i0 = 0; i0 < ne0; i0++) {
                float v;
                memcpy(&v, row + i0*nb0, sizeof(float));
                fwrite(&v, sizeof(float), 1, g_nodes);
            }
        }
    }
    return true;
}

static void put_u32(uint32_t v) { fwrite(&v, sizeof(v), 1, g_out); }
static void put_i32(int32_t v)  { fwrite(&v, sizeof(v), 1, g_out); }
static void put_f32(const float * p, size_t n) { fwrite(p, sizeof(float), n, g_out); }

static int argmax_first(const float * v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (v[i] > v[best]) {
            best = i;
        }
    }
    return best;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <arch.gguf> <out.bin> [--steps N] [--threads N]\n", argv[0]);
        return 1;
    }
    const char * path = argv[1];
    const char * out_path = argv[2];
    int n_steps   = 12;
    int n_threads = 8;
    const char * nodes_path = NULL;

    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--steps") && i + 1 < argc) {
            n_steps = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--nodes") && i + 1 < argc) {
            nodes_path = argv[++i];
        }
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.load_mtp = true; // common.cpp:1713's --spec-type draft-mtp effect

    llama_model * model = llama_model_load_from_file(path, mparams);
    if (model == NULL) {
        fprintf(stderr, "load failed: %s\n", path);
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.ctx_type        = LLAMA_CONTEXT_TYPE_MTP; // speculative.cpp:2546
    cparams.n_ctx           = 512;
    cparams.n_batch         = 512;
    cparams.n_ubatch        = 512;
    cparams.n_seq_max       = 1;
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.no_perf         = true;
    // the port-side driver runs the non-FA path (use_flash_attn = false);
    // the reference's AUTO default enables the CPU flash-attn kernel here,
    // whose online softmax differs from the non-FA path at ~1e-4 — disable
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (nodes_path) {
        g_nodes = fopen(nodes_path, "wb");
        cparams.cb_eval = cb_eval;
    }

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == NULL) {
        fprintf(stderr, "context init failed\n");
        return 1;
    }

    // the t_h_nextn tap (llama-context.cpp:1228-1233)
    llama_set_embeddings_nextn(ctx, true, false);

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n_embd  = llama_model_n_embd_out(model); // the hc-wide MTP archs (qwen4exp)

    g_out = fopen(out_path, "wb");
    if (g_out == NULL) {
        fprintf(stderr, "open %s failed\n", out_path);
        return 1;
    }
    // the header is finalized after the chain (n_steps is known, but write it
    // now — the count is fixed)
    fwrite("MTP2P\0\0\0", 1, 8, g_out);
    put_u32((uint32_t) n_steps);
    put_u32((uint32_t) n_vocab);
    put_u32((uint32_t) n_embd);

    // llama_batch_init allocates only ONE of token/embd; the draft-mtp step
    // feeds both — the reference's own workaround (speculative.cpp:483-485)
    struct llama_batch batch = llama_batch_init(1, n_embd, 1);
    batch.token = (llama_token *) malloc(sizeof(llama_token));

    llama_token tok = 1;
    float * h_buf = (float *) calloc((size_t) n_embd, sizeof(float));

    g_active = 1; // the draft decodes only (skip the init reserve)
    for (int step = 0; step < n_steps; step++) {
        if (getenv("MTP2_FORCE_REBUILD")) {
            // toggling the tap invalidates the cached graph so cb_eval fires
            // for every step (the per-step node bisect)
            llama_set_embeddings_nextn(ctx, false, false);
            llama_set_embeddings_nextn(ctx, true, false);
        }
        batch.n_tokens      = 1;
        batch.token[0]      = tok;
        batch.pos[0]        = (llama_pos) step;
        batch.n_seq_id[0]   = 1;
        batch.seq_id[0][0]  = 0;
        batch.logits[0]     = 1;
        memcpy(batch.embd, h_buf, (size_t) n_embd * sizeof(float));

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "llama_decode failed at step %d\n", step);
            return 1;
        }

        const float * logits = llama_get_logits_ith(ctx, 0);
        const float * h_next = llama_get_embeddings_nextn_ith(ctx, 0);

        put_i32(tok);
        put_f32(logits, (size_t) n_vocab);
        put_f32(h_next, (size_t) n_embd);

        tok = (llama_token) argmax_first(logits, n_vocab);
        memcpy(h_buf, h_next, (size_t) n_embd * sizeof(float));
    }

    fclose(g_out);
    if (g_nodes) {
        fclose(g_nodes);
    }
    // no teardown: batch.token was hand-malloc'd beside llama_batch_init's
    // embd allocation, and freeing both sides trips the allocator's
    // fastbin check at exit (the reference hits the same pair,
    // speculative.cpp:483-485); the dump is complete — just leave
    std::_Exit(0);
}
