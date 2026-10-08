// ref_gemma4_assistant_dump.c — drive the pinned reference's gemma4-assistant
// draft head directly through its library (bd4f514db1), for node-level parity
// against the port's mem-shared draft-mtp mode.
//
// Why a dedicated probe: the pinned reference's own spec driver cannot reach
// this head — common/speculative.cpp:2562 loads `params.model.path` (the
// TARGET) where the `-md` draft path is intended, so `ctx_other` is never
// attached to a gemma4-assistant context through any CLI/server surface.
// This probe constructs the pairing by hand: trunk context first, then the
// assistant context with `cparams.ctx_other = ctx_tgt`
// (llama-context.cpp:147-153), and replays the draft-mtp driver's step shape
// (common/speculative.cpp:1602-1751): seed (id_last, pending_h) at pos0,
// decode, take argmax, feed (id, h_row) at the SAME pos0 (the mem-shared
// rule, :1718-1722).
//
// build (from the repo root):
//   gcc -O2 -I/home/jeffrey/llm/llama.cpp-pinned/include \
//       -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
//       -I/home/jeffrey/llm/llama.cpp-pinned/src \
//       parity/ref_gemma4_assistant_dump.c -o parity/ref_gemma4_assistant_dump \
//       -L/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
//       -lllama -lggml -lggml-base -lggml-cpu -lm \
//       -Wl,-rpath,/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
// run:
//   ./parity/ref_gemma4_assistant_dump <trunk.gguf> <assistant.gguf> <out.bin> \
//          [--text <prompt>] [--steps N] [--fa on|off] [--threads N]
//
// File format (little-endian):
//   char magic[8] = "G4ASST01"
//   u32  n_prompt
//   i32  prompt_tokens[n_prompt]
//   u32  n_steps
//   u32  n_vocab          (the draft's tied lm head width)
//   u32  n_embd_backbone  (the h_nextn row width)
//   f32  trunk_h_last[n_embd_backbone]        — the prefill's last h row
//   per step:
//     i32  fed_token
//     f32  logits[n_vocab]                    — the draft's t_logits row
//     f32  h_next[n_embd_backbone]            — the draft's t_h_nextn row

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

// stream every graph node of the DRAFT decodes (DECDMP1 rules: named nodes
// only carry payloads; over-cap shapes are shape-only) — the port mirrors
// this with its eval callback for the node-level bisect
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
    const int64_t n = ggml_nbytes(t) / ggml_type_size(t->type); // element count via bytes (views incl.)
    const int64_t nel = ggml_nelements(t);
    (void) n;
    fwrite(&nel, sizeof(int64_t), 1, g_nodes);
    if (nel > 0 && nel <= (1 << 19) && (t->type == GGML_TYPE_F32)) {
        // strided walk (views carry parent strides)
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

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <trunk.gguf> <assistant.gguf> <out.bin> [--text <p>] [--steps N] [--fa on|off] [--threads N]\n", argv[0]);
        return 1;
    }
    const char * path_tgt = argv[1];
    const char * path_dft = argv[2];
    const char * out_path = argv[3];
    const char * prompt   = "The capital of France is";
    const char * nodes_path = NULL;
    int n_steps           = 5;
    int n_threads         = 8;
    int fa_on             = 0;

    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--text") && i + 1 < argc) {
            prompt = argv[++i];
        } else if (!strcmp(argv[i], "--steps") && i + 1 < argc) {
            n_steps = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--fa") && i + 1 < argc) {
            i++;
            fa_on = !strcmp(argv[i], "on");
        } else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--nodes") && i + 1 < argc) {
            nodes_path = argv[++i];
        } else {
            fprintf(stderr, "unknown argument %s\n", argv[i]);
            return 1;
        }
    }

    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    struct llama_model * model_tgt = llama_model_load_from_file(path_tgt, mparams);
    if (!model_tgt) { fprintf(stderr, "trunk load failed\n"); return 1; }

    struct llama_model * model_dft = llama_model_load_from_file(path_dft, mparams);
    if (!model_dft) { fprintf(stderr, "assistant load failed\n"); return 1; }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.cb_eval          = nodes_path ? cb_eval : NULL;
    cparams.n_ctx            = 512;
    cparams.n_batch          = 512;
    cparams.n_ubatch         = 512;
    cparams.n_seq_max        = 1;
    cparams.n_threads        = n_threads;
    cparams.n_threads_batch  = n_threads;
    cparams.no_perf          = true;
    cparams.flash_attn_type  = fa_on ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;

    struct llama_context * ctx_tgt = llama_init_from_model(model_tgt, cparams);
    if (!ctx_tgt) { fprintf(stderr, "trunk context failed\n"); return 1; }

    // the gemma4-assistant pairing (llama-context.cpp:147-153)
    cparams.ctx_other = ctx_tgt;
    struct llama_context * ctx_dft = llama_init_from_model(model_dft, cparams);
    if (!ctx_dft) { fprintf(stderr, "assistant context failed (ctx_other)\n"); return 1; }

    // the mtp driver's taps (speculative.cpp:1420-1421)
    llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
    llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

    if (nodes_path) {
        g_nodes = fopen(nodes_path, "wb");
    }

    // tokenize the prompt (add_special = true)
    const struct llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    int n = llama_tokenize(vocab, prompt, (int) strlen(prompt), NULL, 0, true, true);
    if (n <= 0) { n = -n; }
    llama_token * toks = (llama_token *) malloc(sizeof(llama_token) * (size_t) n);
    n = llama_tokenize(vocab, prompt, (int) strlen(prompt), toks, n, true, true);
    if (n <= 0) { fprintf(stderr, "tokenize failed\n"); return 1; }

    // the trunk prefill — every position an output row so the unmasked tap
    // covers all of them
    struct llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        batch.token[i]    = toks[i];
        batch.pos[i]      = (llama_pos) i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0]= 0;
        batch.logits[i]   = 1;
    }
    batch.n_tokens = n;
    if (llama_decode(ctx_tgt, batch) != 0) { fprintf(stderr, "trunk prefill failed\n"); return 1; }

    const int n_embd   = llama_model_n_embd_out(model_dft); // == the trunk's h_nextn width
    const int n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model_dft));
    const float * h_last = llama_get_embeddings_nextn_ith(ctx_tgt, n - 1);
    const float * logits0 = llama_get_logits_ith(ctx_tgt, n - 1);

    // id_last = the trunk's greedy pick (the driver's first draft seed)
    int id_last = 0;
    for (int i = 1; i < n_vocab; i++) {
        if (logits0[i] > logits0[id_last]) { id_last = i; }
    }
    const llama_pos pos0 = (llama_pos) n; // the sampled token's position

    g_out = fopen(out_path, "wb");
    if (!g_out) { fprintf(stderr, "cannot open %s\n", out_path); return 1; }
    fwrite("G4ASST01", 1, 8, g_out);
    put_u32((uint32_t) n);
    for (int i = 0; i < n; i++) { put_i32(toks[i]); }
    put_u32((uint32_t) n_steps);
    put_u32((uint32_t) n_vocab);
    put_u32((uint32_t) n_embd);
    put_f32(h_last, (size_t) n_embd);

    // the draft loop — one token + one h row per step, every step at pos0
    // (the mem-shared rule, speculative.cpp:1718-1722)
    struct llama_batch dbatch = llama_batch_init(1, n_embd, 1);
    dbatch.n_tokens      = 1;
    dbatch.n_seq_id[0]   = 1;
    dbatch.seq_id[0][0]  = 0;
    dbatch.logits[0]     = 1;
    // llama_batch_init allocates only one of token/embd — malloc the other
    dbatch.token = (llama_token *) malloc(sizeof(llama_token));

    for (int s = 0; s < n_steps; s++) {
        dbatch.token[0] = (llama_token) id_last;
        dbatch.pos[0]   = pos0;
        memcpy(dbatch.embd, h_last, sizeof(float) * (size_t) n_embd);

        g_active = 1;
        if (llama_decode(ctx_dft, dbatch) != 0) { fprintf(stderr, "draft step %d failed\n", s); return 1; }
        g_active = 0;

        const float * d_logits = llama_get_logits_ith(ctx_dft, 0);
        const float * d_h      = llama_get_embeddings_nextn_ith(ctx_dft, 0);

        put_i32(id_last);
        put_f32(d_logits, (size_t) n_vocab);
        put_f32(d_h, (size_t) n_embd);

        // the driver's feedback: argmax over the draft logits, the fresh h
        // row pairs it (the p_min gate is skipped — the probe always feeds)
        int next = 0;
        for (int i = 1; i < n_vocab; i++) {
            if (d_logits[i] > d_logits[next]) { next = i; }
        }
        id_last = next;
        h_last  = d_h;
    }

    fclose(g_out);
    fprintf(stderr, "ref_gemma4_assistant_dump: prompt=%d steps=%d n_vocab=%d n_embd=%d -> %s\n",
            n, n_steps, n_vocab, n_embd, out_path);

    // dbatch.token was malloc'd on top of llama_batch_init's embd-only
    // allocation — llama_batch_free frees both pointers
    llama_batch_free(dbatch);
    llama_batch_free(batch);
    free(toks);
    llama_free(ctx_dft);
    llama_free(ctx_tgt);
    llama_model_free(model_dft);
    llama_model_free(model_tgt);
    llama_backend_free();
    return 0;
}
