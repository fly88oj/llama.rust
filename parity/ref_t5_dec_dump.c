// ref_t5_dec_dump.c — drive the pinned reference's own T5 encoder-decoder
// pair (bd4f514db1) over the t5 synth file of tests/t5_dec_e2e.rs: one
// context, `llama_encode` (which captures the encoder state into
// `llama_context::cross`, llama-context.cpp:1625-1649) then the
// `llama_decode` chain (graph<false>, t5.cpp:110-262) — dec_start_token at
// pos 0, the argmax feeding each next step.
//
// build (from the repo root):
//   g++ -O2 -I/home/jeffrey/llm/llama.cpp-pinned/include \
//       -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
//       -I/home/jeffrey/llm/llama.cpp-pinned/src \
//       parity/ref_t5_dec_dump.c -o parity/ref_t5_dec_dump \
//       -L/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
//       -lllama -lggml -lggml-base -lggml-cpu -lm \
//       -Wl,-rpath,/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
// run: ./parity/ref_t5_dec_dump <t5.gguf> <out.bin> [--steps N] [--threads N]
//
// File format (mirrored byte-for-byte by the port's t5_dec_e2e dump):
//   char magic[8] = "T5DEC\0\0\0"
//   u32  n_steps, u32 n_vocab
//   per step: i32 fed_token, f32 logits[n_vocab]

#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "llama.h"
#include "ggml.h"
#include "ggml-cpu.h"

static FILE * g_nodes = NULL;
static bool cb_eval(struct ggml_tensor * t, bool ask, void *) {
    if (ask) return true;
    if (!g_nodes) return true;
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
    if (nel > 0 && nel <= (1 << 19) && t->type == GGML_TYPE_F32) {
        for (int64_t i = 0; i < nel; i++) {
            float v;
            memcpy(&v, (const char *) t->data + i * t->nb[0], sizeof(float));
            fwrite(&v, sizeof(float), 1, g_nodes);
        }
    }
    return true;
}

static FILE * g_out = NULL;

static void put_u32(uint32_t v) { fwrite(&v, sizeof(v), 1, g_out); }
static void put_i32(int32_t v)  { fwrite(&v, sizeof(v), 1, g_out); }
static void put_f32(const float * p, size_t n) { fwrite(p, sizeof(float), n, g_out); }

static int argmax_first(const float * v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <t5.gguf> <out.bin> [--steps N] [--threads N]\n", argv[0]);
        return 1;
    }
    const char * path = argv[1];
    const char * out_path = argv[2];
    int n_steps = 12, n_threads = 8;
    const char * nodes_path = NULL;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--steps") && i + 1 < argc) n_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--nodes") && i + 1 < argc) nodes_path = argv[++i];
    }

    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(path, mparams);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 512;
    cparams.n_batch         = 512;
    cparams.n_ubatch        = 512;
    cparams.n_seq_max       = 1;
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.no_perf         = true;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; // the port runs non-FA
    if (nodes_path) {
        g_nodes = fopen(nodes_path, "wb");
        cparams.cb_eval = cb_eval;
    }

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    g_out = fopen(out_path, "wb");
    fwrite("T5DEC\0\0\0", 1, 8, g_out);
    put_u32((uint32_t) n_steps);
    put_u32((uint32_t) n_vocab);

    // 1) encode the prompt (the same six tokens the port's driver feeds) —
    //    this fills llama_context::cross with the encoder's final state
    {
        struct llama_batch eb = llama_batch_init(6, 0, 1);
        for (int i = 0; i < 6; i++) {
            eb.token[i]     = i + 1;
            eb.pos[i]       = (llama_pos) i; // the relative buckets read them (bidirectional)
            eb.n_seq_id[i]  = 1;
            eb.seq_id[i][0] = 0;
            eb.logits[i]    = 0;
        }
        eb.n_tokens = 6;
        if (llama_encode(ctx, eb) != 0) { fprintf(stderr, "encode failed\n"); return 1; }
        llama_batch_free(eb);
        // the cross state the decoder will read (llama_context::cross) —
        // dump it for the encoder-side bisect
        {
            const float * enc = llama_get_embeddings(ctx);
            FILE * ef = fopen("/tmp/mtp2/t5-enc-ref.bin", "wb");
            fwrite(enc, sizeof(float), 6 * llama_model_n_embd(model), ef);
            fclose(ef);
        }
    }

    // 2) the decode chain
    struct llama_batch batch = llama_batch_init(1, 0, 1);
    llama_token tok = llama_model_decoder_start_token(model);
    if (tok < 0) tok = 1;
    for (int step = 0; step < n_steps; step++) {
        batch.n_tokens     = 1;
        batch.token[0]     = tok;
        batch.pos[0]       = (llama_pos) step;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = 1;

        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed at %d\n", step); return 1; }
        const float * logits = llama_get_logits_ith(ctx, 0);
        put_i32(tok);
        put_f32(logits, (size_t) n_vocab);
        tok = (llama_token) argmax_first(logits, n_vocab);
    }

    fclose(g_out);
    std::_Exit(0);
}
