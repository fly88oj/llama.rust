// ref_lm_tts_dump.c — reference-side probe of the LM TTS archs
// (wavtokenizer-dec / pockettts / qwen3tts), pinned bd4f514db1.
//
// Loads the synthetic GGUF through libllama and decodes ONE fixed token
// sequence (tokens.bin: [u32 n][i32 ids]) in a single ubatch, then dumps
// the last row's logits as [u32 n_vocab][f32 * n_vocab]. With
// `--embeddings`, also dumps `llama_get_embeddings`' last row (t_embd,
// pooling NONE) as [u32 n_out][f32 * n_out] — the wavtokenizer-dec t_embd
// is the decoded waveform frame.
//
// build (from the repo root):
//   gcc -O2 -I/home/jeffrey/llm/llama.cpp-pinned/include \
//       -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
//       parity/ref_lm_tts_dump.c -o parity/ref_lm_tts_dump \
//       -L/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
//       -lllama -lggml -lggml-base -lggml-cpu -lm \
//       -Wl,-rpath,/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
// run:
//   ./parity/ref_lm_tts_dump <model.gguf> <tokens.bin> <out.bin> [--embeddings]
//                            [--threads N] [--fa on|off]

#include "llama.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model> <tokens.bin> <out.bin> [--embeddings] [--threads N] [--fa on|off]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * tok_path   = argv[2];
    const char * out_path   = argv[3];
    int want_embd = 0;
    int n_threads = 4;
    int fa_on = 1;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--embeddings")) want_embd = 1;
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fa") && i + 1 < argc) fa_on = !strcmp(argv[++i], "on");
    }

    FILE * tf = fopen(tok_path, "rb");
    if (!tf) { perror("tokens fopen"); return 1; }
    uint32_t n_tok = 0;
    if (fread(&n_tok, 4, 1, tf) != 1) return 1;
    int32_t * ids = malloc(n_tok * 4);
    if (fread(ids, 4, n_tok, tf) != n_tok) return 1;
    fclose(tf);

    struct llama_model_params mp = llama_model_default_params();
    struct llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    struct llama_context_params cp = llama_context_default_params();
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    cp.embeddings = want_embd;
    cp.n_ctx = 256;
    cp.flash_attn_type = fa_on ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    struct llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx init failed\n"); return 1; }

    struct llama_batch batch = llama_batch_get_one(ids, (int32_t) n_tok);
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }

    FILE * out = fopen(out_path, "wb");
    if (!out) { perror("out fopen"); return 1; }
    // the logits rows are `n_vocab_out` wide (llama-context.cpp:2268); the
    // wavtokenizer head emits the waveform dim there, the others the vocab
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (!logits) { fprintf(stderr, "no logits row\n"); return 1; }
    uint32_t nv = (uint32_t) n_vocab;
    fwrite(&nv, 4, 1, out);
    fwrite(logits, 4, n_vocab, out);
    fprintf(stderr, "n_vocab=%d\n", n_vocab);
    if (want_embd) {
        const int32_t n_out = llama_n_embd(model);
        const float * embd = llama_get_embeddings_ith(ctx, -1);
        uint32_t no = (uint32_t) n_out;
        fwrite(&no, 4, 1, out);
        fwrite(embd, 4, n_out, out);
        fprintf(stderr, "dumped %d logits + %d embd\n", n_vocab, n_out);
    } else {
        fprintf(stderr, "dumped %d logits\n", n_vocab);
    }
    fclose(out);

    llama_free(ctx);
    llama_model_free(model);
    free(ids);
    return 0;
}
