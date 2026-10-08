/* Encoder (llama_encode) ground truth from the reference build.
 *
 * Produces the dumps consumed by crates/llama/tests/{bert,t5}_e2e.rs. Uses the
 * public embedding API only — `llama_model_load_from_file` +
 * `llama_context_default_params()` with `embeddings = true`, then
 * `llama_encode()` (llama-context.cpp:1466) and `llama_get_embeddings()`
 * (llama-context.cpp:964), which returns `res->get_embd_pooled()` when the
 * graph built a pooling tensor and `res->get_embd()` otherwise
 * (llama-context.cpp:1546).
 *
 * The token ids are produced by the reference tokenizer in the same run and are
 * part of the artifact, so the Rust comparison feeds the ids straight into the
 * port's encoder and never depends on tokenizer parity.
 *
 * Build (from the repo root) — see parity/gen_encode_ref.sh:
 *   PINNED=/home/jeffrey/llm/llama.cpp-pinned
 *   REF=/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
 *   g++ -O2 -std=c++17 parity/ref_encode_dump.cpp -I$PINNED/include \
 *       -I$PINNED/ggml/include -L$REF -lllama -Wl,-rpath,$REF -o parity/ref_encode_dump
 *
 * Run: ./parity/ref_encode_dump <model.gguf> <out.bin> [--pool default|none]
 *        [--text <prompt>] [--threads N] [--fa on|off]
 *
 * File format (little-endian):
 *   char   magic[8]   = "LENCE1\0\0"
 *   u32    n_tokens          // batch size handed to llama_encode
 *   u32    n_embd_out        // hparams.n_embd_out() — row length in `embd`
 *   u32    n_rows            // 1 for a pooled dump, n_tokens for --pool none
 *   u32    pooling_type      // llama_pooling_type actually used (0 = NONE)
 *   i32    tokens[n_tokens]
 *   f32    embd[n_embd_out * n_rows]   // llama_get_embeddings() contents
 *
 * A `.txt` sidecar with the prompt, the ids and the reference log line is
 * written next to <out.bin> for human inspection (not consumed by tests).
 *
 * flash_attn: the reference default is LLAMA_FLASH_ATTN_TYPE_AUTO, which on
 * this CPU build resolves to enabled (llama-context.cpp:556) — BERT then takes
 * the ggml_flash_attn_ext branch of build_attn_mha (no kq_b). The port's
 * encoder anchors to the non-FA branch (its `use_flash_attn == false`
 * baseline, same choice as the qwen2 decode path), so the dump uses --fa off
 * (LLAMA_FLASH_ATTN_TYPE_DISABLED) by default. --fa on exists to measure the
 * FA/non-FA gap on the reference side itself.
 */
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "llama.h"

static void die(const char * msg) {
    fprintf(stderr, "ref_encode_dump: %s\n", msg);
    exit(1);
}

/// Split a whitespace/comma separated token id list (`--ids`).
static std::vector<llama_token> parse_ids(const char * s) {
    std::vector<llama_token> out;
    std::string cur;
    for (const char * p = s;; ++p) {
        if (*p == ',' || *p == ' ' || *p == '\t' || *p == '\0') {
            if (!cur.empty()) {
                out.push_back((llama_token) atoi(cur.c_str()));
                cur.clear();
            }
            if (*p == '\0') {
                break;
            }
        } else {
            cur.push_back(*p);
        }
    }
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        die("usage: ref_encode_dump <model.gguf> <out.bin> [--pool default|none|mean|cls|last|rank] "
            "[--text <prompt> | --ids <i,i,...>] [--threads N] [--fa on|off]");
    }

    const char * model_path = argv[1];
    const char * out_path   = argv[2];
    std::string prompt      = "The quick brown fox jumps over the lazy dog";
    std::vector<llama_token> forced_ids;   // --ids: skip the tokenizer entirely
    enum llama_pooling_type pooling = LLAMA_POOLING_TYPE_UNSPECIFIED; // "default" = the file's
    bool        flat        = false;   // force the flat llama_get_embeddings() buffer
    int         n_threads   = 8;
    bool        fa_on       = false;

    for (int i = 3; i < argc; ++i) {
        if (!strcmp(argv[i], "--pool") && i + 1 < argc) {
            const char * v = argv[++i];
            if (!strcmp(v, "default")) {
                pooling = LLAMA_POOLING_TYPE_UNSPECIFIED;
            } else if (!strcmp(v, "none")) {
                pooling = LLAMA_POOLING_TYPE_NONE;
            } else if (!strcmp(v, "mean")) {
                pooling = LLAMA_POOLING_TYPE_MEAN;
            } else if (!strcmp(v, "cls")) {
                pooling = LLAMA_POOLING_TYPE_CLS;
            } else if (!strcmp(v, "last")) {
                pooling = LLAMA_POOLING_TYPE_LAST;
            } else if (!strcmp(v, "rank")) {
                pooling = LLAMA_POOLING_TYPE_RANK;
            } else {
                die("--pool takes default|none|mean|cls|last|rank");
            }
        } else if (!strcmp(argv[i], "--flat")) {
            flat = true;
        } else if (!strcmp(argv[i], "--ids") && i + 1 < argc) {
            forced_ids = parse_ids(argv[++i]);
        } else if (!strcmp(argv[i], "--text") && i + 1 < argc) {
            prompt = argv[++i];
        } else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--fa") && i + 1 < argc) {
            const char * v = argv[++i];
            fa_on = !strcmp(v, "on");
        } else {
            die("unknown argument");
        }
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        die("failed to load model");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // tokenize with the model's own defaults (add_special = the file's
    // add_bos/add_eos flags — llama-vocab.cpp:llama_tokenize); --ids bypasses
    // the tokenizer so a caller can pin single tokens for bisection
    std::vector<llama_token> tokens;
    if (!forced_ids.empty()) {
        tokens = forced_ids;
    } else {
        tokens.resize(prompt.size() + 8);
        int n_tok = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
                                   tokens.data(), (int32_t) tokens.size(), true, true);
        if (n_tok < 0) {
            tokens.resize(-n_tok);
            n_tok = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
                                   tokens.data(), (int32_t) tokens.size(), true, true);
        }
        if (n_tok <= 0) {
            die("tokenization produced no tokens");
        }
        tokens.resize(n_tok);
    }
    const int n_tok = (int) tokens.size();

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx        = 0;                     // 0 => hparams.n_ctx_train
    cparams.n_batch      = 2048;
    cparams.n_ubatch     = 2048;                  // encode asserts n_ubatch >= n_tokens
    cparams.n_seq_max    = 1;
    cparams.n_threads    = n_threads;
    cparams.embeddings   = true;
    cparams.no_perf      = true;
    cparams.pooling_type = pooling;
    cparams.flash_attn_type = fa_on ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        die("failed to create context");
    }

    if (llama_n_ubatch(ctx) < (uint32_t) n_tok) {
        die("n_ubatch < n_tokens");
    }

    llama_batch batch = llama_batch_init(n_tok, 0, 1);
    for (int i = 0; i < n_tok; ++i) {
        batch.token[i]    = tokens[i];
        batch.pos[i]      = (llama_pos) i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = 1;
    }
    batch.n_tokens = n_tok;

    if (llama_encode(ctx, batch) != 0) {
        die("llama_encode failed");
    }

    const enum llama_pooling_type ptype = llama_pooling_type(ctx);

    // MEAN/CLS/LAST/RANK land in the per-sequence map (`embd_seq`,
    // llama-context.cpp:1573-1598 "extract sequence embeddings"), only NONE
    // fills the flat `embd` buffer (llama-context.cpp:1565). --flat forces the
    // flat buffer even when the file's pooling would collapse it, which is how
    // the "is t_embd already pooled?" question is answered from the reference.
    const bool flat_buf = flat || ptype == LLAMA_POOLING_TYPE_NONE;
    const float * embd;
    if (flat_buf) {
        embd = llama_get_embeddings(ctx);
        if (!embd) {
            die("llama_get_embeddings returned NULL");
        }
    } else {
        embd = llama_get_embeddings_seq(ctx, 0);
        if (!embd) {
            die("llama_get_embeddings_seq returned NULL");
        }
    }

    const int n_rows = flat_buf ? n_tok : 1;

    // hparams.n_embd_out() — the row length llama_context::output_reserve sizes
    // the embedding buffer with (llama-context.cpp:2143, llama-hparams.h:426).
    // RANK reads n_cls_out floats per sequence instead (llama-context.cpp:1583-
    // 1596 "extract the rerank score"), so the header carries that width for a
    // rank dump (llama_model_n_cls_out, llama.h:606).
    const int n_embd_out = ptype == LLAMA_POOLING_TYPE_RANK
        ? (int) llama_model_n_cls_out(model)
        : llama_model_n_embd_out(model);

    FILE * f = fopen(out_path, "wb");
    if (!f) {
        die("cannot open output file");
    }
    uint32_t hdr[4] = { (uint32_t) n_tok, (uint32_t) n_embd_out, (uint32_t) n_rows, (uint32_t) ptype };
    fwrite("LENCE1\0\0", 1, 8, f);
    fwrite(hdr, sizeof(hdr), 1, f);
    fwrite(tokens.data(), sizeof(llama_token), tokens.size(), f);
    fwrite(embd, sizeof(float), (size_t) n_embd_out * n_rows, f);
    fclose(f);

    // human-readable sidecar
    std::string txt_path = std::string(out_path) + ".txt";
    FILE * t = fopen(txt_path.c_str(), "w");
    if (t) {
        fprintf(t, "model     %s\n", model_path);
        fprintf(t, "prompt    %s\n", prompt.c_str());
        fprintf(t, "n_tokens  %d  n_embd_out %d  n_rows %d  pooling %d\n",
                n_tok, n_embd_out, n_rows, (int) ptype);
        fprintf(t, "tokens   ");
        for (int i = 0; i < n_tok; ++i) {
            fprintf(t, " %d", tokens[i]);
        }
        fprintf(t, "\n");
        double sum = 0.0, sq = 0.0, mn = 1e30, mx = -1e30;
        const size_t n = (size_t) n_embd_out * n_rows;
        for (size_t i = 0; i < n; ++i) {
            const double v = embd[i];
            sum += v;
            sq  += v * v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        fprintf(t, "stats     n=%zu mean=%.9g rms=%.9g min=%.9g max=%.9g\n",
                n, sum / n, sqrt(sq / n), mn, mx);
        for (int r = 0; r < n_rows && r < 4; ++r) {
            fprintf(t, "row[%d]   ", r);
            for (int i = 0; i < n_embd_out && i < 8; ++i) {
                fprintf(t, " %.7g", embd[(size_t) r * n_embd_out + i]);
            }
            fprintf(t, "\n");
        }
        fclose(t);
    }

    fprintf(stderr, "ref_encode_dump: %s -> %s (%d tokens, %d rows x %d, pooling %d)\n",
            model_path, out_path, n_tok, n_rows, n_embd_out, (int) ptype);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}