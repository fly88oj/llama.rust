/* DFlash drafted-chain probe: runs examples/speculative-simple's exact driver
 * loop (speculative-simple.cpp:80-342 @ bd4f514db1) over a target model + a
 * dflash/dspark draft, with the SPC_DBG candidate traces enabled — the oracle
 * for the port's draft chain (crates/llama/src/speculative.rs's draft-dflash
 * impl driven by llama-cli, LLAMA_SPEC_VERBOSE=1).
 *
 * Build + run:
 *
 *   g++ -O2 -std=c++17 parity/ref_dflash_chain.cpp -o /tmp/ref_dflash_chain \
 *     -I /home/jeffrey/llm/llama.cpp-pinned/include \
 *     -I /home/jeffrey/llm/llama.cpp-pinned/ggml/include \
 *     -I /home/jeffrey/llm/llama.cpp-pinned/common \
 *     -L /home/jeffrey/llm/llama.cpp/build-rust-ref/bin -lllama -lcommon \
 *     -Wl,-rpath,/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
 *   LLAMA_LOG_LEVEL=DEBUG /tmp/ref_dflash_chain <tgt.gguf> <draft.gguf> \
 *       <draft-dflash|draft-dspark> <n_predict>
 */
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"
// internal access for the DFlash2 lattice dump: the nextn tap has no public
// C API (llama-context.cpp:1004-1008)
#include "llama-context.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/* --dump-first: after the prompt injection, dump the fused encoder output
 * (llama_get_embeddings of the draft context — the injection graph's
 * res->t_embd, dflash.cpp:673) and then run the FIRST noise block manually
 * (speculative.cpp:1187-1221) and dump its logits rows — the port-side
 * differential oracle for the injection + noise graphs. */
static bool g_dump_first = false;

static void dump_f32(const char * tag, const float * p, int n) {
    printf("%s:", tag);
    for (int i = 0; i < n; i++) {
        printf(" %.6e", p[i]);
    }
    printf("\n");
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <tgt.gguf> <draft.gguf> <draft-dflash|draft-dspark> <n_predict> [--dump-first]\n", argv[0]);
        return 1;
    }
    for (int i = 5; i < argc; i++) {
        if (strcmp(argv[i], "--dump-first") == 0) {
            g_dump_first = true;
        }
    }

    common_params params;
    params.model.path         = argv[1];
    params.n_ctx              = 512;
    params.n_batch            = 2048;
    params.n_ubatch           = 512;
    params.n_predict          = atoi(argv[4]);
    params.no_perf            = true;
    /* CHAIN_FA=on selects flash attention (the fa=on chain/dump cells); the
     * default stays DISABLED — the original cells' configuration */
    params.flash_attn_type    = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (getenv("CHAIN_FA") != NULL && strcmp(getenv("CHAIN_FA"), "on") == 0) {
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
    params.n_parallel         = 1;
    params.cpuparams.n_threads       = 8;
    params.cpuparams_batch.n_threads = 8;

    params.speculative.draft.mparams.path = argv[2];
    params.speculative.draft.n_max      = 3;
    params.speculative.draft.n_min      = 0;
    params.speculative.draft.p_min      = 0.0f;
    {
        std::vector<std::string> names;
        names.push_back(argv[3]);
        params.speculative.types = common_speculative_types_from_names(names);
    }

    // the committed stream is what matters — temperature 0
    params.sampling.temp = 0.0f;

    const auto output_limits = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
    params.n_outputs_max = output_limits.total;
    params.n_outputs_max_per_seq = output_limits.per_seq;

    common_init();
    // SPC_DBG's candidate traces flow at DEBUG — the same threshold the
    // server's -v sets (arg.cpp:3920-3926: verbosity = INT_MAX)
    common_log_set_verbosity_thold(INT_MAX);

    llama_backend_init();

    // the target context (common_init_from_params)
    llama_model * model_tgt = NULL;
    llama_context * ctx_tgt = NULL;
    {
        common_init_result_ptr res = common_init_from_params(params);
        if (!res || !res->model() || !res->context()) {
            fprintf(stderr, "target init failed\n");
            return 1;
        }
        model_tgt = res->model();
        ctx_tgt   = res->context();
        // keep both alive: release the wrapper without freeing
        (void) res.release();
    }

    // the draft context (common_speculative_init_from_params)
    common_speculative_init_result_ptr spec_init;
    {
        common_params params_dft = common_base_params_to_speculative(params);
        spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = spec_init->context();
    }

    // the same prompt the port tokenizes
    const std::string prompt = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12";
    std::vector<llama_token> inp = common_tokenize(ctx_tgt, prompt, true, true);

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    common_sampler_ptr smpl(common_sampler_init(model_tgt, params.sampling));

    common_speculative * spec = common_speculative_init(params.speculative, 1);
    if (spec == nullptr) {
        fprintf(stderr, "failed to initialize speculative decoding\n");
        return 1;
    }

    // ---- the driver loop (speculative-simple.cpp:126-342) ----
    const llama_seq_id seq_id = 0;

    {
        llama_batch batch_prompt = llama_batch_init(inp.size(), 0, 1);
        for (size_t i = 0; i < inp.size() - 1; ++i) {
            common_batch_add(batch_prompt, inp[i], i, { seq_id }, false);
        }

        if (llama_decode(ctx_tgt, batch_prompt) != 0) {
            fprintf(stderr, "failed to decode prompt\n");
            return 1;
        }

        if (!common_speculative_process(spec, batch_prompt)) {
            fprintf(stderr, "failed to process speculative prompt\n");
            return 1;
        }
        llama_batch_free(batch_prompt);
    }

    if (g_dump_first) {
        // the injected draft-cache state (the raw K/V rows of the prompt
        // prefix) — byte-comparable against the port's plain-cache
        // state_seq_get_data (kv_cache.rs)
        {
            const size_t sz = llama_state_seq_get_size(params.speculative.draft.ctx_dft, 0);
            uint8_t * buf = (uint8_t *) malloc(sz);
            const size_t got = llama_state_seq_get_data(params.speculative.draft.ctx_dft, buf, sz, 0);
            FILE * f = fopen(getenv("DUMP_STATE_OUT") ? getenv("DUMP_STATE_OUT") : "/tmp/ref-dft-state.bin", "wb");
            if (f) {
                uint32_t len = (uint32_t) got;
                fwrite(&len, sizeof(uint32_t), 1, f);
                fwrite(buf, 1, got, f);
                fclose(f);
                fprintf(stderr, "dump: draft state %zu bytes\n", got);
            }
            free(buf);
        }

        // the injection's fused output rides res->t_embd, which the draft
        // context does not extract (`cparams.embeddings = false`,
        // common_base_params_to_speculative) — the comparable oracle is the
        // first noise block's logits (the injection's only consumer)

        // the first noise block (speculative.cpp:1187-1221, n_max 3 → 4 rows)
        // — at the driver's n_past = inp.size() - 1 (the prompt tokenizes to
        // more tokens than the comma string suggests)
        const int n_past0 = (int) inp.size() - 1;
        fprintf(stderr, "dump: n_past %d draft pos_max %d\n",
                n_past0,
                (int) llama_memory_seq_pos_max(llama_get_memory(params.speculative.draft.ctx_dft), seq_id));
        const llama_token mask_id = llama_vocab_mask(llama_model_get_vocab(llama_get_model(params.speculative.draft.ctx_dft)));
        llama_batch batch_dft = llama_batch_init(4, 0, 1);
        common_batch_add(batch_dft, inp[inp.size() - 1], n_past0 + 0, { seq_id }, true);
        common_batch_add(batch_dft, mask_id, n_past0 + 1, { seq_id }, true);
        common_batch_add(batch_dft, mask_id, n_past0 + 2, { seq_id }, true);
        common_batch_add(batch_dft, mask_id, n_past0 + 3, { seq_id }, true);
        if (llama_decode(params.speculative.draft.ctx_dft, batch_dft) != 0) {
            fprintf(stderr, "noise decode failed\n");
            return 1;
        }
        const int n_vocab = llama_vocab_n_tokens(vocab);
        for (int r = 0; r < 4; r++) {
            const float * logits = llama_get_logits_ith(params.speculative.draft.ctx_dft, r);
            if (!logits) {
                fprintf(stderr, "no logits row %d\n", r);
                return 1;
            }
            char tag[32];
            snprintf(tag, sizeof(tag), "noise_logits_%d", r);
            dump_f32(tag, logits, n_vocab);
        }
        /* DFlash2: the selector lattice rides the (unmasked) nextn tap —
         * res->t_h_nextn = the packed [top_k | top_k*top_k] rows of the block
         * (dflash.cpp:545-566). DSpark's conf rows are not comparable here. */
        {
            char top_k_str[64] = {0};
            const llama_model * model_dft = llama_get_model(params.speculative.draft.ctx_dft);
            int has_selector = 0;
            if (llama_model_meta_val_str(model_dft, "dflash.selector_top_k", top_k_str, sizeof(top_k_str)) >= 0) {
                has_selector = atoi(top_k_str) > 0;
            }
            const float * lattice = has_selector ? params.speculative.draft.ctx_dft->get_embeddings_nextn() : NULL;
            const int n_embd = llama_model_n_embd(model_dft);
            if (lattice != NULL) {
                for (int r = 0; r < 4; r++) {
                    char tag[32];
                    snprintf(tag, sizeof(tag), "lattice_%d", r);
                    dump_f32(tag, lattice + (size_t) r * n_embd, n_embd);
                }
            }
        }
        llama_batch_free(batch_dft);
        printf("dumped\n");
        return 0;
    }

    llama_token id_last = inp[inp.size() - 1];

    std::vector<llama_token> prompt_tgt(inp.begin(), inp.end() - 1);

    int n_past = inp.size() - 1;

    common_speculative_begin(spec, seq_id, prompt_tgt);

    llama_batch batch_tgt = llama_batch_init(inp.size(), 0, 1);

    std::vector<llama_token> draft;

    int n_predict_done = 0;
    int n_drafted = 0;
    int n_accept = 0;
    bool has_eos = false;
    std::vector<llama_token> committed;

    while (true) {
        if (draft.empty()) {
            int32_t n_draft_max = 512 - n_past - 2;
            n_draft_max = std::min(n_draft_max, params.n_predict - n_predict_done - 1);
            n_draft_max = std::max(n_draft_max, 0);

            common_speculative_draft_params & dp = common_speculative_get_draft_params(spec, seq_id);
            dp.drafting = true;
            dp.n_max    = n_draft_max;
            dp.pos0     = n_past;
            dp.id_last  = id_last;
            dp.prompt   = &prompt_tgt;
            dp.result   = &draft;

            common_speculative_draft(spec);

            // the checkpoint-free path: drop the draft tail past n_past
            // (speculative-simple.cpp:206-213)
            llama_memory_seq_rm(llama_get_memory(params.speculative.draft.ctx_dft), seq_id, n_past, -1);
        }

        common_batch_clear(batch_tgt);
        common_batch_add(batch_tgt, id_last, n_past, { seq_id }, true);
        n_past++;

        for (size_t i = 0; i < draft.size(); ++i) {
            common_batch_add(batch_tgt, draft[i], n_past + i, { seq_id }, true);
        }

        if (llama_decode(ctx_tgt, batch_tgt) != 0) {
            fprintf(stderr, "failed to decode target batch\n");
            return 1;
        }

        if (!common_speculative_process(spec, batch_tgt)) {
            fprintf(stderr, "failed to process speculative batch\n");
            return 1;
        }

        const size_t n_draft = draft.size();

        // the DFlash2 selector path prints no SPC_DBG candidates; the drafted
        // ids of every round are the walk output — mirror of the port's
        // "round draft:" spec trace (LLAMA_SPEC_VERBOSE=1)
        if (!draft.empty()) {
            printf("round_draft:");
            for (auto id : draft) {
                printf(" %d", id);
            }
            printf("\n");
        }

        // sample from the full target batch and accept
        std::vector<llama_token> ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft, false);

        common_speculative_accept(spec, seq_id, ids.size() - 1);

        n_past += ids.size() - 1;
        n_drafted += n_draft;
        n_accept += ids.size() - 1;
        n_predict_done += ids.size();

        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);
            id_last = ids[i];

            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            }

            committed.push_back(id_last);
        }

        draft.clear();

        llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, n_past, -1);
        llama_memory_seq_rm(llama_get_memory(params.speculative.draft.ctx_dft), seq_id, n_past, -1);

        if ((params.n_predict >= 0 && n_predict_done > params.n_predict) || has_eos) {
            break;
        }
    }

    fprintf(stderr, "\n");
    printf("committed:");
    for (auto id : committed) {
        printf(" %d", id);
    }
    printf("\n");
    printf("spec: n_drafted = %d\n", n_drafted);
    printf("spec: n_accept  = %d\n", n_accept);
    printf("spec: n_predict = %d\n", n_predict_done);

    return 0;
}
