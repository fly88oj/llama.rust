// sampler_dry_ref.cpp — bit-exactness probe for the DRY / adaptive-p /
// infill samplers and the common_sampler chain construction, driving the
// PINNED REFERENCE LIBRARY (bd4f514db1, /home/jeffrey/llm/llama.cpp/build-rust-ref)
// on the qwen2 vocab GGUF.
//
// Build + run: parity/gen_sampler_dry_ref.sh
// Output:      parity/sampler_dry_ref.txt (committed fixture; replayed by
//              crates/llama/tests/sampler_dry_parity.rs)
//
// Sections:
//   CHAIN <label> <s0>s1>...   — common_sampler_init chain names in order
//   DRY <case> ACCEPT a,b,c    — the accepted token stream (token ids)
//   DRY <case> APPLY id=bits … — post-apply logit bits (hex f32) per candidate
//   DRYCHAIN <case> STEP n tok — full chain (penalties->dry->top-k->temp->dist)
//   ADAPTIVE <case> STEP n sel — adaptive-p selected ids (+ EMA echo case)
//   INFILL <case> IDS a,b,c    — the candidate ids used
//   INFILL <case> APPLY id=p-bits … — post-apply survivors (prob bits)
#include "sampling.h"
#include "llama.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static uint32_t f2b(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

static std::vector<llama_token> toks(const llama_vocab * v, const char * s) {
    // add_special = false, parse_special = false (the DRY breaker path)
    const int32_t n = llama_tokenize(v, s, (int32_t) strlen(s), nullptr, 0, false, false);
    std::vector<llama_token> out(n < 0 ? -n : n);
    out.resize(llama_tokenize(v, s, (int32_t) strlen(s), out.data(), (int32_t) out.size(), false, false));
    return out;
}

static void print_apply(const char * label, llama_token_data_array * p) {
    printf("%s APPLY", label);
    for (size_t i = 0; i < p->size; ++i) {
        printf(" %d=%08x", p->data[i].id, f2b(p->data[i].logit));
    }
    printf("\n");
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <vocab-gguf>\n", argv[0]);
        return 1;
    }
    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        fprintf(stderr, "failed to load vocab '%s'\n", argv[1]);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // ------------------------------------------------------------------
    // 1) chain construction (common_sampler_init) — default + custom order
    // ------------------------------------------------------------------
    {
        common_params_sampling ps; // defaults
        common_sampler * g = common_sampler_init(model, ps);
        printf("CHAIN DEFAULT");
        for (int i = 0; i < llama_sampler_chain_n(common_sampler_get(g)); ++i) {
            printf(">%s", llama_sampler_name(llama_sampler_chain_get(common_sampler_get(g), i)));
        }
        printf("\n");
        common_sampler_free(g);
    }
    {
        common_params_sampling ps;
        ps.samplers = common_sampler_types_from_names({"dry", "top_k", "top_p", "temperature"});
        common_sampler * g = common_sampler_init(model, ps);
        printf("CHAIN NAMES");
        for (int i = 0; i < llama_sampler_chain_n(common_sampler_get(g)); ++i) {
            printf(">%s", llama_sampler_name(llama_sampler_chain_get(common_sampler_get(g), i)));
        }
        printf("\n");
        common_sampler_free(g);
    }
    {
        common_params_sampling ps;
        ps.samplers = common_sampler_types_from_chars("dkpt");
        common_sampler * g = common_sampler_init(model, ps);
        printf("CHAIN CHARS");
        for (int i = 0; i < llama_sampler_chain_n(common_sampler_get(g)); ++i) {
            printf(">%s", llama_sampler_name(llama_sampler_chain_get(common_sampler_get(g), i)));
        }
        printf("\n");
        common_sampler_free(g);
    }
    {
        // adaptive-p is always appended at the very end (replaces dist)
        common_params_sampling ps;
        ps.samplers = common_sampler_types_from_names({"top_k", "adaptive_p"});
        common_sampler * g = common_sampler_init(model, ps);
        printf("CHAIN ADAPTIVE");
        for (int i = 0; i < llama_sampler_chain_n(common_sampler_get(g)); ++i) {
            printf(">%s", llama_sampler_name(llama_sampler_chain_get(common_sampler_get(g), i)));
        }
        printf("\n");
        common_sampler_free(g);
    }
    {
        // dry disabled (multiplier 0) -> "?dry" empty sampler in the chain
        common_params_sampling ps;
        ps.dry_multiplier = 0.0f;
        ps.temp = 0.8f;
        common_sampler * g = common_sampler_init(model, ps);
        printf("CHAIN DRYOFF");
        for (int i = 0; i < llama_sampler_chain_n(common_sampler_get(g)); ++i) {
            printf(">%s", llama_sampler_name(llama_sampler_chain_get(common_sampler_get(g), i)));
        }
        printf("\n");
        common_sampler_free(g);
    }
    {
        // dry enabled -> "dry" sits at position 2 of the default chain
        common_params_sampling ps;
        ps.dry_multiplier = 0.8f;
        ps.temp = 0.8f;
        common_sampler * g = common_sampler_init(model, ps);
        printf("CHAIN DRYON");
        for (int i = 0; i < llama_sampler_chain_n(common_sampler_get(g)); ++i) {
            printf(">%s", llama_sampler_name(llama_sampler_chain_get(common_sampler_get(g), i)));
        }
        printf("\n");
        common_sampler_free(g);
    }

    // ------------------------------------------------------------------
    // 2) DRY apply fixtures (llama_sampler_init_dry on the real vocab)
    // ------------------------------------------------------------------
    // a repetitive stream: "Hello world," x6 — period-3 token repetition
    const std::vector<llama_token> rep = [&] {
        std::vector<llama_token> t;
        for (int i = 0; i < 6; ++i) {
            auto u = toks(vocab, "Hello world, ");
            t.insert(t.end(), u.begin(), u.end());
        }
        return t;
    }();
    printf("DRY IDS REP");
    for (auto t : rep) printf(",%d", t);
    printf("\n");

    // candidates: the repeated ids plus a few fixed ones, hand-set logits
    auto dry_case = [&](const char * label, float mult, float base, int allowed, int last_n,
                        const std::vector<std::string> & breakers,
                        const std::vector<llama_token> & accept_stream,
                        const std::vector<std::pair<llama_token, float>> & cands) {
        std::vector<const char *> cstr;
        cstr.reserve(breakers.size());
        for (auto & b : breakers) cstr.push_back(b.c_str());

        llama_sampler * s = llama_sampler_init_dry(vocab, mult, base, allowed, last_n,
                                                   cstr.data(), cstr.size());
        printf("%s ACCEPT", label);
        for (auto t : accept_stream) {
            printf(",%d", t);
            llama_sampler_accept(s, t);
        }
        printf("\n");

        std::vector<llama_token_data> data;
        for (auto & [id, l] : cands) data.push_back({id, l, 0.0f});
        llama_token_data_array p = { data.data(), data.size(), -1, false };
        llama_sampler_apply(s, &p);
        print_apply(label, &p);
        llama_sampler_free(s);
    };

    // candidate list shared by the DRY cases: every distinct repeated token
    // (repeats get penalized) + ids 5, 6, 7 (unrepeated fillers)
    std::vector<std::pair<llama_token, float>> cands;
    {
        std::vector<llama_token> uniq;
        for (auto t : rep) {
            bool seen = false;
            for (auto u : uniq) seen |= (u == t);
            if (!seen) uniq.push_back(t);
        }
        float lg = 1.5f;
        for (auto t : uniq) cands.push_back({t, lg -= 0.25f});
        cands.push_back({5, 0.75f});
        cands.push_back({6, 0.5f});
        cands.push_back({7, 0.25f});
    }

    // DRY1: defaults breakers, multiplier 0.8 base 1.75 allowed 2 last_n 64,
    //       accept 12 of the 18 stream tokens (a 4x period-3 repetition)
    dry_case("DRY1", 0.8f, 1.75f, 2, 64,
             {"\n", ":", "\"", "*"},
             std::vector<llama_token>(rep.begin(), rep.begin() + 12), cands);

    // DRY2: no breakers at all (empty list == none) — rep_limit stays full
    dry_case("DRY2", 0.8f, 1.75f, 2, 64, {},
             std::vector<llama_token>(rep.begin(), rep.begin() + 12), cands);

    // DRY3: allowed_length larger than any repeat -> early return
    dry_case("DRY3", 0.8f, 1.75f, 10, 64, {"\n"},
             std::vector<llama_token>(rep.begin(), rep.begin() + 12), cands);

    // DRY4: short window — only the last 4 tokens considered
    dry_case("DRY4", 0.8f, 1.75f, 2, 4, {"\n"},
             std::vector<llama_token>(rep.begin(), rep.begin() + 12), cands);

    // DRY5: a breaker token inside the window limits rep_limit. "\n" is a
    //       single-token breaker in qwen2 (piece "\n" == token 198); accept
    //       an interleaved stream with newlines
    {
        std::vector<llama_token> stream;
        auto nl = toks(vocab, "\n");
        for (int i = 0; i < 4; ++i) {
            auto u = toks(vocab, "Hello world, ");
            stream.insert(stream.end(), u.begin(), u.end());
            if (!nl.empty()) stream.push_back(nl[0]);
        }
        dry_case("DRY5", 0.8f, 1.75f, 2, 64, {"\n"}, stream, cands);
    }

    // DRY6: big base exercises the max_exponent clamp
    dry_case("DRY6", 1.5f, 3.0e6f, 1, 64, {"\n"},
             std::vector<llama_token>(rep.begin(), rep.begin() + 12), cands);

    // DRY7: a breaker token INSIDE the window cuts rep_limit to the distance
    //       (3 periods, then "\n", then one bare period at the end)
    {
        auto nl = toks(vocab, "\n");
        std::vector<llama_token> stream(rep.begin(), rep.begin() + 12);
        if (!nl.empty()) stream.push_back(nl[0]);
        auto tail = toks(vocab, "Hello world, ");
        stream.insert(stream.end(), tail.begin(), tail.end());
        dry_case("DRY7", 0.8f, 1.75f, 2, 64, {"\n"}, stream, cands);
    }

    // DRY8: a MULTI-token breaker ("Hello\n" = head "Hello" + tail "\n")
    //       matched via the processed-breaker tail comparison
    {
        std::vector<llama_token> stream(rep.begin(), rep.begin() + 12);
        auto brk = toks(vocab, "Hello\n"); // [1879, 198] in qwen2
        stream.insert(stream.end(), brk.begin(), brk.end());
        auto tail = toks(vocab, "Hello world, ");
        stream.insert(stream.end(), tail.begin(), tail.end());
        printf("DRY8 BREAKERS");
        for (auto t : brk) printf(",%d", t);
        printf("\n");
        dry_case("DRY8", 0.8f, 1.75f, 2, 64, {"Hello\n"}, stream, cands);
    }

    // DRY9: the server-parity scenario — prompt [the cat sat on the mat]*3 +
    //       "the cat sat on", then the 7 generated tokens; the next " mat"
    //       must receive the reference's exact penalty
    {
        auto stream = toks(vocab, "the cat sat on the mat the cat sat on the mat the cat sat on the mat the cat sat on");
        const std::vector<llama_token> gen = {264, 5517, 279, 8251, 7578, 389, 264};
        stream.insert(stream.end(), gen.begin(), gen.end());
        std::vector<std::pair<llama_token, float>> c2 = {
            {5517, 10.0f}, {61056, 2.0f}, {279, 1.0f}, {271, 0.5f}, {198, 0.3f}, {8251, 0.0f},
        };
        dry_case("DRY9", 0.8f, 1.75f, 2, 64, {"\n", ":", "\"", "*"}, stream, c2);
    }

    // ------------------------------------------------------------------
    // 3) full chain: penalties -> dry -> top_k -> temp-ext -> dist
    // ------------------------------------------------------------------
    auto chain_case = [&](const char * label, float temp, uint32_t seed, int n_steps,
                          const std::vector<llama_token> & prompt) {
        common_params_sampling ps;
        ps.temp = temp;
        ps.seed = seed;
        ps.penalty_last_n = 16;
        ps.penalty_repeat = 1.1f;
        ps.penalty_freq = 0.30f;
        ps.penalty_present = 0.25f;
        ps.dry_multiplier = 0.8f;
        ps.dry_base = 1.75f;
        ps.dry_allowed_length = 2;
        ps.dry_penalty_last_n = 64;
        ps.dry_sequence_breakers = {"\n", ":", "\"", "*"};
        ps.top_k = 40;
        ps.top_p = 0.95f;
        ps.min_p = 0.05f;
        ps.samplers = {
            COMMON_SAMPLER_TYPE_PENALTIES,
            COMMON_SAMPLER_TYPE_DRY,
            COMMON_SAMPLER_TYPE_TOP_K,
            COMMON_SAMPLER_TYPE_TOP_P,
            COMMON_SAMPLER_TYPE_MIN_P,
            COMMON_SAMPLER_TYPE_TEMPERATURE,
        };
        common_sampler * g = common_sampler_init(model, ps);
        for (auto t : prompt) common_sampler_accept(g, t, false);

        // fixed logits over the repeated-token slice: token t_i always has
        // logit 2.0 so the raw argmax is the first repeated token
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        std::vector<float> logits(n_vocab, -3.0f);
        for (auto t : rep) logits[t] = 2.0f;
        logits[5] = 1.9f;
        logits[6] = 1.8f;
        logits[7] = 1.7f;

        for (int step = 0; step < n_steps; ++step) {
            // drive the chain directly (identical to what the port's
            // SamplingContext::sample does with a raw logits slice)
            llama_sampler * chain = common_sampler_get(g);
            std::vector<llama_token_data> data(n_vocab);
            for (int32_t i = 0; i < n_vocab; ++i) data[i] = {i, logits[i], 0.0f};
            llama_token_data_array p = { data.data(), (size_t) n_vocab, -1, false };
            llama_sampler_apply(chain, &p);
            const llama_token tok = p.data[p.selected].id;
            llama_sampler_accept(chain, tok);
            printf("DRYCHAIN %s STEP %d tok %d\n", label, step, tok);
        }
        common_sampler_free(g);
    };

    // prompt = the 12-token repetition (so the first sample is already DRY'd)
    chain_case("GREEDY", 0.0f, 42, 8,
               std::vector<llama_token>(rep.begin(), rep.begin() + 12));
    chain_case("TEMP", 0.8f, 42, 12,
               std::vector<llama_token>(rep.begin(), rep.begin() + 12));

    // ------------------------------------------------------------------
    // 4) adaptive-p (bare sampler, like the chain's last member)
    // ------------------------------------------------------------------
    {
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        std::vector<float> logits(n_vocab, -3.0f);
        for (auto t : rep) logits[t] = 2.0f;
        logits[5] = 1.9f;
        logits[6] = 1.8f;
        logits[7] = 1.7f;

        for (auto label : {"ADAPTIVE1", "ADAPTIVE2"}) {
            const float target = (label[8] == '1') ? 0.30f : 0.05f;
            llama_sampler * s = llama_sampler_init_adaptive_p(target, 0.9f, 42);
            for (int step = 0; step < 10; ++step) {
                std::vector<llama_token_data> data(n_vocab);
                for (int32_t i = 0; i < n_vocab; ++i) data[i] = {i, logits[i], 0.0f};
                llama_token_data_array p = { data.data(), (size_t) n_vocab, -1, false };
                llama_sampler_apply(s, &p);
                const llama_token tok = p.data[p.selected].id;
                llama_sampler_accept(s, tok);
                printf("%s STEP %d sel %d\n", label, step, tok);
            }
            llama_sampler_free(s);
        }

        // negative target = disabled: plain dist sampling from the softmax
        {
            llama_sampler * s = llama_sampler_init_adaptive_p(-1.0f, 0.9f, 42);
            for (int step = 0; step < 4; ++step) {
                std::vector<llama_token_data> data(n_vocab);
                for (int32_t i = 0; i < n_vocab; ++i) data[i] = {i, logits[i], 0.0f};
                llama_token_data_array p = { data.data(), (size_t) n_vocab, -1, false };
                llama_sampler_apply(s, &p);
                const llama_token tok = p.data[p.selected].id;
                llama_sampler_accept(s, tok);
                printf("ADAPTIVE3 STEP %d sel %d\n", step, tok);
            }
            llama_sampler_free(s);
        }
    }

    // ------------------------------------------------------------------
    // 5) infill (real vocab: prefix merging + EOG gating)
    // ------------------------------------------------------------------
    {
        // candidate ids: pieces that form prefix chains in qwen2 plus fillers
        auto ids_of = [&](const char * s) { return toks(vocab, s); };
        std::vector<llama_token> pick;
        auto add = [&](const char * s) {
            auto v = ids_of(s);
            if (!v.empty()) pick.push_back(v.back());
        };
        add("h");      // single letter
        add("he");
        add("hel");
        add("hell");
        add("hello");
        add(" world");
        add("\n");
        add(":");

        printf("INFILL1 IDS");
        for (auto t : pick) printf(",%d", t);
        printf("\n");

        llama_sampler * s = llama_sampler_init_infill(vocab);
        // probs chosen so a prefix pair merges and some tokens fall under
        // both thresholds
        std::vector<float> probs = {0.30f, 0.15f, 0.12f, 0.10f, 0.08f, 0.13f, 0.06f, 0.06f};
        std::vector<llama_token_data> data;
        for (size_t i = 0; i < pick.size(); ++i) {
            // pre-softmax via log — apply() runs softmax itself
            data.push_back({pick[i], (float) log((double) probs[i]), 0.0f});
        }
        llama_token_data_array p = { data.data(), data.size(), -1, false };
        llama_sampler_apply(s, &p);
        printf("INFILL1 APPLY");
        for (size_t i = 0; i < p.size; ++i) {
            printf(" %d=%08x", p.data[i].id, f2b(p.data[i].p));
        }
        printf("\n");
        llama_sampler_free(s);

        // EOG-dominant case: all mass on the EOG candidates -> keep only EOG
        llama_sampler * s2 = llama_sampler_init_infill(vocab);
        std::vector<llama_token_data> data2;
        float pr2[8] = {0.001f, 0.001f, 0.001f, 0.001f, 0.001f, 0.001f, 0.6f, 0.394f};
        for (size_t i = 0; i < pick.size(); ++i) {
            data2.push_back({pick[i], (float) log((double) pr2[i]), 0.0f});
        }
        llama_token_data_array p2 = { data2.data(), data2.size(), -1, false };
        llama_sampler_apply(s2, &p2);
        printf("INFILL2 APPLY");
        for (size_t i = 0; i < p2.size; ++i) {
            printf(" %d=%08x", p2.data[i].id, f2b(p2.data[i].p));
        }
        printf("\n");
        llama_sampler_free(s2);
    }

    llama_model_free(model);
    return 0;
}
