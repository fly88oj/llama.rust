// ref_grammar_dump.cpp — reference dumps for the Rust GBNF port (agent Z).
//
// Links against the pinned llama.cpp build (bd4f514db1) and drives the *real*
// llama-grammar implementation through its exported internal entry points:
//
//   llama_grammar_parser            (parse → symbol_ids + rules)
//   llama_grammar_init_impl         (vocab + grammar string → grammar object)
//   llama_grammar_accept_token      (stack advance per token)
//   llama_grammar_apply_impl        (logit mask over all vocab candidates)
//
// Build (from the repo root):
//   g++ -O2 -std=c++17 -o parity/ref_grammar_dump parity/ref_grammar_dump.cpp \
//       -I/home/jeffrey/llm/llama.cpp-pinned/src \
//       -I/home/jeffrey/llm/llama.cpp-pinned/include \
//       -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
//       -L/home/jeffrey/llm/llama.cpp/build-rust-ref/bin -lllama -lggml-base \
//       -Wl,-rpath,/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
//
// Usage:
//   ref_grammar_dump --parser FILE.gbnf [FILE.gbnf ...]      → SYM/RULE lines
//   ref_grammar_dump --pieces --vocab V.gguf                 → PIECE lines
//   ref_grammar_dump --vocab V.gguf --grammar G.gbnf \
//                    [--root root] [--tokens "text"]         → VOCAB/TOKENS/STEP lines
//
// The text format is line-oriented and compared verbatim by
// crates/llama/tests/grammar_parity.rs.

#include "llama.h"
#include "llama-grammar.h"
#include "llama-vocab.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

static std::string read_file(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "failed to open '%s'\n", path.c_str());
        exit(1);
    }
    std::string out;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        out.append(buf, n);
    }
    fclose(f);
    return out;
}

static std::string basename_of(const std::string & path) {
    auto pos = path.find_last_of('/');
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

static std::string hex(const std::string & s) {
    static const char * d = "0123456789abcdef";
    std::string out;
    for (unsigned char c : s) {
        out += d[c >> 4];
        out += d[c & 15];
    }
    return out;
}

// map an element pointer back to (rule index, element index) — the Rust port
// represents stacks as these pairs
static std::pair<uint32_t, uint32_t> elem_ref(
        const llama_grammar_rules & rules, const llama_grammar_element * p) {
    for (size_t r = 0; r < rules.size(); r++) {
        for (size_t e = 0; e < rules[r].size(); e++) {
            if (&rules[r][e] == p) {
                return { (uint32_t) r, (uint32_t) e };
            }
        }
    }
    return { 0xFFFFFFFFu, 0xFFFFFFFFu };
}

static void print_stacks(const llama_grammar * g) {
    const auto & rules  = llama_grammar_get_rules(g);
    const auto & stacks = llama_grammar_get_stacks(const_cast<llama_grammar *>(g));

    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> canon;
    for (const auto & st : stacks) {
        std::vector<std::pair<uint32_t, uint32_t>> s;
        for (auto p : st) {
            s.push_back(elem_ref(rules, p));
        }
        std::sort(s.begin(), s.end());
        canon.push_back(s);
    }
    std::sort(canon.begin(), canon.end());

    printf(" STACKS %zu", canon.size());
    for (const auto & s : canon) {
        printf(" %zu", s.size());
        for (auto & e : s) {
            printf(" %u.%u", e.first, e.second);
        }
    }
}

static uint32_t fnv1a(const std::vector<llama_token> & ids) {
    uint32_t h = 2166136261u;
    for (auto id : ids) {
        uint32_t v = (uint32_t) id;
        for (int b = 0; b < 4; b++) {
            h ^= (v >> (8 * b)) & 0xFFu;
            h *= 16777619u;
        }
    }
    return h;
}

static void print_mask(llama_grammar * g, int32_t n_vocab) {
    std::vector<llama_token_data> data(n_vocab);
    for (int32_t i = 0; i < n_vocab; i++) {
        data[i] = { i, 1.0f, 0.0f };
    }
    llama_token_data_array cur = { data.data(), (size_t) n_vocab, -1, false };
    llama_grammar_apply_impl(*g, &cur);

    std::vector<llama_token> keep;
    for (int32_t i = 0; i < n_vocab; i++) {
        if (cur.data[i].logit != -INFINITY) {
            keep.push_back(i);
        }
    }
    // compact: count + FNV-1a over the surviving ids + the first up-to-12 ids
    printf(" MASK %zu %08x", keep.size(), fnv1a(keep));
    for (size_t i = 0; i < keep.size() && i < 12; i++) {
        printf(" %d", keep[i]);
    }
}

static const llama_vocab * load_vocab(const char * path, llama_model ** out_model) {
    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(path, mparams);
    if (!model) {
        fprintf(stderr, "failed to load vocab '%s'\n", path);
        exit(1);
    }
    *out_model = model;
    return llama_model_get_vocab(model);
}

int main(int argc, char ** argv) {
    std::vector<std::string> parser_files;
    std::string vocab_path, grammar_path, root = "root", tokens_text, token_ids, pieces_bin;
    bool pieces = false, have_tokens = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--parser" && i + 1 < argc) {
            parser_files.push_back(argv[++i]);
        } else if (a == "--vocab" && i + 1 < argc) {
            vocab_path = argv[++i];
        } else if (a == "--grammar" && i + 1 < argc) {
            grammar_path = argv[++i];
        } else if (a == "--root" && i + 1 < argc) {
            root = argv[++i];
        } else if (a == "--tokens" && i + 1 < argc) {
            tokens_text = argv[++i];
            have_tokens = true;
        } else if (a == "--token-ids" && i + 1 < argc) {
            token_ids = argv[++i];
            have_tokens = true;
        } else if (a == "--pieces") {
            pieces = true;
        } else if (a == "--pieces-bin" && i + 1 < argc) {
            pieces_bin = argv[++i];
        } else if (a.rfind("--", 0) != 0) {
            // bare argument: another grammar file for --parser
            parser_files.push_back(a);
        } else {
            fprintf(stderr, "unknown arg '%s'\n", a.c_str());
            return 1;
        }
    }

    llama_backend_init();

    // ---- parser dump (no vocab needed; token rules must use <[id]> form) ----
    for (const auto & f : parser_files) {
        std::string bytes = read_file(f);
        llama_grammar_parser parser;
        const bool ok = parser.parse(bytes.c_str());
        printf("PARSE %s %s\n", basename_of(f).c_str(), ok ? "ok" : "FAIL");
        printf("GSRC %s\n", hex(bytes).c_str());
        if (!ok) {
            continue;
        }
        for (const auto & kv : parser.symbol_ids) {
            printf("SYM %s %u\n", kv.first.c_str(), kv.second);
        }
        for (size_t r = 0; r < parser.rules.size(); r++) {
            printf("RULE %zu", r);
            for (const auto & e : parser.rules[r]) {
                printf(" %d %u", (int) e.type, e.value);
            }
            printf("\n");
        }
    }

    if (vocab_path.empty()) {
        llama_backend_free();
        return 0;
    }

    llama_model * model = nullptr;
    const llama_vocab * vocab = load_vocab(vocab_path.c_str(), &model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    printf("VOCAB %d\n", n_vocab);

    if (pieces) {
        for (int32_t id = 0; id < n_vocab; id++) {
            const std::string & p = vocab->token_to_piece(id);
            printf("PIECE %d %s %d\n", id, hex(p).c_str(), (int) vocab->is_eog(id));
        }
    }

    // binary dump: [u32 n][ (u32 len, bytes, u8 is_eog) * n ]  (little-endian)
    if (!pieces_bin.empty()) {
        FILE * f = fopen(pieces_bin.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "cannot write '%s'\n", pieces_bin.c_str());
            return 1;
        }
        const uint32_t n = (uint32_t) n_vocab;
        fwrite(&n, 4, 1, f);
        for (int32_t id = 0; id < n_vocab; id++) {
            const std::string & p = vocab->token_to_piece(id);
            const uint32_t len = (uint32_t) p.size();
            const uint8_t eog = vocab->is_eog(id) ? 1 : 0;
            fwrite(&len, 4, 1, f);
            fwrite(p.data(), 1, len, f);
            fwrite(&eog, 1, 1, f);
        }
        fclose(f);
        fprintf(stderr, "wrote %s (%d tokens)\n", pieces_bin.c_str(), n_vocab);
    }

    if (!grammar_path.empty()) {
        std::string gstr = read_file(grammar_path);
        llama_grammar * g = llama_grammar_init_impl(
                vocab, gstr.c_str(), root.c_str(), /* lazy */ false, nullptr, 0, nullptr, 0);

        // token list (reference ids) — printed first so the block is keyed by CASE
        std::vector<llama_token> toks;
        if (have_tokens) {
            if (!token_ids.empty()) {
                size_t pos = 0;
                while (pos <= token_ids.size()) {
                    size_t comma = token_ids.find(',', pos);
                    std::string part = token_ids.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                    if (!part.empty()) {
                        toks.push_back(std::stoi(part));
                    }
                    if (comma == std::string::npos) {
                        break;
                    }
                    pos = comma + 1;
                }
            } else {
                // tokenize with the reference vocab (add_special=false, parse_special=true)
                toks.resize(n_vocab);
                int32_t n = llama_tokenize(vocab, tokens_text.c_str(), (int32_t) tokens_text.size(),
                                           toks.data(), (int32_t) toks.size(), false, true);
                toks.resize(n);
            }
            printf("CASE %s", basename_of(grammar_path).c_str());
            for (auto t : toks) {
                printf(" %d", t);
            }
            printf("\n");
        }

        printf("GSRC %s\n", hex(gstr).c_str());
        printf("GRAMMAR %s %s\n", basename_of(grammar_path).c_str(), g ? "ok" : "FAIL");
        if (!g) {
            llama_backend_free();
            return 1;
        }

        if (have_tokens) {
            printf("TOKENS");
            for (auto t : toks) {
                printf(" %d", t);
            }
            printf("\n");

            for (size_t i = 0; i < toks.size(); i++) {
                const std::string & piece = vocab->token_to_piece(toks[i]);
                printf("STEP %zu %d %s", i, toks[i], hex(piece).c_str());
                try {
                    llama_grammar_accept_token(*g, toks[i], piece);
                } catch (const std::exception & e) {
                    printf(" ACCEPT_FAILED\n");
                    break;
                }
                print_stacks(g);
                print_mask(g, n_vocab);
                printf("\n");
            }
        } else {
            print_stacks(g);
            print_mask(g, n_vocab);
            printf("\n");
        }
        llama_grammar_free_impl(g);
    }

    llama_model_free(model);
    llama_backend_free();
    return 0;
}