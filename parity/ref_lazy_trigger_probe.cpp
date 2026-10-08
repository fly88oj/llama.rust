// ref_lazy_trigger_probe.cpp — reference dumps for the lazy-grammar PATTERN
// triggers (agent: lazy PATTERN trigger engine).
//
// Links against the pinned llama.cpp build (bd4f514db1) and drives the *real*
// trigger machinery of src/llama-grammar.cpp through the exported internals:
//
//   llama_grammar_trigger_pattern::find   (the fire-position regex path,
//                                          llama-grammar.cpp:378-409)
//   llama_grammar_init_impl               (lazy grammar + trigger patterns,
//                                          llama-grammar.cpp:1216-1315)
//   llama_grammar_accept_impl             (the awaiting-trigger accept loop,
//                                          llama-grammar.cpp:1398-1455)
//
// Build (from the repo root, like parity/gen_grammar_ref.sh):
//   g++ -O2 -std=c++17 -o parity/ref_lazy_trigger_probe \
//       parity/ref_lazy_trigger_probe.cpp \
//       -I/home/jeffrey/llm/llama.cpp-pinned/src \
//       -I/home/jeffrey/llm/llama.cpp-pinned/include \
//       -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
//       -L/home/jeffrey/llm/llama.cpp/build-rust-ref/bin -lllama -lggml-base \
//       -Wl,-rpath,/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
//
// Modes (output compared by engine::lazy_trigger_tests in llama-server):
//
//   --find CASES
//       CASES is a text file of `<pattern_hex> <buffer_hex>` lines. For each,
//       the reference `std::regex` is built exactly like
//       llama_grammar_init_impl (:1293-1298) and `find` runs:
//         FIND <pattern_hex> <buffer_hex> <pos|npos>
//
//   --accept --vocab V.gguf --grammar F.gbnf [--pattern HEX]... \
//            [--text "…"] [--case NAME]
//       Initializes the lazy grammar with the trigger patterns, tokenizes the
//       text (add_special=false, parse_special=true) and accepts token by
//       token through llama_grammar_accept_impl, dumping the awaiting flag,
//       the trigger buffer and the stack count after every step:
//         CASE <name>
//         GRAMMAR <hex>
//         PATTERN <hex>            (one per trigger, in order)
//         TOK <dense_id> <orig_id> <piece_hex>
//         STEP <i> <awaiting> <buffer_len> <n_stacks>
//       Dense ids renumber the used tokens 0..n-1 so the Rust replay can use
//       a synthetic piece table.

#include "llama.h"
#include "llama-grammar.h"
#include "llama-vocab.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::string unhex(const std::string & s) {
    std::string out;
    for (size_t i = 0; i + 1 < s.size(); i += 2) {
        out.push_back((char) strtol(s.substr(i, 2).c_str(), nullptr, 16));
    }
    return out;
}

static std::string hex(const std::string & s) {
    static const char * d = "0123456789abcdef";
    std::string out;
    for (unsigned char c : s) {
        out.push_back(d[c >> 4]);
        out.push_back(d[c & 15]);
    }
    return out;
}

static std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "failed to open '%s'\n", path.c_str());
        exit(1);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static int run_find(const std::string & cases_path) {
    std::ifstream f(cases_path);
    if (!f) {
        fprintf(stderr, "failed to open '%s'\n", cases_path.c_str());
        return 1;
    }
    std::string line;
    while (std::getline(f, line)) {
        // line-based: `<pattern_hex> <buffer_hex>` (buffer may be empty hex)
        const size_t sp = line.find(' ');
        if (sp == std::string::npos) {
            continue;
        }
        const std::string phex = line.substr(0, sp);
        const std::string bhex = line.substr(sp + 1);
        const std::string pattern = unhex(phex);
        const std::string input   = unhex(bhex);
        try {
            // llama_grammar_init_impl's trigger loop (llama-grammar.cpp:1293-1298)
            llama_grammar_trigger_pattern tp;
            tp.pattern = pattern;
            tp.regex   = std::regex(tp.pattern);
            // llama_grammar_trigger_pattern::find (:378-409)
            const size_t pos = tp.find(input);
            if (pos == std::string::npos) {
                printf("FIND %s %s npos\n", phex.c_str(), bhex.c_str());
            } else {
                printf("FIND %s %s %zu\n", phex.c_str(), bhex.c_str(), pos);
            }
        } catch (const std::exception & e) {
            printf("FIND %s %s REGEX_ERR %s\n", phex.c_str(), bhex.c_str(), e.what());
        }
    }
    return 0;
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

static int run_accept(int argc, char ** argv) {
    std::string vocab_path, grammar_path, text, case_name = "case";
    std::vector<std::string> patterns;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--vocab" && i + 1 < argc) {
            vocab_path = argv[++i];
        } else if (a == "--grammar" && i + 1 < argc) {
            grammar_path = argv[++i];
        } else if (a == "--pattern" && i + 1 < argc) {
            patterns.push_back(unhex(argv[++i]));
        } else if (a == "--text" && i + 1 < argc) {
            text = argv[++i];
        } else if (a == "--case" && i + 1 < argc) {
            case_name = argv[++i];
        }
    }
    if (vocab_path.empty() || grammar_path.empty() || patterns.empty() || text.empty()) {
        fprintf(stderr, "--accept needs --vocab, --grammar, --pattern and --text\n");
        return 1;
    }

    llama_backend_init();

    llama_model * model = nullptr;
    const llama_vocab * vocab = load_vocab(vocab_path.c_str(), &model);

    const std::string gstr = read_file(grammar_path);

    std::vector<const char *> patterns_c;
    patterns_c.reserve(patterns.size());
    for (const auto & p : patterns) {
        patterns_c.push_back(p.c_str());
    }

    // llama_grammar_init_impl (llama-grammar.cpp:1216-1315) with lazy=true
    llama_grammar * g = llama_grammar_init_impl(
            vocab, gstr.c_str(), "root", /* lazy */ true,
            patterns_c.data(), patterns_c.size(), nullptr, 0);
    if (!g) {
        printf("CASE %s\nGRAMMAR %s\nINIT_FAIL\n", case_name.c_str(), hex(gstr).c_str());
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

    // tokenize with the reference vocab (add_special=false, parse_special=true)
    std::vector<llama_token> toks(text.size() + 64);
    const int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(),
                                     toks.data(), (int32_t) toks.size(), false, true);
    if (n < 0) {
        fprintf(stderr, "tokenization failed\n");
        return 1;
    }
    toks.resize(n);

    printf("CASE %s\n", case_name.c_str());
    printf("GRAMMAR %s\n", hex(gstr).c_str());
    for (const auto & p : patterns) {
        printf("PATTERN %s\n", hex(p).c_str());
    }
    for (size_t i = 0; i < toks.size(); i++) {
        printf("TOK %zu %d %s\n", i, toks[i], hex(vocab->token_to_piece(toks[i])).c_str());
    }

    for (size_t i = 0; i < toks.size(); i++) {
        try {
            // llama_grammar_accept_impl's awaiting-trigger branch
            // (llama-grammar.cpp:1398-1455)
            llama_grammar_accept_impl(*g, toks[i]);
        } catch (const std::exception & e) {
            printf("STEP %zu %d %zu %zu ACCEPT_FAILED\n",
                   i, g->awaiting_trigger ? 1 : 0, g->trigger_buffer.size(),
                   llama_grammar_get_stacks(g).size());
            llama_grammar_free_impl(g);
            llama_model_free(model);
            llama_backend_free();
            return 0;
        }
        printf("STEP %zu %d %zu %zu\n",
               i, g->awaiting_trigger ? 1 : 0, g->trigger_buffer.size(),
               llama_grammar_get_stacks(g).size());
    }

    llama_grammar_free_impl(g);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

int main(int argc, char ** argv) {
    if (argc > 1 && std::string(argv[1]) == "--find") {
        if (argc < 3) {
            fprintf(stderr, "--find needs a cases file\n");
            return 1;
        }
        return run_find(argv[2]);
    }
    if (argc > 1 && std::string(argv[1]) == "--accept") {
        return run_accept(argc, argv);
    }
    fprintf(stderr, "usage: %s --find CASES | --accept --vocab V --grammar G --pattern HEX [--text T]\n", argv[0]);
    return 1;
}
