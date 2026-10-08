// ref_json_schema_dump.cpp — reference dumps for the Rust JSON-schema → GBNF port (agent A).
//
// Links against the pinned llama.cpp build (bd4f514db1) and calls the *real*
// converter exported by libllama-common.so:
//
//   json_schema_to_grammar(common_json const&, bool)      (json-schema-to-grammar.cpp:993)
//   build_grammar(callback, common_grammar_options const&) (:1015, the dotall path)
//   llama_grammar_init_impl / llama_grammar_accept_token   (the reference matcher, for
//                                                           the accept/reject verdicts)
//
// Build (from the repo root), see parity/gen_json_schema_ref.sh:
//   g++ -O2 -std=c++17 -o parity/ref_json_schema_dump parity/ref_json_schema_dump.cpp \
//       -I/home/jeffrey/llm/llama.cpp-pinned/common \
//       -I/home/jeffrey/llm/llama.cpp-pinned/src \
//       -I/home/jeffrey/llm/llama.cpp-pinned/include \
//       -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
//       -L/home/jeffrey/llm/llama.cpp/build-rust-ref/bin -lllama-common -lllama -lggml-base \
//       -Wl,-rpath,/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
//
// Usage:
//   ref_json_schema_dump SCHEMA.json [...]              → one CASE block per schema file
//   ref_json_schema_dump --roundtrip SCHEMA.json [...]  → RT lines: common_json::parse().dump()
//   ref_json_schema_dump --dotall SCHEMA.json [...]     → same, via build_grammar(dotall=true)
//   ref_json_schema_dump --cases CASES.json             → CASES.json is
//        [{"name":..., "schema":{...}, "passing":[...], "failing":[...]}, ...]
//        each entry also gets the reference matcher's ACCEPT/REJECT verdicts.
//
// The text format is line-oriented, everything variable is hex-encoded, and it is
// compared verbatim by crates/llama/tests/json_schema_parity.rs.

#include "json-schema-to-grammar.h"
#include "json-schema.h"
#include "json.h"

#include "../src/llama-grammar.h"
#include "../src/unicode.h"

#include <cstdio>
#include <string>
#include <vector>

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
    std::string base = pos == std::string::npos ? path : path.substr(pos + 1);
    auto dot = base.find_last_of('.');
    return dot == std::string::npos ? base : base.substr(0, dot);
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

// ---------------------------------------------------------------- matcher ----
// The accept/reject semantics of tests/test-grammar-integration.cpp:56-114.

struct token_and_piece {
    llama_token token;
    std::string piece;
};

static std::vector<token_and_piece> parse_tokens(const std::string & input) {
    std::vector<token_and_piece> result;
    result.reserve(input.size());
    size_t offset = 0;
    while (offset < input.size()) {
        if (static_cast<unsigned char>(input[offset]) == 0xff) {
            if (offset + 5 > input.size()) {
                fprintf(stderr, "not enough bytes for token id\n");
                exit(1);
            }
            uint32_t val =
                (static_cast<unsigned char>(input[offset + 1]) << 24) |
                (static_cast<unsigned char>(input[offset + 2]) << 16) |
                (static_cast<unsigned char>(input[offset + 3]) << 8)  |
                (static_cast<unsigned char>(input[offset + 4]));
            auto piece = "<[" + std::to_string(val) + "]>";
            result.push_back({static_cast<llama_token>(val), piece});
            offset += 5;
        } else {
            uint32_t cpt = unicode_cpt_from_utf8(input, offset);
            result.push_back({0, unicode_cpt_to_utf8(cpt)});
        }
    }
    return result;
}

static bool match_string(const std::string & input, llama_grammar * grammar) {
    const auto parsed = parse_tokens(input);

    auto & stacks_cur = llama_grammar_get_stacks(grammar);

    for (const auto & in : parsed) {
        try {
            llama_grammar_accept_token(*grammar, in.token, in.piece);
        } catch (const std::runtime_error & /*e*/) {
            return false;
        }
        if (stacks_cur.empty()) {
            return false;
        }
    }

    for (const auto & stack : stacks_cur) {
        if (stack.empty()) {
            return true;
        }
    }

    return false;
}

static bool match_grammar(const std::string & gbnf, const std::string & input) {
    llama_grammar * grammar = llama_grammar_init_impl(
            nullptr, gbnf.c_str(), "root", false, nullptr, 0, nullptr, 0);
    if (grammar == nullptr) {
        return false;
    }
    const bool matched = match_string(input, grammar);
    llama_grammar_free_impl(grammar);
    return matched;
}

// ------------------------------------------------------------------- dump ----

/// one CASE block: the schema text, the generated GBNF, and (for the integration
/// cases) the reference matcher's verdict for every expected-pass / expected-fail
/// string.
static void dump_case(const std::string & name, const std::string & schema_text, bool dotall,
                      const std::vector<std::string> * passing,
                      const std::vector<std::string> * failing) {
    common_json schema;
    try {
        schema = common_json::parse(schema_text);
    } catch (const std::exception & e) {
        printf("CASE %s\nSCHEMA %s\nPARSE_FAIL %s\nEND\n",
               name.c_str(), hex(schema_text).c_str(), hex(e.what()).c_str());
        return;
    }

    std::string gbnf;
    bool failed = false;
    try {
        if (!dotall) {
            gbnf = json_schema_to_grammar(schema, true);
        } else {
            // the build_grammar() path — json-schema-to-grammar.cpp:1015-1028
            common_chat_schema_document doc = common_chat_schema_from_json(schema);
            common_grammar_options options;
            options.dotall = true;
            gbnf = build_grammar([&](const common_grammar_builder & builder) {
                builder.add_schema("root", *doc.root);
            }, options);
        }
    } catch (const std::exception & e) {
        failed = true;
        gbnf = e.what();
    }

    printf("CASE %s\n", name.c_str());
    printf("SCHEMA %s\n", hex(schema_text).c_str());
    if (failed) {
        printf("GBNF_FAIL %s\n", hex(gbnf).c_str());
    } else {
        printf("GBNF %s\n", hex(gbnf).c_str());
    }

    const std::vector<std::string> * lists[2] = { passing, failing };
    for (int kind = 0; kind < 2; kind++) {
        if (lists[kind] == nullptr) {
            continue;
        }
        for (const auto & str : *lists[kind]) {
            if (failed) {
                printf("%s ? %s\n", kind == 0 ? "ACCEPT" : "REJECT", hex(str).c_str());
                continue;
            }
            const bool matched = match_grammar(gbnf, str);
            // the reference must agree with its own test expectations
            if (matched != (kind == 0)) {
                printf("REF_MISMATCH %s %s\n", kind == 0 ? "ACCEPT" : "REJECT", hex(str).c_str());
            }
            printf("%s %s %s\n", kind == 0 ? "ACCEPT" : "REJECT", matched ? "1" : "0", hex(str).c_str());
        }
    }
    printf("END\n");
}

int main(int argc, char ** argv) {
    std::vector<std::string> schema_files;
    std::string cases_file;
    bool dotall = false;
    bool roundtrip = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--dotall") {
            dotall = true;
        } else if (a == "--roundtrip") {
            roundtrip = true;
        } else if (a == "--cases" && i + 1 < argc) {
            cases_file = argv[++i];
        } else if (a.rfind("--", 0) != 0) {
            schema_files.push_back(a);
        } else {
            fprintf(stderr, "unknown arg '%s'\n", a.c_str());
            return 1;
        }
    }

    for (const auto & f : schema_files) {
        const std::string name = basename_of(f);
        const std::string text = read_file(f);
        if (roundtrip) {
            // common_json::parse() + dump(), the JSON layer the converter prints
            // const/enum values through
            printf("RT %s %s %s\n", name.c_str(), hex(text).c_str(), hex(common_json::parse(text).dump()).c_str());
            continue;
        }
        dump_case(name, text, dotall, nullptr, nullptr);
    }

    if (!cases_file.empty()) {
        common_json cases = common_json::parse(read_file(cases_file));
        for (const auto & tc : cases) {
            const std::string name = tc.at("name").get<std::string>();
            std::vector<std::string> passing, failing;
            for (const auto & x : tc.at("passing")) {
                passing.push_back(x.get<std::string>());
            }
            for (const auto & x : tc.at("failing")) {
                failing.push_back(x.get<std::string>());
            }
            dump_case("cases/" + name, tc.at("schema").dump(), false, &passing, &failing);
        }
    }

    return 0;
}