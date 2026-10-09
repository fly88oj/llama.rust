// ref_k2_plamo_split.c — the NEW-reference twin of the port's K2-Horizon
// splitter and PLaMo-3 pre-segmentation (462524043 + abeada335).
//
// unicode.cpp's symbols are not exported by libllama.so, so the probe
// compiles the reference's own unicode.cpp from the NEW tree
// (/home/jeffrey/llm/llama.cpp-next @c35b66744) and links against it; the
// PLaMo-3 pass-1/pass-2 bodies below are copied verbatim from
// llama-vocab.cpp:1550-1599 (they only need unicode_cpt_flags_from_cpt).
//
// usage: ref_k2_plamo_split <mode>   (mode: k2 | plamo3)
//   k2     — one split word per line, "U+XXXX..." codepoints per word
//   plamo3 — one segment per line, same format
#include "unicode.h"

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

static const char * K2_REGEX =
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\u200C|\\u200D)+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

// the corpus: ZWNJ/ZWJ inside letter runs, combining marks, 1-3 digit runs,
// contractions with long s, punctuation with \r\n tails, whitespace runs
static const std::vector<std::string> K2_CORPUS = {
    "Amy\u200Ckhaham",                 // ZWNJ keeps the letter run intact
    "a\u200Db",                        // ZWJ inside a run
    "code\u0301e\u0301",               // combining marks inside a run
    "12345",                           // digit run splits at 3
    "12",
    "'S 're 'Ll",                      // case-insensitive contractions
    "'\u017F",                         // long s folds to 's
    " !x??\r\n",                       // punct run with \r\n tail
    "a  b   c",                        // space runs
    "\u200Cleading",                   // ZWNJ starts a run
    "\u6C49\u5B57abc123",              // han + letters + digits
};

// the PLaMo-3 corpus: special-token fencing (incl. empty body, 64+ body,
// whitespace bodies), repeated-char runs, space runs, the U+EE00 boundary
static const std::vector<std::string> PLAMO3_CORPUS = {
    "<|plamo:chat|>hello<|plamo:eos|>",
    "<|plamo:|>x",                      // empty body is valid
    "<|plamo:\u001Dsep|>y",             // U+001C..1F treated as whitespace ends the body
    "<|plamo:notspecial",               // no |> terminator — not fenced
    "<|plamo:a\U0001F600|>z",           // emoji body is fine
    "aaaa",
    "aaa",
    "ababab",
    "   x",
    "\uFEFFkeep",                       // leading BOM is KEPT (PLaMo-3)
    "pre\uEE00post",                    // U+EE00 splits, not emitted
    "<|plamo:fim_prefix|>\uEE00<|plamo:fim_suffix|>",
};

// ---- verbatim from llama-vocab.cpp:1550-1599 (llm_tokenizer_plamo2::encode
// ---- pre-segmentation), reduced to return the segments ----
static std::vector<std::vector<uint32_t>> plamo3_segments(const std::vector<uint32_t> & unicode_data) {
    const size_t n = unicode_data.size();
    std::vector<bool> cut(n + 1, false);

    // pass 1: <|plamo:...|>
    {
        static constexpr uint32_t prefix[]   = { '<', '|', 'p', 'l', 'a', 'm', 'o', ':' };
        const size_t              prefix_len = std::size(prefix);
        size_t                    i          = 0;
        while (i + prefix_len <= n) {
            if (!std::equal(prefix, prefix + prefix_len, unicode_data.begin() + i)) {
                i++;
                continue;
            }
            size_t j = i + prefix_len;
            while (j < n && j - (i + prefix_len) < 64 && unicode_data[j] != '|' &&
                   (unicode_data[j] < 0x1C || unicode_data[j] > 0x1F) &&
                   !unicode_cpt_flags_from_cpt(unicode_data[j]).is_whitespace) {
                j++;
            }
            if (j + 1 < n && unicode_data[j] == '|' && unicode_data[j + 1] == '>') {
                cut[i]     = true;
                cut[j + 2] = true;
                i          = j + 2;
            } else {
                i++;
            }
        }
    }

    // pass 2: runs of repeated characters / spaces
    {
        size_t i = 0;
        while (i < n) {
            const uint32_t c   = unicode_data[i];
            size_t         run = 1;
            while (i + run < n && unicode_data[i + run] == c && !cut[i + run]) {
                run++;
            }

            const bool is_repeated_chars = c != '\n' && run >= 4;
            const bool is_spaces         = c == ' ' && run >= 2;
            if (is_repeated_chars || is_spaces) {
                cut[i]       = true;
                cut[i + run] = true;
            }

            i += run;
        }
    }

    std::vector<std::vector<uint32_t>> segments;
    size_t seg_start = 0;
    for (size_t seg_end = 0; seg_end <= n; ++seg_end) {
        const bool is_boundary = seg_end < n && unicode_data[seg_end] == 0xEE00;
        if (seg_end == n || cut[seg_end] || is_boundary) {
            if (seg_start < seg_end) {
                segments.emplace_back(unicode_data.begin() + seg_start, unicode_data.begin() + seg_end);
            }
            seg_start = seg_end + is_boundary;
        }
    }
    return segments;
}

static void print_cpts(const std::vector<uint32_t> & cpts) {
    for (uint32_t c : cpts) {
        printf("U+%04X ", c);
    }
    printf("\n");
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <k2|plamo3>\n", argv[0]);
        return 1;
    }
    const std::string mode = argv[1];
    if (mode == "k2") {
        for (const auto & text : K2_CORPUS) {
            const auto words = unicode_regex_split(text, { K2_REGEX }, false);
            for (const auto & w : words) {
                print_cpts(unicode_cpts_from_utf8(w));
            }
            printf("---\n");
        }
    } else if (mode == "plamo3") {
        for (const auto & text : PLAMO3_CORPUS) {
            for (const auto & seg : plamo3_segments(unicode_cpts_from_utf8(text))) {
                print_cpts(seg);
            }
            printf("---\n");
        }
    } else {
        fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 1;
    }
    return 0;
}
