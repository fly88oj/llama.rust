#!/usr/bin/env bash
# gen_lazy_trigger_ref.sh — regenerate the lazy PATTERN-trigger parity dump
# (agent: lazy PATTERN trigger engine).
#
#   parity/lazy_trigger_ref.txt
#       FIND lines   — fire positions of the reference
#                     `llama_grammar_trigger_pattern::find`
#                     (llama-grammar.cpp:378-409) for every (pattern, buffer)
#                     pair below, incl. incremental token-boundary prefixes
#       CASE blocks  — the reference's lazy accept flow
#                     (`llama_grammar_accept_impl`, llama-grammar.cpp:1398-1455)
#                     driven token-by-token over the real qwen2 vocabulary
#
# Compared bit-for-bit by crates/tools/llama-server's
# engine::lazy_trigger_tests::lazy_trigger_reference_replay.
#
# Requires the pinned reference build:
#   /home/jeffrey/llm/llama.cpp-pinned            (source, bd4f514db1)
#   /home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin (libllama.so)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN=/home/jeffrey/llm/llama.cpp-pinned
REF=/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin
VOCAB="$PIN/models/ggml-vocab-qwen2.gguf"
PROBE="$ROOT/parity/ref_lazy_trigger_probe"
OUT="$ROOT/parity/lazy_trigger_ref.txt"

g++ -O2 -std=c++17 -o "$PROBE" "$ROOT/parity/ref_lazy_trigger_probe.cpp" \
    -I"$PIN/src" -I"$PIN/include" -I"$PIN/ggml/include" \
    -L"$REF" -lllama -lggml-base -Wl,-rpath,"$REF"

export LLAMA_LOG_LEVEL=error

# ---- the (pattern, buffer) matrix ------------------------------------------
# the six trigger patterns the specialized chat parsers emit, plus synthetic
# ones pinning individual constructs (\s bytes, `.`, {m,n}, captures under ^…$)
CASES="$(mktemp)"
python3 - "$CASES" <<'PYEOF'
import sys

def h(s):
    if isinstance(s, str):
        s = s.encode("utf-8")
    return s.hex()

out = open(sys.argv[1], "w")

P1 = r"^\s+to$"
P2 = r"^<\|channel\|>(?:commentary|analysis)\s+to=functions$"
P3 = r"<\|start\|>assistant(\s+to)"
P4 = r"<\|start\|>assistant(<\|channel\|>(?:commentary|analysis)\s+to)"
P5 = (r"(?:^|<\|start\|>assistant)"
      r"( to=(?!self<\|message\|>)(?!user<\|message\|>)[^<]*?<\|message\|>)")
P6 = r">>>(?!all)"
PATTERNS = [P1, P2, P3, P4, P5, P6]

common = [
    "", " ", "  ", "to", " to", "  to", " to ", " tox", "x to", "\t\n to",
    "to ", " tto", " t to", "\n\n to", " \t\r\n\v\fto",
]
chan = [
    "<|channel|>commentary to=functions",
    "<|channel|>analysis to=functions",
    "<|channel|>final to=functions",
    "x<|channel|>commentary to=functions",
    "<|channel|>commentary  to=functions",
    "<|channel|>commentary to=function",
    "<|channel|>commentary to=functions\n",
    "<|channel|>commentary",
    "<|channel|>commentary to=",
]
start = [
    "foo<|start|>assistant to",
    "<|start|>assistant to",
    "<|start|>assistant  to",
    "zz<|start|>assistant to",
    "<|start|>assistant tox",
    "<|start|>assistant",
    "<|start|>assistant ",
    "<|start|>assistant to\n",
]
muse = [
    " to=functions\n{\"a\":1}<|message|>",
    "<|start|>assistant to=functions<|message|>",
    " to=self<|message|>",
    " to=user<|message|>x",
    "x to=functions<|message|>",
    " to=abc\ndef<|message|>tail<|message|>",
    " to=functions",
    " to=functions\n",
    " to=a<b<|message|>",
    "<|start|>assistant to=self<|message|>",
    " to=functions\nx<|message|>",
    "to=functions<|message|>",
    "<|start|>assistant  to=functions<|message|>",
    " to=TOOL<|message|>",
]
fnry = [
    ">>>", ">>>all", ">>>x", ">>> ", ">>>al", ">>>allz", ">>>>all",
    "x>>>all", "x>>>get_weather", ">>", "> >>", ">>>all}", ">>>ALL",
    ">>>all>", ">> >>>", "say >>>get(",
]
per_pattern = {P1: common, P2: chan, P3: start, P4: start, P5: muse, P6: fnry}
all_buffers = sorted(set(common + chan + start + muse + fnry))

for p in PATTERNS:
    for b in all_buffers + per_pattern[p]:
        out.write(f"{h(p)} {h(b)}\n")

# synthetic construct probes
synth = [
    (r"^hello$", ["hello", "hell", "hellox", "xhello", ""]),
    (r"^(\s+to)$", [" to", "  to", "x to", " to "]),
    (r"^x(\s+to)$", ["x to", "x  to", "x toz"]),
    (r"^a{2,3}$", ["a", "aa", "aaa", "aaaa"]),
    (r"^a{2,}$", ["a", "aa", "aaaaa"]),
    (r"^a{2}b$", ["aab", "ab", "aaab"]),
    (r"^\s$", [bytes([b]) for b in
               [0x00, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x41, 0x7F, 0x85, 0xA0]]),
    (r"^.$", [bytes([b]) for b in
              [0x0A, 0x0D, 0x09, 0x41, 0x85, 0xA0, 0xFF]]),
    (r"^\S$", [bytes([b]) for b in [0x09, 0x0A, 0x20, 0x85, 0xA0, 0x41]]),
    (r"[^<]+<\|end\|>", ["abc<|end|>", "<|end|>", "a<b<|end|>", "abc<|en"]),
    (r"(?:commentary|analysis)\s+to", ["commentary to", "analysis  to",
                                       "commentaryto", "xcommentary to"]),
]
for p, bufs in synth:
    for b in bufs:
        out.write(f"{h(p)} {h(b)}\n")

# incremental token-boundary prefixes: every growing buffer a token stream
# produces, checked against every real pattern
streams = [
    ["<|", "start", "|>", "assistant", " to"],
    ["<|", "start", "|>", "assistant", "<|channel|>", "commentary", " to=functions"],
    [" ", " to"],
    ["  ", "to"],
    [" ", "to", "=functions"],
    [">>", ">", "a", "l", "l"],
    [">>", ">", "x"],
    [">", ">>>all"],
    [">>", ">", "get_weather"],
    [" to=functions", "\n{\"city\":\"Tokyo\"}", "<|message|>"],
    [" to=self", "<|message|>"],
    ["<|start|>assistant", " to=functions", "<|message|>"],
    ["x", ">>>", "all"],
]
for pieces in streams:
    buf = ""
    for piece in pieces:
        buf += piece
        for p in PATTERNS:
            out.write(f"{h(p)} {h(buf)}\n")

n = sum(1 for _ in open(sys.argv[1]))
print(f"generated {n} FIND cases", file=sys.stderr)
PYEOF

"$PROBE" --find "$CASES" > "$OUT"
rm -f "$CASES"
echo "== FIND dump: $(grep -c '^FIND' "$OUT") lines =="

# ---- the reference lazy accept flow (real qwen2 tokenization) ---------------
GDIR="$(mktemp -d)"
printf 'root ::= " to" "!"\n'                       > "$GDIR/gptoss3.gbnf"
printf 'root ::= "  to" "!"\n'                      > "$GDIR/gptoss1.gbnf"
printf 'root ::= [a-z_>-]+\n'                       > "$GDIR/fnry.gbnf"
printf 'root ::= " to=" [^<]+ "<|message|>"\n'      > "$GDIR/muse.gbnf"

run_case() { # name grammar pattern_hex text
    "$PROBE" --accept --vocab "$VOCAB" --grammar "$2" --pattern "$3" \
        --text "$4" --case "$1" >> "$OUT" 2>/dev/null
}
run_case gptoss3 "$GDIR/gptoss3.gbnf" \
    "$(python3 -c 'import sys; sys.stdout.write("<\\|start\\|>assistant(\\s+to)".encode().hex())')" \
    'Hello there<|start|>assistant to!'
run_case gptoss1 "$GDIR/gptoss1.gbnf" \
    "$(python3 -c 'import sys; sys.stdout.write("^\\s+to$".encode().hex())')" \
    '  to!'
run_case fnry_fires "$GDIR/fnry.gbnf" \
    "$(python3 -c 'import sys; sys.stdout.write(">>>(?!all)".encode().hex())')" \
    'x>>>get_weather'
run_case fnry_all "$GDIR/fnry.gbnf" \
    "$(python3 -c 'import sys; sys.stdout.write(">>>(?!all)".encode().hex())')" \
    'x>>>all done'
run_case muse "$GDIR/muse.gbnf" \
    "$(python3 -c 'import sys; sys.stdout.write(r"(?:^|<\|start\|>assistant)( to=(?!self<\|message\|>)(?!user<\|message\|>)[^<]*?<\|message\|>)".encode().hex())')" \
    "$(printf ' to=functions\n{"a":1}<|message|>')"

rm -rf "$GDIR"
echo "== ACCEPT cases: $(grep -c '^CASE' "$OUT") =="
echo "wrote $OUT"
