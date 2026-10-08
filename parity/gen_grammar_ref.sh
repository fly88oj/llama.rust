#!/usr/bin/env bash
# gen_grammar_ref.sh — regenerate the GBNF parity artifacts (agent Z).
#
#   parity/grammar_ref.txt        parser dumps + matcher replays (text)
#   parity/grammar_pieces_ref.bin full qwen2 piece table (binary)
#
# Requires the pinned reference build:
#   /home/jeffrey/llm/llama.cpp-pinned            (source, bd4f514db1)
#   /home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin (libllama.so)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN=/home/jeffrey/llm/llama.cpp-pinned
REF=/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin
VOCAB="$PIN/models/ggml-vocab-qwen2.gguf"

DUMP="$ROOT/parity/ref_grammar_dump"

g++ -O2 -std=c++17 -o "$DUMP" "$ROOT/parity/ref_grammar_dump.cpp" \
    -I"$PIN/src" -I"$PIN/include" -I"$PIN/ggml/include" \
    -L"$REF" -lllama -lggml-base -Wl,-rpath,"$REF"

export LLAMA_LOG_LEVEL=error

echo "== parser dumps =="
"$DUMP" --parser \
    "$PIN"/grammars/*.gbnf > "$ROOT/parity/grammar_ref.txt" 2>/dev/null

echo "== matcher replays =="
run() { "$DUMP" --vocab "$VOCAB" "$@" >> "$ROOT/parity/grammar_ref.txt" 2>/dev/null; }

# json.gbnf: a small object, then an empty-ish structure
run --grammar "$PIN/grammars/json.gbnf" --tokens '{"a": 1}'
run --grammar "$PIN/grammars/json.gbnf" --tokens '{"name": "abc", "n": -1.5e3}'
run --grammar "$PIN/grammars/json_arr.gbnf" --tokens '[
1,
2]'
run --grammar "$PIN/grammars/arithmetic.gbnf" --tokens '1+1=2
(x1 + 3)*4=100
'
run --grammar "$PIN/grammars/list.gbnf" --tokens '- first item
- second item
'
# multi-byte / partial UTF-8: emoji, then kana through japanese.gbnf ranges
run --grammar "$PIN/grammars/json.gbnf" --tokens '{"k": "🔵🟠ok"}'
run --grammar "$PIN/grammars/japanese.gbnf" --tokens 'こんにちは。'
# byte-split UTF-8: accept the raw bytes of 🔵 (f0 9f 94 b5) one byte-token at a
# time inside a [\x{1F535}-\x{1F7E0}] range; ids are qwen2 single-byte tokens
BYTES_GRAMMAR=/tmp/emoji_range.gbnf
printf 'root ::= [\\U0001F535-\\U0001F7E0]+\n' > "$BYTES_GRAMMAR"
# single-byte pieces of qwen2 (dump once: 152k lines, awk exits early otherwise)
"$DUMP" --vocab "$VOCAB" --pieces 2>/dev/null > /tmp/qwen2_pieces.txt
byte_id() { awk -v h="$1" '$1 == "PIECE" && $3 == h { print $2; exit }' /tmp/qwen2_pieces.txt; }
E0=$(byte_id f0); E1=$(byte_id 9f); E2=$(byte_id 94); E3=$(byte_id b5)
echo "byte tokens: $E0 $E1 $E2 $E3"
run --grammar "$BYTES_GRAMMAR" --token-ids "$E0,$E1,$E2,$E3,$E0,$E1,$E2,$E3"
rm -f "$BYTES_GRAMMAR"

echo "== piece table =="
"$DUMP" --vocab "$VOCAB" --pieces-bin "$ROOT/parity/grammar_pieces_ref.bin" 2>/dev/null

wc -l "$ROOT/parity/grammar_ref.txt"