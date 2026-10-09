#!/usr/bin/env bash
# gen_lfm2_decision_ref.sh — the lfm2 d1/d1-omni decision-model reference
# artifacts vs the NEW reference build (c35b66744, 88dcc460d + a657f7e98):
# banner parity (ref_model_saver vs the port's llama-cli) + the node stream
# (ref_decode_dump --fa off — decode reroutes the null-memory decision model
# to encode) vs tests/lfm2_decision_e2e.rs's dump driver, compared by the
# named-node pairing (lfm2d_reference_bitcompare).
#
# usage: bash parity/gen_lfm2_decision_ref.sh
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=/tmp/synca/lfm2d
mkdir -p "$OUT"
NEXT=/home/jeffrey/llm/llama.cpp-next
REF="$NEXT/build-rust-ref/bin"
PROMPT="The capital of France is"

echo "== 1) port writer"
bash parity/limited.sh -- cargo test --release -p llama --test lfm2_decision_e2e lfm2d_write_synth_file -- --ignored --nocapture || exit 1
SYNTH="$OUT/lfm2-decision-synth.gguf"
[ -f "$SYNTH" ] || { echo "missing $SYNTH" >&2; exit 1; }

echo "== 2) banner parity"
gcc -O2 parity/ref_model_saver.c -o "$OUT/ref_model_saver" \
    -I"$NEXT/include" -I"$NEXT/ggml/include" -L"$REF" -lllama -lggml -lggml-base || exit 1
gcc -O2 parity/ref_decode_dump.c -o "$OUT/ref_decode_dump" \
    -I"$NEXT/include" -I"$NEXT/ggml/include" \
    -L"$REF" -lllama -lggml -lggml-base -lggml-cpu -lm \
    -Wl,-rpath,"$REF" || exit 1

fail=0
if ! bash parity/limited.sh -- env LD_LIBRARY_PATH="$REF" "$OUT/ref_model_saver" "$SYNTH" "$OUT/ref_saved.gguf" \
        2> "$OUT/banner_ref.err"; then
    echo "REFERENCE LOAD FAILED:" >&2; tail -5 "$OUT/banner_ref.err" >&2; exit 1
fi
rm -f "$OUT/ref_saved.gguf"
grep -E "^print_info:" "$OUT/banner_ref.err" | sed 's/^print_info: //' > "$OUT/banner_ref.txt"

# the graph abort after the banner is expected (no CLI routing)
bash parity/limited.sh -- ./target/release/llama-cli -m "$SYNTH" -p hi -n 1 -t 8 -c 512 --temp 0 \
    > /dev/null 2> "$OUT/banner_port.err" || true
grep -E "^print_info:" "$OUT/banner_port.err" | sed 's/^print_info: //' > "$OUT/banner_port.txt"

diff -u "$OUT/banner_ref.txt" "$OUT/banner_port.txt" > "$OUT/banner_diff.txt" || true
n=$(grep -cE '^[+-][^+-]' "$OUT/banner_diff.txt" || true)
if [ "$n" -eq 0 ]; then
    echo "   banner: identical ($(wc -l < "$OUT/banner_ref.txt") lines)"
else
    echo "   banner: $n DIFFERING lines:" >&2
    cat "$OUT/banner_diff.txt" >&2
    fail=1
fi

echo "== 3) reference node stream + port dump + bitcompare"
if bash parity/limited.sh -- env LD_LIBRARY_PATH="$REF" "$OUT/ref_decode_dump" "$SYNTH" "$OUT/ref.bin" \
        --fa off --text "$PROMPT" 2> "$OUT/graph_ref.err"; then
    echo "   ref dump ok ($(stat -c%s "$OUT/ref.bin") bytes)"
else
    echo "   REF DUMP FAILED — $(tail -2 "$OUT/graph_ref.err" | tr '\n' ' ')" >&2
    exit 1
fi
bash parity/limited.sh -- env LFM2D_DUMP_OUT="$OUT/port.bin" \
    cargo test --release -p llama --test lfm2_decision_e2e lfm2d_prefill_node_dump -- --ignored --nocapture \
    > /dev/null 2>&1 || { echo "   PORT DUMP FAILED" >&2; fail=1; }
bash parity/limited.sh -- cargo test --release -p llama --test lfm2_decision_e2e lfm2d_reference_bitcompare -- --ignored --nocapture || fail=1

exit $fail
