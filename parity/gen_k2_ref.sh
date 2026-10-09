#!/usr/bin/env bash
# gen_k2_ref.sh — build the K2 Horizon reference artifacts vs the NEW
# reference build (c35b66744, commit 462524043):
#   1. write the two synthetic k2-horizon GGUFs with the port's writer
#      (tests/k2_horizon_e2e.rs k2_write_synth_files),
#   2. load each with the NEW reference library (parity/ref_model_saver.c)
#      and byte-compare the print_info banner against the port's llama-cli
#      load banner (the glm5_parity.sh protocol),
#   3. drive the reference decoder probe (parity/ref_decode_dump.c,
#      --fa off --decode-tail 12, the same prompt as the port dump) and
#      write <name>-ref.bin; the port half is tests/k2_horizon_e2e.rs's
#      k2_prefill_node_dump (K2_SPEC/K2_DUMP_OUT env).
#
# usage: bash parity/gen_k2_ref.sh
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=/tmp/synca/k2
mkdir -p "$OUT"
NEXT=/home/jeffrey/llm/llama.cpp-next
REF="$NEXT/build-rust-ref/bin"
PROMPT="The capital of France is"

echo "== 1) port writer"
bash parity/limited.sh -- cargo test --release -p llama --test k2_horizon_e2e k2_write_synth_files -- --ignored --nocapture || exit 1

echo "== 2) banner parity (ref_model_saver vs port llama-cli)"
gcc -O2 parity/ref_model_saver.c -o "$OUT/ref_model_saver" \
    -I"$NEXT/include" -I"$NEXT/ggml/include" -L"$REF" -lllama -lggml -lggml-base || exit 1
gcc -O2 parity/ref_decode_dump.c -o "$OUT/ref_decode_dump" \
    -I"$NEXT/include" -I"$NEXT/ggml/include" \
    -L"$REF" -lllama -lggml -lggml-base -lggml-cpu -lm \
    -Wl,-rpath,"$REF" || exit 1

fail=0
for name in dense mova; do
    SYNTH="$OUT/k2-horizon-$name-synth.gguf"
    [ -f "$SYNTH" ] || { echo "missing $SYNTH"; fail=1; continue; }

    if ! bash parity/limited.sh -- env LD_LIBRARY_PATH="$REF" "$OUT/ref_model_saver" "$SYNTH" "$OUT/ref_saved.gguf" \
            2> "$OUT/$name-banner_ref.err"; then
        echo "$name: REFERENCE LOAD FAILED:" >&2; tail -5 "$OUT/$name-banner_ref.err" >&2; fail=1; continue
    fi
    rm -f "$OUT/ref_saved.gguf"
    grep -E "^print_info:" "$OUT/$name-banner_ref.err" | sed 's/^print_info: //' > "$OUT/$name-banner_ref.txt"

    # the graph abort after the banner is expected (no CLI routing); capture
    # stderr through the abort (the glm5_parity.sh convention)
    bash parity/limited.sh -- ./target/release/llama-cli -m "$SYNTH" -p hi -n 1 -t 8 -c 512 --temp 0 \
        > /dev/null 2> "$OUT/$name-banner_port.err" || true
    grep -E "^print_info:" "$OUT/$name-banner_port.err" | sed 's/^print_info: //' > "$OUT/$name-banner_port.txt"

    diff -u "$OUT/$name-banner_ref.txt" "$OUT/$name-banner_port.txt" > "$OUT/$name-banner_diff.txt" || true
    n=$(grep -cE '^[+-][^+-]' "$OUT/$name-banner_diff.txt" || true)
    if [ "$n" -eq 0 ]; then
        echo "   $name banner: identical ($(wc -l < "$OUT/$name-banner_ref.txt") lines)"
    else
        echo "   $name banner: $n DIFFERING lines:" >&2
        cat "$OUT/$name-banner_diff.txt" >&2
        fail=1
    fi
done

echo "== 3) reference node streams (--fa off --decode-tail 12)"
for name in dense mova; do
    SYNTH="$OUT/k2-horizon-$name-synth.gguf"
    if bash parity/limited.sh -- env LD_LIBRARY_PATH="$REF" "$OUT/ref_decode_dump" "$SYNTH" "$OUT/$name-ref.bin" \
            --fa off --decode-tail 12 --text "$PROMPT" 2> "$OUT/$name-graph_ref.err"; then
        echo "   $name: ref dump ok ($(stat -c%s "$OUT/$name-ref.bin") bytes)"
    else
        echo "   $name: REF DUMP FAILED — $(tail -2 "$OUT/$name-graph_ref.err" | tr '\n' ' ')" >&2
        fail=1
    fi
done

echo "== 4) port node streams + bitcompare"
for name in dense mova; do
    bash parity/limited.sh -- env K2_SPEC=$name K2_DUMP_OUT="$OUT/$name-port.bin" K2_DECODE_TAIL=12 \
        cargo test --release -p llama --test k2_horizon_e2e k2_prefill_node_dump -- --ignored --nocapture \
        > /dev/null 2>&1 || { echo "   $name: PORT DUMP FAILED" >&2; fail=1; continue; }
done
bash parity/limited.sh -- cargo test --release -p llama --test k2_horizon_e2e k2_reference_bitcompare -- --ignored --nocapture || fail=1

exit $fail
