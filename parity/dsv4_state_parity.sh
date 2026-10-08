#!/usr/bin/env bash
# dsv4_state_parity.sh — the DSV4 sequence-state byte-format cell (PARITY.md's
# dsv4-state section): the port's `DecodeContext::state_seq_get_data` blob
# must equal the reference's `llama_state_seq_get_data` byte-for-byte on the
# same synthetic deepseek4 model (the batch-7 generator, tests/
# dsv4_state_e2e.rs), after the same fixed 16-token prefill — and again with
# an 8-step tail that rolls the compressor rings past block boundaries.
#
# The context geometry is pinned on both sides: n_ctx 512, n_seq_max 1 (one
# compressed stream), n_ubatch 512, flash attention ON (the dsv4 raw pair is
# built with `v_trans = !flash_attn`, llama-model.cpp:2480 — only the
# v_trans = 0 layout matches the port's always-!v_trans caches).
#
# usage: bash parity/dsv4_state_parity.sh   (regenerates the model + blobs)
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PIN=/home/jeffrey/llm/llama.cpp-pinned
REF=/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin
OUT=/tmp/arch-dsv4state
mkdir -p $OUT

fail=0

# 1. the port side: the synthetic model + the two blobs
cargo test --release -p llama --test dsv4_state_e2e dsv4_state_dump_blob -- --ignored \
    > $OUT/port-dump.log 2>&1 || { echo "PORT DUMP FAILED"; tail -5 $OUT/port-dump.log; exit 1; }

# 2. the reference probe (parity/ref_dsv4_state.c)
PROBE=$OUT/ref_dsv4_state
gcc -O2 "$ROOT/parity/ref_dsv4_state.c" -o "$PROBE" \
    -I "$PIN/include" -I "$PIN/ggml/include" \
    -L "$REF" -lllama -Wl,-rpath,"$REF" || { echo "PROBE BUILD FAILED"; exit 1; }

for tail in 0 8; do
    LLAMA_LOG_LEVEL=error "$PROBE" "$OUT/deepseek4-synth-state.gguf" "$OUT/ref-state$([ $tail -eq 0 ] && echo '' || echo -n -tail).bin" $tail \
        > $OUT/probe-$tail.log 2>&1 || { echo "PROBE RUN FAILED (tail $tail)"; tail -3 $OUT/probe-$tail.log; fail=1; continue; }
done

# 3. byte comparison
for pair in "port-state.bin ref-state.bin prefill-only" \
            "port-state-tail.bin ref-state-tail.bin with-8-step-tail"; do
    set -- $pair
    if cmp -s "$OUT/$1" "$OUT/$2"; then
        echo "  $3: port == reference, byte-identical ($(stat -c%s "$OUT/$1") bytes) PASS"
    else
        echo "  $3: BYTE MISMATCH (port $(stat -c%s "$OUT/$1") vs ref $(stat -c%s "$OUT/$2")) FAIL"
        cmp "$OUT/$1" "$OUT/$2" | head -2
        fail=1
    fi
done

if [ $fail -eq 0 ]; then
    echo "DSV4 STATE BYTE FORMAT PARITY PASS"
else
    echo "DSV4 STATE PARITY FAILURES"
fi
exit $fail
