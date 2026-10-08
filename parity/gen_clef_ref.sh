#!/usr/bin/env bash
# gen_clef_ref.sh — sync batch A2: regenerate the clef decision-model
# reference dump from the NEW reference (a7b94df2c) and re-run the port's
# acceptance test.
#
# The probe drives the same public path as the reference server's
# /v1/systemone (llama_batch_ext + llama_batch_ext_set_decision_order +
# llama_process(DECODER) — the reference reroutes the memory-less arch
# through encode(), the path clef.cpp:393-407 anticipates).
#
# usage: bash parity/gen_clef_ref.sh [ref|cmp]
set -uo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-all}"
REFBIN=/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin
NEXT=/home/jeffrey/llm/llama.cpp-next
WORK=/tmp/s2m-clef
SYNTH=/tmp/clef-synth/clef-synth.gguf

mkdir -p "$WORK"

# 1. the synthetic clef file (qwen35 trunk + the joint decision head)
if [ ! -f "$SYNTH" ]; then
    ./parity/limited.sh -- cargo test --release -p llama --test clef_e2e \
        clef_write_synth_file -- --ignored --nocapture
fi

# 2. the reference dump
if [ "$MODE" = ref ] || [ "$MODE" = all ]; then
    # C++ — the staging-API setter exports with C++ linkage only
    g++ -O2 -x c++ -I"$NEXT/include" -I"$NEXT/ggml/include" \
        parity/ref_clef_dump.c -o "$WORK/ref_clef_dump" \
        -L"$REFBIN" -lllama -lggml -lggml-base -lggml-cpu -lm \
        -Wl,-rpath,"$REFBIN"
    "$WORK/ref_clef_dump" "$SYNTH" parity/clef_scores_ref.bin --fa off
fi

# 3. the port's acceptance (bit-exact scores vs the dump)
if [ "$MODE" = cmp ] || [ "$MODE" = all ]; then
    ./parity/limited.sh -- cargo test --release -p llama --test clef_e2e \
        clef_scores_match_reference -- --nocapture
fi
