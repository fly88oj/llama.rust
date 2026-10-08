#!/usr/bin/env bash
# gen_t5_dec_ref.sh — the t5 decoder reference dump: synthesize the file with
# the port's writer (tests/t5_dec_e2e), then drive the pinned reference's own
# llama_encode/llama_decode pair (parity/ref_t5_dec_dump) and write
# /tmp/mtp2/t5-dec-ref.bin. The comparison is tests/t5_dec_e2e.rs's #[ignore]d
# t5_decoder_reference_bitcompare.
set -u
cd "$(dirname "$0")/.."

OUT=/tmp/mtp2
mkdir -p $OUT

bash parity/limited.sh -- cargo test --release -p llama --test t5_dec_e2e t5_decoder_load_and_chain -- --nocapture || exit 1

if [ ! -x parity/ref_t5_dec_dump ]; then
    g++ -O2 -I/home/jeffrey/llm/llama.cpp-pinned/include \
        -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
        -I/home/jeffrey/llm/llama.cpp-pinned/src \
        parity/ref_t5_dec_dump.c -o parity/ref_t5_dec_dump \
        -L/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin \
        -lllama -lggml -lggml-base -lggml-cpu -lm \
        -Wl,-rpath,/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin || exit 1
fi

if bash parity/limited.sh -- ./parity/ref_t5_dec_dump "$OUT/t5-synth-dec.gguf" "$OUT/t5-dec-ref.bin" --steps 12; then
    echo "t5: ref dump ok ($(stat -c%s $OUT/t5-dec-ref.bin) bytes)"
else
    echo "t5: REF DUMP FAILED"; exit 1
fi
