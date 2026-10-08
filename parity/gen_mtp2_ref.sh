#!/usr/bin/env bash
# gen_mtp2_ref.sh — build the mtp2 reference dumps: synthesize the nine
# nextn GGUFs with the port's writer (cargo test -p llama --test mtp2_e2e
# mtp2_write_synth_files), then drive the pinned reference's own MTP context
# (parity/ref_mtp2_dump) over each and write /tmp/mtp2/<arch>-ref.bin.
#
# The port side of the bit-compare is /tmp/mtp2/<arch>-port.bin (written by
# the default mtp2_synth_load_and_chain test); the comparison itself is
# tests/mtp2_e2e.rs's #[ignore]d mtp2_reference_bitcompare.
#
# usage: bash parity/gen_mtp2_ref.sh
set -u
cd "$(dirname "$0")/.."

OUT=/tmp/mtp2
mkdir -p $OUT

# 1) the synthetic files (idempotent)
bash parity/limited.sh -- cargo test --release -p llama --test mtp2_e2e mtp2_write_synth_files -- --ignored --nocapture || exit 1

# 2) build the probe if missing
if [ ! -x parity/ref_mtp2_dump ]; then
    g++ -O2 -I/home/jeffrey/llm/llama.cpp-pinned/include \
        -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
        -I/home/jeffrey/llm/llama.cpp-pinned/src \
        parity/ref_mtp2_dump.c -o parity/ref_mtp2_dump \
        -L/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin \
        -lllama -lggml -lggml-base -lggml-cpu -lm \
        -Wl,-rpath,/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin || exit 1
fi

# 3) the reference chain per arch (fresh process each — the MTP context is
#    one-shot per model)
fail=0
for arch in qwen35 qwen35moe qwen3next glm4moe cohere2moe bailingmoe3 hy_v3 mimo2 step35; do
    f=$OUT/$arch-synth-mtp.gguf
    [ -f "$f" ] || { echo "missing $f"; fail=1; continue; }
    if bash parity/limited.sh -- ./parity/ref_mtp2_dump "$f" "$OUT/$arch-ref.bin" 2> "$OUT/$arch-ref.err"; then
        echo "$arch: ref dump ok ($(stat -c%s $OUT/$arch-ref.bin) bytes)"
    else
        echo "$arch: REF DUMP FAILED — $(tail -2 $OUT/$arch-ref.err | tr '\n' ' ')"
        fail=1
    fi
done

exit $fail
