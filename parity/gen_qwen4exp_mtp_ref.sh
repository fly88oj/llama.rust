#!/usr/bin/env bash
# gen_qwen4exp_mtp_ref.sh — the qwen4exp MTP reference dumps (batch 42f):
# synthesize the qwen4exp nextn GGUF with the port's writer
# (tests/qwen4exp_mtp_e2e.rs — the qwen4exp_qsa_dump fixture + a QSA MTP
# block), then drive the NEW reference's own MTP context
# (parity/ref_mtp2_dump) over it and write /tmp/mtp2/qwen4exp-ref.bin.
#
# usage: bash parity/gen_qwen4exp_mtp_ref.sh
set -u
cd "$(dirname "$0")/.."

OUT=/tmp/mtp2
mkdir -p $OUT

# 1) the synthetic file + the port chain
bash parity/limited.sh -- cargo test --release -p llama --test qwen4exp_mtp_e2e qwen4exp_mtp_write_synth_and_chain -- --ignored --nocapture || exit 1
[ -f "$OUT/qwen4exp-synth-mtp.gguf" ] || { echo "missing synth"; exit 1; }
[ -f "$OUT/qwen4exp-port.bin" ] || { echo "missing port dump"; exit 1; }

# 2) build a FRESH probe against the NEW reference tree (the shared
# parity/ref_mtp2_dump binary may predate the arch — each run links its own
# copy under /tmp)
NEXT=/home/jeffrey/llm/llama.cpp-next
REF="$NEXT/build-rust-ref/bin"
PROBE=$OUT/ref_mtp2h_dump_q4e
g++ -O2 -I"$NEXT/include" \
    -I"$NEXT/ggml/include" \
    -I"$NEXT/src" \
    parity/ref_mtp2h_dump.c -o "$PROBE" \
    -L"$REF" -lllama -lggml -lggml-base -lggml-cpu -lm \
    -Wl,-rpath,"$REF" || exit 1

# 3) the reference chain (12 steps)
if bash parity/limited.sh -- env LD_LIBRARY_PATH="$REF" "$PROBE" "$OUT/qwen4exp-synth-mtp.gguf" "$OUT/qwen4exp-ref.bin" \
        --steps 12 2> "$OUT/qwen4exp-ref.err"; then
    echo "qwen4exp mtp: ref dump ok ($(stat -c%s $OUT/qwen4exp-ref.bin) bytes)"
else
    echo "qwen4exp mtp: REF DUMP FAILED — $(tail -2 $OUT/qwen4exp-ref.err | tr '\n' ' ')"
    exit 1
fi

# 4) the value bit-compare
bash parity/limited.sh -- cargo test --release -p llama --test qwen4exp_mtp_e2e qwen4exp_mtp_reference_bitcompare -- --ignored --nocapture || exit 1

echo "ok — artifacts in $OUT (qwen4exp-{synth-mtp.gguf,ref.bin,port.bin})"
