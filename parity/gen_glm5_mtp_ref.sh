#!/usr/bin/env bash
# gen_glm5_mtp_ref.sh — the glm5-next MTP (b9acf138a) reference dumps:
# synthesize the glm5-next nextn GGUF with the port's writer
# (tests/glm5_mtp_e2e.rs glm5_mtp_write_synth_and_chain — the batch-19
# fixture + the NextN block), then drive the NEW reference's own MTP context
# (parity/ref_mtp2_dump — arch agnostic: ctx_type = MTP + load_mtp) over it
# and write /tmp/mtp2/glm5-next-ref.bin (+ the optional --nodes stream for
# the 12-step node-stream bisect).
#
# The port side of the bit-compare is /tmp/mtp2/glm5-next-port.bin (the
# default glm5_mtp_synth_load_and_chain test); the comparison is
# tests/glm5_mtp_e2e.rs's #[ignore]d glm5_mtp_reference_bitcompare.
#
# usage: bash parity/gen_glm5_mtp_ref.sh
set -u
cd "$(dirname "$0")/.."

OUT=/tmp/mtp2
mkdir -p $OUT

# 1) the synthetic file + the port chain (idempotent; also writes the port dump)
bash parity/limited.sh -- cargo test --release -p llama --test glm5_mtp_e2e glm5_mtp_write_synth_and_chain -- --ignored --nocapture || exit 1
[ -f "$OUT/glm5-next-synth-mtp.gguf" ] || { echo "missing synth"; exit 1; }
[ -f "$OUT/glm5-next-port.bin" ] || { echo "missing port dump"; exit 1; }

# 2) build a FRESH probe against the NEW reference tree (the shared
# parity/ref_mtp2_dump binary may predate glm5-next — "unknown model
# architecture" — so each run links its own copy under /tmp)
NEXT=/home/jeffrey/llm/llama.cpp-next
REF="$NEXT/build-rust-ref/bin"
PROBE=$OUT/ref_mtp2_dump_glm5
g++ -O2 -I"$NEXT/include" \
    -I"$NEXT/ggml/include" \
    -I"$NEXT/src" \
    parity/ref_mtp2_dump.c -o "$PROBE" \
    -L"$REF" -lllama -lggml -lggml-base -lggml-cpu -lm \
    -Wl,-rpath,"$REF" || exit 1

# 3) the reference chain (12 steps; --nodes for the node-stream bisect)
if bash parity/limited.sh -- env LD_LIBRARY_PATH="$REF" "$PROBE" "$OUT/glm5-next-synth-mtp.gguf" "$OUT/glm5-next-ref.bin" \
        --steps 12 --nodes "$OUT/glm5-next-nodes-ref.bin" 2> "$OUT/glm5-next-ref.err"; then
    echo "glm5-next mtp: ref dump ok ($(stat -c%s $OUT/glm5-next-ref.bin) bytes, nodes $(stat -c%s $OUT/glm5-next-nodes-ref.bin))"
else
    echo "glm5-next mtp: REF DUMP FAILED — $(tail -2 $OUT/glm5-next-ref.err | tr '\n' ' ')"
    exit 1
fi

# 4) the value bit-compare
bash parity/limited.sh -- cargo test --release -p llama --test glm5_mtp_e2e glm5_mtp_reference_bitcompare -- --ignored --nocapture || exit 1

echo "ok — artifacts in $OUT (glm5-next-{synth-mtp.gguf,ref.bin,port.bin,nodes-ref.bin})"
