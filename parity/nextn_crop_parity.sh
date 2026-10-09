#!/usr/bin/env bash
# nextn_crop_parity.sh — the f0c41e016 crop-timing acceptance (M-domain task
# 3): one mimo2 trunk prefill in the unmasked-nextn configuration with a
# partial-output batch, node-stream compared port vs the NEW reference.
#
#   ref : parity/ref_nextn_crop_dump.c (a ref_decode_dump twin with
#         llama_set_embeddings_nextn(ctx, true, false) + logits on the last
#         row only) over the mtp2 mimo2 synth, --fa off
#   port: tests/nextn_crop_dump.rs's ignored nextn_crop_node_dump cell (the
#         same batch through DecodeContext::decode_batch)
# compare: the DEFAULT nextn_crop_nodes_bit_exact_vs_reference test
#          (glm5_dump's named-node pairing)
#
# usage: bash parity/nextn_crop_parity.sh
set -u
cd "$(dirname "$0")/.."

NEXT=/home/jeffrey/llm/llama.cpp-next
REF="$NEXT/build-rust-ref/bin"
OUT=/tmp/closm-nextn-crop
mkdir -p $OUT

# 1) the mimo2 synth (the mtp2 generator, idempotent)
bash parity/limited.sh -- cargo test --release -p llama --test mtp2_e2e mtp2_write_synth_files -- --ignored --nocapture > /dev/null || exit 1
SYNTH=/tmp/mtp2/mimo2-synth-mtp.gguf
[ -f "$SYNTH" ] || { echo "missing $SYNTH"; exit 1; }

# 2) the reference stream
gcc -O2 -I"$NEXT/include" -I"$NEXT/ggml/include" parity/ref_nextn_crop_dump.c -o "$OUT/ref_nextn_crop_dump" \
    -L"$REF" -lllama -lggml -lggml-base -lggml-cpu -lm -Wl,-rpath,"$REF" 2> "$OUT/build.err" || {
    # llama-ext.h needs the C++ frontend
    g++ -O2 -I"$NEXT/include" -I"$NEXT/ggml/include" -I"$NEXT/src" parity/ref_nextn_crop_dump.c -o "$OUT/ref_nextn_crop_dump" \
        -L"$REF" -lllama -lggml -lggml-base -lggml-cpu -lm -Wl,-rpath,"$REF" || exit 1
}
if ! bash parity/limited.sh -- env LD_LIBRARY_PATH="$REF" "$OUT/ref_nextn_crop_dump" "$SYNTH" "$OUT/nodes_ref.bin" \
        --fa off --text "1, 2, 3" 2> "$OUT/ref.err"; then
    echo "REFERENCE RUN FAILED:"; tail -3 "$OUT/ref.err"; exit 1
fi

# 3) the port stream + the DEFAULT comparison
bash parity/limited.sh -- cargo test --release -p llama --test nextn_crop_dump nextn_crop_node_dump -- --ignored --nocapture || exit 1
cp "$OUT/nodes_ref.bin" parity/mimo2/nextn_crop_nodes_ref.bin
bash parity/limited.sh -- cargo test --release -p llama --test nextn_crop_dump -- --nocapture || exit 1
echo "ok — ref stream stored at parity/mimo2/nextn_crop_nodes_ref.bin"
