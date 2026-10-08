#!/usr/bin/env bash
# ling3vl_parity.sh — the ling3vl projector (models/ling3vl.cpp, upstream
# f830688e9 "model : add Ling 3.0 VL support") against the NEW reference:
# the port's synthetic ling3vl mmproj (crates/llama/src/clip.rs
# `synth_mmproj_ling3vl_load_preprocess_encode` with LLAMA_E2E_DUMP) encodes
# parity/mtmd-fixture.png; the reference llama-mtmd-cli encodes the same
# image from the same mmproj and dumps MTMD_DEBUG_EMBEDDINGS — byte compare.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF_BIN_DIR="${REF_BIN_DIR:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin}"
PORT_BIN="${PORT_BIN:-$ROOT/target/release/llama-mtmd-cli}"
MMPROJ="/tmp/llama-rust-mtmd/synth-ling3vl-mmproj.gguf"
PNG="$ROOT/parity/mtmd-fixture.png"
PORT_DUMP=/tmp/ling3vl-port.bin
REF_DUMP=/tmp/ling3vl-ref.bin

export LD_LIBRARY_PATH="$REF_BIN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 1. the port's embedding dump (also writes the synth mmproj)
if [ ! -f "$MMPROJ" ] || [ "$PORT_BIN" -nt "$PORT_DUMP" ]; then
    (cd "$ROOT" && cargo test -p llama --lib synth_mmproj_ling3vl -- --nocapture >/dev/null 2>&1)
    LLAMA_E2E_DUMP=1 cargo test -p llama --lib synth_mmproj_ling3vl -- --nocapture 2>&1 | grep -q "dumped" \
        || { echo "FAIL: port dump missing (test output above)"; exit 1; }
fi

# 2. the reference on the same mmproj + image
if [ ! -x "$REF_BIN_DIR/llama-mtmd-cli" ]; then
    echo "SKIP: $REF_BIN_DIR/llama-mtmd-cli not built yet"; exit 0
fi
MTMD_DEBUG_EMBEDDINGS="$REF_DUMP" "$REF_BIN_DIR/llama-mtmd-cli" \
    -m /home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --mmproj "$MMPROJ" --image "$PNG" -p "describe" -n 0 >/tmp/ling3vl-ref.log 2>&1
[ -s "$REF_DUMP" ] || { echo "FAIL: reference produced no dump"; tail -5 /tmp/ling3vl-ref.log; exit 1; }

# 3. byte compare
if cmp -s "$PORT_DUMP" "$REF_DUMP"; then
    echo "MATCH  ling3vl synthetic mmproj embeddings (byte-exact, $(stat -c%s "$REF_DUMP") bytes)"
else
    echo "DIFF   ling3vl embeddings"
    cmp "$PORT_DUMP" "$REF_DUMP" | head -2
    exit 1
fi
