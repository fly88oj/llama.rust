#!/usr/bin/env bash
# gen_sampler_dry_ref.sh — regenerate the DRY / adaptive-p / infill sampler
# parity fixture (parity/sampler_dry_ref.txt) from the pinned reference build.
#
# Requires the pinned reference build:
#   /home/jeffrey/llm/llama.cpp-pinned            (source, bd4f514db1)
#   /home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin (libllama.so, libllama-common.so)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN=/home/jeffrey/llm/llama.cpp-pinned
REF=/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin
VOCAB="$PIN/models/ggml-vocab-qwen2.gguf"

OUT="$ROOT/parity/sampler_dry_ref"
g++ -O2 -std=c++17 -o "$OUT" "$ROOT/parity/sampler_dry_ref.cpp" \
    -I"$PIN/src" -I"$PIN/include" -I"$PIN/common" -I"$PIN/ggml/include" \
    -L"$REF" -lllama -lllama-common -Wl,-rpath,"$REF"

export LLAMA_LOG_LEVEL=error
"$OUT" "$VOCAB" > "$ROOT/parity/sampler_dry_ref.txt" 2>/dev/null

echo "wrote $ROOT/parity/sampler_dry_ref.txt ($(wc -l < "$ROOT/parity/sampler_dry_ref.txt") lines)"
