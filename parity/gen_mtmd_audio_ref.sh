#!/usr/bin/env bash
# gen_mtmd_audio_ref.sh — regenerate the mtmd-audio preprocessing ground truth
# (parity/mtmd_audio_ref.bin) from the pinned reference.
#
# The probe compiles tools/mtmd/mtmd-audio.cpp from the pinned tree directly
# and stubs clip_get_hparams, so every preprocessor runs with the reference's
# exact float math. Consumed by crates/llama/tests/mtmd_audio_parity.rs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN="${PIN:-/home/jeffrey/llm/llama.cpp-pinned}"
REFLIB="${REFLIB:-$PIN/../llama.cpp/build-rust-ref/bin}"

python3 "$ROOT/parity/gen_audio_fixture.py"

# the probe #includes mtmd-audio.cpp directly (single TU; no duplicate defs)
g++ -O2 -std=c++17 -o "$ROOT/parity/ref_mtmd_audio_dump" \
    "$ROOT/parity/ref_mtmd_audio_dump.cpp" \
    -I"$PIN/tools/mtmd" -I"$PIN/src" -I"$PIN/include" -I"$PIN/common" -I"$PIN/ggml/include" \
    -L"$REFLIB" -lggml-base -lggml -Wl,-rpath,"$REFLIB" 2>&1 | head -20

"$ROOT/parity/ref_mtmd_audio_dump" \
    "$ROOT/parity/mtmd-fixture-audio.f32" \
    "$ROOT/parity/mtmd_audio_ref.bin"

ls -la "$ROOT/parity/mtmd_audio_ref.bin"
