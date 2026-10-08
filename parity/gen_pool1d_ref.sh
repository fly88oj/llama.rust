#!/usr/bin/env bash
# gen_pool1d_ref.sh — regenerate the GGML_OP_POOL_1D ground truth
# (parity/pool1d_ref.bin) from the pinned reference build.
#
# The probe drives ggml_pool_1d (ggml.c:5071, kernel ops.cpp:7690-7754)
# through the real graph-compute path over a (k,s,p) x width x op sweep
# (plus F16 sources and the old pool_2d(k0=2,k1=1,s0=2,s1=1) composition the
# whisper-enc round used). Consumed by crates/ggml
# compute::tests::pool_1d_bit_exact_vs_reference.
# llama-rust-data-home: root of the local model/reference corpus (default: $HOME)
DATA_HOME="${LLAMA_RUST_DATA_HOME:-$HOME}"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN=${DATA_HOME}/llm/llama.cpp-pinned
REF=${DATA_HOME}/llm/llama.cpp/build-rust-ref

gcc -O2 -o "$ROOT/parity/ref_pool1d_dump" "$ROOT/parity/ref_pool1d_dump.c" \
    -I"$PIN/ggml/include" \
    -L"$REF/bin" -Wl,-rpath,"$REF/bin" \
    -lggml-cpu -lggml-base -lggml -lm

"$ROOT/parity/ref_pool1d_dump" "$ROOT/parity/pool1d_ref.bin"
