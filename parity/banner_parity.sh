#!/usr/bin/env bash
# banner_parity.sh — the model-load banner (display family) of the port vs the
# pinned reference, byte-for-byte on the payload lines.
#
# Reference capture: parity/ref_model_saver.c's probe (a default-params
# llama_model_load_from_file — exactly what the port's load path mirrors).
# The reference's own tools gate LLAMA_LOG_INFO behind common/log verbosity
# (common.cpp:1326 llama_log_set(..., ERROR) unless -lv is raised), so the
# probe is the apples-to-apples driver; with `llama-server -lv 6` the same
# print_info payload lines appear behind common/log's timestamp prefixes.
#
# The port side: ./target/release/llama-cli's model-load banner (routed
# through llama::impl_log — the default callback writes stderr like the C).
#
# usage: bash parity/banner_parity.sh [model]
set -euo pipefail

MODEL="${1:-/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
REF="${REF:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin}"
PIN="${PIN:-/home/jeffrey/llm/llama.cpp-pinned}"
cd "$(dirname "$0")/.."

echo "== reference banner (default-params load via the C probe)"
gcc -O2 parity/ref_model_saver.c -o /tmp/ref_model_saver \
    -I"$PIN/include" -I"$PIN/ggml/include" -L"$REF" -lllama -lggml -lggml-base
LD_LIBRARY_PATH="$REF" /tmp/ref_model_saver "$MODEL" /tmp/banner_probe_out.gguf 2>&1 \
    | grep -E "^print_info:" | sed 's/^print_info: //' > /tmp/banner_ref.txt || true
rm -f /tmp/banner_probe_out.gguf

echo "== port banner"
./target/release/llama-cli -m "$MODEL" -p hi -n 1 -t 8 -c 512 --temp 0 \
    2>&1 >/dev/null | grep -E "^print_info:" | sed 's/^print_info: //' > /tmp/banner_rust.txt

echo "== diff"
# the one documented divergence: EOT id selection (vocab.rs:863 — the C scans
# token_to_id unordered_map order, the port scans ids ascending; on this
# build the reference picks <|im_end|> 151645, the port <|endoftext|> 151643)
diff -u /tmp/banner_ref.txt /tmp/banner_rust.txt > /tmp/banner_diff.txt || true
n=$(grep -cE '^[+-][^+-]' /tmp/banner_diff.txt || true)
if [ "$n" -eq 0 ]; then
    echo "banner: identical ($(wc -l < /tmp/banner_ref.txt) lines)"
elif [ "$n" -eq 2 ] && grep -q 'EOT token' /tmp/banner_diff.txt; then
    echo "banner: identical except the documented EOT-id divergence ($(wc -l < /tmp/banner_ref.txt) lines)"
else
    echo "banner: UNEXPECTED differences (see /tmp/banner_diff.txt)"
    exit 1
fi
