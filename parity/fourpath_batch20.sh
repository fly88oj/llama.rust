#!/usr/bin/env bash
# fourpath_batch20.sh — the four-engine token-identity regression on
# qwen2.5-0.5b (batch 20 re-run of the batch-18 four-path record): the same
# 16-token greedy stream through
#
#   1. the port's own CPU engine   (no flags)
#   2. the foreign CPU backend     (--ggml-libs <ref-cpu>/bin --foreign-cpu)
#   3. Vulkan partial offload      (-ngl 12)
#   4. Vulkan full offload         (-ngl 99)
#
# all four must equal the canonical anchor [12095, 13, 1084, 374, 279, 7772,
# 3283, 304, 4505, 323, 279, 2086, 7772, 304, 279, 1879].
#
# usage: bash parity/fourpath_batch20.sh
set -uo pipefail
cd "$(dirname "$0")/.."

PORT_CLI=./target/release/llama-cli
REF_CPU=/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin
VK_LIBS=/home/jeffrey/llm/build-rust-vk/bin
QWEN25=/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf
OUT=parity/gpu-batch20
mkdir -p "$OUT"

ANCHOR="12095 13 1084 374 279 7772 3283 304 4505 323 279 2086 7772 304 279 1879"
fail=0

port_tokens_of() { sed -n 's/^gen tokens: \[\(.*\)\]$/\1/p' "$1" | head -1 | tr -d ' ' | tr ',' '\n' | head -16 | tr '\n' ' ' | sed 's/ $//'; }

run() { # name extra...
    local name=$1; shift
    free -g | head -2
    bash parity/limited.sh -m 30G -h 26G -- "$PORT_CLI" -m "$QWEN25" -p "The capital of France is" \
        -n 16 -t 8 -c 512 -fa off --temp 0.0 "$@" > "$OUT/four-$name.txt" 2>"$OUT/four-$name.err" \
        || { echo "$name: RUN FAILED (see $OUT/four-$name.err)"; fail=1; return; }
    local got; got=$(port_tokens_of "$OUT/four-$name.txt")
    if [ "$got" = "$ANCHOR" ]; then
        echo "$name: 16/16 IDENTICAL [$got ...]"
    else
        echo "$name: MISMATCH — want [$ANCHOR] got [$got]"
        fail=1
    fi
}

run cpu
run foreign-cpu --ggml-libs "$REF_CPU" --foreign-cpu
run vk12 --ggml-libs "$VK_LIBS" --device Vulkan0 -ngl 12
run vk99 --ggml-libs "$VK_LIBS" --device Vulkan0 -ngl 99

grep -h "enable_gpu: device" "$OUT"/four-vk12.err "$OUT"/four-vk99.err | sed 's/^/  /'

echo
if [ $fail -eq 0 ]; then
    echo "FOUR-PATH IDENTITY: ALL PASS"
else
    echo "FOUR-PATH IDENTITY: FAILURES"
fi
exit $fail
