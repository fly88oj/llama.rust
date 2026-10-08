#!/usr/bin/env bash
# server_gpu_anchor.sh — the llama-server GPU surface's token anchors (batch
# 20): a FRESH port llama-server's FIRST /completion (temperature 0,
# cache_prompt false, return_tokens — the anchor_newref.sh convention) on
# qwen2.5-0.5b, compared across the engine choices:
#
#   * server-CPU      (the port's own CPU engine, no GPU flags)
#   * server-VK ngl99 (--device Vulkan0 -ngl 99 — every layer on the iGPU)
#   * server-VK ngl12 (a partial offload, the layer-split rule exercised)
#
# All three must reproduce the canonical qwen2.5 16-token anchor
# [12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 2086, 7772,
#  304, 279, 1879] (parity/anchors/qwen25.ref.tokens — the NEW reference
# fresh-server first request).
#
# usage: bash parity/server_gpu_anchor.sh
set -uo pipefail
cd "$(dirname "$0")/.."

PORT_BIN=./target/release/llama-server
VK_LIBS=/home/jeffrey/llm/build-rust-vk/bin
QWEN25=/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf
OUT=parity/gpu-batch20
TMP=/tmp/svg2-gpu20
mkdir -p "$OUT" "$TMP"

ANCHOR="12095 13 1084 374 279 7772 3283 304 4505 323 279 2086 7772 304 279 1879"
PROMPT="The capital of France is"
BASE_PORT=${BASE_PORT:-8881}
fail=0

tokens_of() { python3 -c "import json;d=json.load(open('$1'));print(' '.join(map(str,(d.get('tokens') or [])[:16])))"; }

# run_server <name> <port> <extra flags...>
run_server() {
    local name=$1 port=$2; shift 2
    free -g | head -2
    bash parity/limited.sh -m 30G -h 26G -- "$PORT_BIN" -m "$QWEN25" -c 512 -t 8 \
        --host 127.0.0.1 --port "$port" "$@" > "$OUT/$name.log" 2>&1 &
    local pid=$!
    local ok=""
    for _ in $(seq 1 300); do
        curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { ok=1; break; }
        kill -0 $pid 2>/dev/null || break
        sleep 1
    done
    if [ -z "$ok" ]; then
        echo "$name: SERVER FAILED (see $OUT/$name.log)" | tee -a "$OUT/errors"
        kill $pid 2>/dev/null; pkill -P $pid 2>/dev/null; wait $pid 2>/dev/null
        return 1
    fi
    curl -s -X POST "http://127.0.0.1:$port/completion" \
        -H 'Content-Type: application/json' \
        -d "{\"prompt\":\"$PROMPT\",\"n_predict\":16,\"temperature\":0,\"cache_prompt\":false,\"return_tokens\":true}" \
        -o "$OUT/$name.json"
    kill $pid 2>/dev/null; pkill -P $pid 2>/dev/null; wait $pid 2>/dev/null
    return 0
}

check() { # name
    local name=$1
    local got; got=$(tokens_of "$OUT/$name.json")
    if [ "$got" = "$ANCHOR" ]; then
        echo "$name: 16/16 anchor tokens IDENTICAL [$got ...]"
    else
        echo "$name: ANCHOR MISMATCH"
        echo "  want: $ANCHOR"
        echo "  got : $got"
        fail=1
    fi
}

[ -f "$QWEN25" ] || { echo "missing model $QWEN25"; exit 1; }

echo "== server-CPU (the port's own engine) =="
run_server cpu "$BASE_PORT" && check cpu
BASE_PORT=$((BASE_PORT + 1))

echo "== server-VK ngl99 (full offload) =="
run_server vk99 "$BASE_PORT" --ggml-libs "$VK_LIBS" --device Vulkan0 -ngl 99 && check vk99
BASE_PORT=$((BASE_PORT + 1))

echo "== server-VK ngl12 (partial offload) =="
run_server vk12 "$BASE_PORT" --ggml-libs "$VK_LIBS" --device Vulkan0 -ngl 12 && check vk12

# the offload banner evidence (llama_log_info of enable_gpu)
grep -h "enable_gpu: device" "$OUT"/vk99.log "$OUT"/vk12.log | sed 's/^/  /'

echo
if [ $fail -eq 0 ]; then
    echo "SERVER GPU ANCHORS: ALL PASS"
else
    echo "SERVER GPU ANCHORS: FAILURES (see above + $OUT/errors)"
fi
exit $fail
