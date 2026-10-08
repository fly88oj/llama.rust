#!/usr/bin/env bash
# gen_model_saver_ref.sh — produce the pinned reference's own
# `llama_model_save_to_file` output (llama.cpp:498-503 -> llama-model-saver.cpp)
# for the byte-comparison of the port's saver (crates/llama/src/saver.rs), and
# run that comparison.
#
# The pinned build ships no tool that calls the saver (finetune is not built),
# so parity/ref_model_saver.c drives libllama.so directly. The probe loads
# with `use_extra_bufts = false`: the reference's CPU_REPACK extra buffer type
# repacks some weights into converted buffers at load time, which both changes
# the bytes a save would write and crashes the reference's write path — the
# comparison pins the plain mmap'd weights, the same storage the port saves.
#
# Round-trip generation check: the reference llama-cli loads the port's saved
# file and generates 16 greedy tokens, compared with the same run on the
# original file (both fresh processes).
#
# usage: bash parity/gen_model_saver_ref.sh [model]
set -euo pipefail

MODEL="${1:-/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
REF="${REF:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin}"
PIN="${PIN:-/home/jeffrey/llm/llama.cpp-pinned}"
OUT_REF=/tmp/ref_model_saver_out.gguf
OUT_RUST=/tmp/rust_model_saver_out.gguf
cd "$(dirname "$0")/.."

echo "== building the C probe"
gcc -O2 parity/ref_model_saver.c -o /tmp/ref_model_saver \
    -I"$PIN/include" -I"$PIN/ggml/include" -L"$REF" -lllama -lggml -lggml-base

echo "== reference save"
LD_LIBRARY_PATH="$REF" /tmp/ref_model_saver "$MODEL" "$OUT_REF" 2>/dev/null

echo "== port save (via the byte-comparison test)"
LLAMA_SAVER_REF="$OUT_REF" cargo test --release -p llama --test saver_e2e -- --nocapture

echo
echo "== round trip: the reference generates identically from the port's save"
# the port's saved file (the test artifact)
cp "$(ls -t /tmp/llama_rust_saver_qwen25*.gguf | head -1)" "$OUT_RUST"

gen_via_server() {
    local model="$1" port="$2"
    LD_LIBRARY_PATH="$REF" "$REF/llama-server" -m "$model" -c 512 -t 8 \
        --port "$port" --host 127.0.0.1 >"/tmp/saver_srv_$port.log" 2>&1 &
    local pid=$!
    for _ in $(seq 1 120); do
        curl -s "http://127.0.0.1:$port/health" 2>/dev/null | grep -q '"ok"' && break
        sleep 1
    done
    curl -s "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' \
        -d '{"prompt":"The capital of France is","n_predict":16,"temperature":0,"logprobs":1}' \
        > "/tmp/saver_gen_$port.json"
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}
gen_via_server "$MODEL" 8871
gen_via_server "$OUT_RUST" 8872

python3 - <<'EOF2'
import json, sys
a = json.load(open('/tmp/saver_gen_8871.json'))
b = json.load(open('/tmp/saver_gen_8872.json'))
ia = [p['id'] for p in a.get('completion_probabilities', [])]
ib = [p['id'] for p in b.get('completion_probabilities', [])]
print("orig tokens:", ia)
print("save tokens:", ib)
ok = ia == ib and len(ia) == 16
print(f"ROUND-TRIP GENERATION: {'identical 16/16' if ok else 'DIFFERS'}")
print("text:", repr(a.get('content')))
sys.exit(0 if ok else 1)
EOF2
