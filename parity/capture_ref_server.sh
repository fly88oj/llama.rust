#!/usr/bin/env bash
# Reference-server response capture (used while porting llama-server; see
# PARITY.md's "llama-server" section for the resulting comparison).
set -euo pipefail
REF="${REF:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin}"
MODEL="${MODEL:-/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
PORT="${PORT:-8137}"
OUT="${OUT:-/tmp/refcap}"
mkdir -p "$OUT"

pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
sleep 1

"$REF/llama-server" -m "$MODEL" -c 512 -t 8 --port "$PORT" --host 127.0.0.1 \
  "${@}" >"$OUT/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
for _ in $(seq 1 300); do
  curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && break
  sleep 1
done

curl -s "http://127.0.0.1:$PORT/health" -o "$OUT/health.json"
curl -s "http://127.0.0.1:$PORT/props"  -o "$OUT/props.json"
curl -s -X POST "http://127.0.0.1:$PORT/completion" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","n_predict":16,"temperature":0,"cache_prompt":false}' \
  -o "$OUT/completion.json"
curl -s -X POST "http://127.0.0.1:$PORT/completion" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","n_predict":4,"temperature":0,"n_probs":3}' \
  -o "$OUT/completion_probs.json"
curl -s -N -X POST "http://127.0.0.1:$PORT/completion" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","n_predict":4,"temperature":0,"stream":true}' \
  -o "$OUT/completion_stream.txt"
curl -s -X POST "http://127.0.0.1:$PORT/tokenize" -H "Content-Type: application/json" \
  -d '{"content":"The capital of France is","add_special":true}' -o "$OUT/tokenize.json"
curl -s -X POST "http://127.0.0.1:$PORT/detokenize" -H "Content-Type: application/json" \
  -d '{"tokens":[785,6722,315,9625,374]}' -o "$OUT/detokenize.json"
curl -s -X POST "http://127.0.0.1:$PORT/completion" -H "Content-Type: application/json" \
  -d '{"prompt":' -o "$OUT/badjson.json" -w '%{http_code}\n' > "$OUT/badjson.status"
curl -s "http://127.0.0.1:$PORT/completion" -H "Content-Type: application/json" \
  -o "$OUT/badmethod.json" -w '%{http_code}\n' > "$OUT/badmethod.status"
curl -s -X POST "http://127.0.0.1:$PORT/nope" -H "Content-Type: application/json" -d '{}' \
  -o "$OUT/notfound.json" -w '%{http_code}\n' > "$OUT/notfound.status"
echo "captured into $OUT"
kill $SRV 2>/dev/null || true
wait $SRV 2>/dev/null || true
trap - EXIT