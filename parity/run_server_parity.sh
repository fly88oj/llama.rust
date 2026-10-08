#!/usr/bin/env bash
# llama-server parity: the reference server and the port's server answer the
# same request bodies on the same model; the responses are compared field by
# field (except timings) and the generated tokens must match.
#
# protocol (PARITY.md): a *fresh* server and the *first* request on it,
# temperature 0, cache_prompt=false, -fa as given.
#
# Artifacts: /tmp/srvparity-<tag>-{ref,rust}.json + .log and the SSE captures.
set -euo pipefail

MODEL="${MODEL:-/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
REF="${REF:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin}"
CTX="${CTX:-512}"
THREADS="${THREADS:-8}"
NP="${NP:-4}"
FA="${FA:-off}"
PROMPT="${PROMPT:-The capital of France is}"
N="${N:-16}"
TAG="${TAG:-qwen25}"
PORT_REF="${PORT_REF:-8141}"
PORT_RUST="${PORT_RUST:-8142}"
OUT="${OUT:-/tmp/srvparity}"
mkdir -p "$OUT"
cd "$(dirname "$0")/.."

pkill -f "llama-server.*--port $PORT_REF" 2>/dev/null || true
pkill -f "llama-server.*--port $PORT_RUST" 2>/dev/null || true
sleep 1

start_ref() {
  "$REF/llama-server" -m "$MODEL" -c "$CTX" -t "$THREADS" -np "$NP" -fa "$FA" \
    --port "$PORT_REF" --host 127.0.0.1 >"$OUT/$TAG-ref.log" 2>&1 &
  REF_PID=$!
}
start_rust() {
  ./target/release/llama-server -m "$MODEL" -c "$CTX" -t "$THREADS" -np "$NP" -fa "$FA" \
    --port "$PORT_RUST" --host 127.0.0.1 >"$OUT/$TAG-rust.log" 2>&1 &
  RUST_PID=$!
}
wait_ready() {
  local port="$1"
  for _ in $(seq 1 300); do
    curl -s "http://127.0.0.1:$port/health" 2>/dev/null | grep -q '"ok"' && return 0
    sleep 1
  done
  return 1
}

# body: prompt / n_predict / temperature 0 / no prompt caching
BODY=$(python3 - "$PROMPT" "$N" <<'PY'
import json, sys
print(json.dumps({"prompt": sys.argv[1], "n_predict": int(sys.argv[2]), "temperature": 0.0,
                  "cache_prompt": False}))
PY
)

capture() {
  local port="$1" tag="$2"
  curl -s "http://127.0.0.1:$port/health"                     -o "$OUT/$tag-health.json"
  curl -s "http://127.0.0.1:$port/props"                      -o "$OUT/$tag-props.json"
  curl -s -X POST "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' \
       -d "$BODY"                                             -o "$OUT/$tag-completion.json"
  curl -s -X POST "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' \
       -d "{\"prompt\":\"$PROMPT\",\"n_predict\":4,\"temperature\":0,\"n_probs\":3,\"cache_prompt\":false}" \
       -o "$OUT/$tag-probs.json"
  curl -s -X POST "http://127.0.0.1:$port/tokenize" -H 'Content-Type: application/json' \
       -d "{\"content\":\"$PROMPT\",\"add_special\":true}"     -o "$OUT/$tag-tokenize.json"
  curl -s -X POST "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' \
       -d '{"prompt":' -o "$OUT/$tag-badjson.json" -w '%{http_code}' > "$OUT/$tag-badjson.status"
  curl -s -X POST "http://127.0.0.1:$port/nope" -H 'Content-Type: application/json' -d '{}' \
       -o "$OUT/$tag-notfound.json" -w '%{http_code}' > "$OUT/$tag-notfound.status"
  curl -s -N -X POST "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' \
       -d "{\"prompt\":\"$PROMPT\",\"n_predict\":4,\"temperature\":0,\"stream\":true,\"cache_prompt\":false}" \
       -o "$OUT/$tag-stream.txt"
}

start_ref
wait_ready "$PORT_REF"
capture "$PORT_REF" "$TAG-ref"
kill "$REF_PID" 2>/dev/null || true
wait "$REF_PID" 2>/dev/null || true

start_rust
wait_ready "$PORT_RUST"
capture "$PORT_RUST" "$TAG-rust"
kill "$RUST_PID" 2>/dev/null || true
wait "$RUST_PID" 2>/dev/null || true

python3 parity/server_parity_cmp.py "$TAG" "$OUT"