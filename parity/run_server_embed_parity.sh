#!/usr/bin/env bash
# llama-server embedding parity: decoder-model `-fe` embeddings
# (`cparams.embeddings`, llama-context.cpp:1729 output_all → build_pooling →
# the 1561-1598 extraction) — the reference server and the port's server answer
# the same /embedding and /v1/embeddings requests, and the embedding vectors
# are compared elementwise.
#
# protocol (PARITY.md): a *fresh* server and the *first* request on it, one
# pooling mode per server start (the pooling type is resolved at load).
#
# Artifacts: /tmp/srvembed-<tag>-{ref,rust}-pool<mode>.json + .log
set -euo pipefail

MODEL="${MODEL:-/home/jeffrey/localai/models/Qwen3-Embedding-0.6B-Q8_0.gguf}"
REF="${REF:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin}"
CTX="${CTX:-512}"
THREADS="${THREADS:-8}"
FA="${FA:-off}"
PROMPT="${PROMPT:-The capital of France is Paris.}"
POOLING="${POOLING:-mean none last cls}"   # space-separated modes to test
TAG="${TAG:-qwen3emb}"
PORT_REF="${PORT_REF:-8161}"
PORT_RUST="${PORT_RUST:-8162}"
OUT="${OUT:-/tmp/srvembed}"
mkdir -p "$OUT"
cd "$(dirname "$0")/.."

pkill -f "llama-server.*--port $PORT_REF" 2>/dev/null || true
pkill -f "llama-server.*--port $PORT_RUST" 2>/dev/null || true
sleep 1

start_ref() {
  "$REF/llama-server" -m "$MODEL" -c "$CTX" -t "$THREADS" -fa "$FA" --embeddings \
    --pooling "$1" --port "$PORT_REF" --host 127.0.0.1 >"$OUT/$TAG-ref-pool$1.log" 2>&1 &
  REF_PID=$!
}
start_rust() {
  ./target/release/llama-server -m "$MODEL" -c "$CTX" -t "$THREADS" -fa "$FA" --embeddings \
    --pooling "$1" --port "$PORT_RUST" --host 127.0.0.1 >"$OUT/$TAG-rust-pool$1.log" 2>&1 &
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
capture() {
  local port="$1" pool="$2" tag="$3"
  # legacy /embedding (TASK_RESPONSE_TYPE_NONE reports every row)
  curl -s -X POST "http://127.0.0.1:$port/embedding" -H 'Content-Type: application/json' \
       -d "{\"content\":\"$PROMPT\"}" -o "$OUT/$tag-pool$pool-embedding.json"
  # OAI /v1/embeddings (a single embedding object; rejected for pooling none)
  curl -s -X POST "http://127.0.0.1:$port/v1/embeddings" -H 'Content-Type: application/json' \
       -d "{\"input\":\"$PROMPT\"}" -o "$OUT/$tag-pool$pool-oai.json" \
       -w '%{http_code}' > "$OUT/$tag-pool$pool-oai.status"
}

overall=0
for pool in $POOLING; do
  echo "== pooling $pool"
  start_ref "$pool"
  if ! wait_ready "$PORT_REF"; then echo "ref server ($pool) failed to start"; overall=1; continue; fi
  capture "$PORT_REF" "$pool" "$TAG-ref"
  kill "$REF_PID" 2>/dev/null || true
  wait "$REF_PID" 2>/dev/null || true

  start_rust "$pool"
  if ! wait_ready "$PORT_RUST"; then echo "rust server ($pool) failed to start"; overall=1; continue; fi
  capture "$PORT_RUST" "$pool" "$TAG-rust"
  kill "$RUST_PID" 2>/dev/null || true
  wait "$RUST_PID" 2>/dev/null || true

  python3 parity/server_embed_parity_cmp.py "$TAG" "$OUT" "$pool" || overall=1
done

if [ "$overall" -eq 0 ]; then echo "EMBED PARITY: MATCH"; else echo "EMBED PARITY: MISMATCH"; exit 1; fi
