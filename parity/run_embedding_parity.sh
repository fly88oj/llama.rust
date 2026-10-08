#!/usr/bin/env bash
# Decoder-model embeddings parity (`-fe`): the reference server and the port's
# server answer identical /embedding requests on a decoder (qwen3) embedding
# model and on a plain decoder model. Fresh servers + first request per case.
#
# usage: bash parity/run_embedding_parity.sh [model] [pooling]
set -euo pipefail
MODEL="${1:-/home/jeffrey/localai/models/Qwen3-Embedding-0.6B-Q8_0.gguf}"
POOL="${2:-mean}"
REF="${REF:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin}"
PORT="${PORT:-8861}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRV="$ROOT/target/release/llama-server"

BODY='{"content":"The capital of France is Paris.","temperature":0}'

run_one() {
  local bin="$1" out="$2" tag="$3"
  "$bin" -m "$MODEL" --embeddings --pooling "$POOL" -c 512 -t 8 -fa off \
      --port "$PORT" --host 127.0.0.1 >"/tmp/embd-$tag.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 90); do
    curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 1
  done
  curl -s -m 300 "http://127.0.0.1:$PORT/embedding" \
      -H 'Content-Type: application/json' -d "$BODY" >"$out"
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

bash "$ROOT/parity/kill_stale_servers.sh" >/dev/null 2>&1 || true
run_one "$REF/llama-server" /tmp/embd-ref.json ref
run_one "$SRV" /tmp/embd-rust.json rust

python3 - <<'EOF'
import json
ref = json.load(open('/tmp/embd-ref.json'))
rst = json.load(open('/tmp/embd-rust.json'))
def emb(d):
    # non-OAI /embedding answers a top-level list; /v1/embeddings a dict
    if isinstance(d, list):
        d = d[0] if d else {}
    e = d.get('embedding') or (d.get('data') or [{}])[0].get('embedding')
    # pooled rows come nested ([[...]]); take the first row
    if e and isinstance(e[0], list):
        e = e[0]
    return e
er, es = emb(ref), emb(rst)
if er is None or es is None or len(er) != len(es):
    print('shape/keys differ:', {k: ref.get(k) is not None for k in ('embedding','data','error')},
          {k: rst.get(k) is not None for k in ('embedding','data','error')})
    print('ref keys:', list(ref)[:6], 'rust keys:', list(rst)[:6])
    raise SystemExit(1)
same = sum(1 for a, b in zip(er, es) if a == b)
import struct
bit = sum(1 for a, b in zip(er, es) if struct.pack('<f', a) == struct.pack('<f', b))
mx = max(abs(a - b) for a, b in zip(er, es))
import math
norm = math.sqrt(sum(a * a for a in er))
print(f"n={len(er)} equal={same} bit-identical={bit} max|d|={mx:.3e} ref-l2={norm:.4f}")
print('RESULT:', 'MATCH' if bit == len(er) else ('BAND' if mx / max(norm, 1e-9) < 1e-4 else 'DIFFER'))
EOF