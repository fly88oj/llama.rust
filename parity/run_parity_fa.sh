#!/usr/bin/env bash
# Token-level parity check with flash-attn ON on both sides, so the Rust side
# exercises the tiled FA kernel for prefill (T >= 64).
#
# PROTOCOL (see PARITY.md): a fresh reference server + the first request on it,
# because slot reuse moves near-tied tokens.
#
# Usage: bash parity/run_parity_fa.sh [prompt] [n_predict] [n_ctx] [model]
set -euo pipefail
PROMPT="${1:-The capital of France is}"
N="${2:-32}"
CTX="${3:-512}"
MODEL="${4:-${MODEL:-/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}}"
REF="${REF:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin}"
PORT="${PORT:-8795}"

cd "$(dirname "$0")/.."

pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
sleep 2

"$REF/llama-server" -m "$MODEL" -c "$CTX" -t 8 -fa on --port "$PORT" --host 127.0.0.1 >/tmp/parity-fa-server.log 2>&1 &
SRV_PID=$!
trap '[ -n "${SRV_PID:-}" ] && kill $SRV_PID 2>/dev/null || true' EXIT
# wait for the *ready* health body — the listener answers 503 "Loading model"
# before the model finishes loading (run_server_parity.sh's convention)
for _ in $(seq 1 120); do
  curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && break
  sleep 1
done

curl -s "http://127.0.0.1:$PORT/completion" -H "Content-Type: application/json" \
  -d "{\"prompt\":\"$PROMPT\",\"n_predict\":$N,\"temperature\":0,\"logprobs\":1}" -o /tmp/parity-fa-ref.json

./target/release/llama-cli -m "$MODEL" -p "$PROMPT" -n "$N" -t 8 -c "$CTX" --temp 0 -fa on 2>/dev/null \
  | grep "gen tokens" | grep -oE '\[.*\]' | tr -d '[]' | tr ',' '\n' > /tmp/parity-fa-rust.txt

python3 - "$N" <<'EOF'
import json, sys
n = int(sys.argv[1])
d = json.load(open('/tmp/parity-fa-ref.json'))
rust = [int(x) for x in open('/tmp/parity-fa-rust.txt').read().split()]
print(f"ref  text: {d['content']!r}")
print(f"ref  tokens_evaluated={d['tokens_evaluated']} predicted={d['tokens_predicted']}")
print(f"rust tokens ({len(rust)}): {rust}")
cp = d.get('completion_probabilities')
if cp:
    ref_ids = [p['id'] for p in cp]
    print(f"ref  tokens ({len(ref_ids)}): {ref_ids}")
    m = sum(1 for a, b in zip(ref_ids, rust) if a == b)
    first = next((i for i, (a, b) in enumerate(zip(ref_ids, rust)) if a != b), None)
    print(f"MATCH: {m}/{min(len(ref_ids), len(rust))}  first_diff={first}")
    sys.exit(0 if m == min(len(ref_ids), len(rust)) else 1)
print("NOTE: no per-token ids from server; comparing text only")
EOF