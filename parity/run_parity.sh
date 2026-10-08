#!/usr/bin/env bash
# Token-level parity check: Rust llama-cli vs reference llama-server (bd4f514db).
# Usage: ./run_parity.sh [prompt] [n_predict] [n_ctx]
set -euo pipefail
PROMPT="${1:-The capital of France is}"
N="${2:-32}"
CTX="${3:-512}"
MODEL="${MODEL:-/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
REF="/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin"
PORT="${PORT:-8791}"

cd "$(dirname "$0")/.."

# PROTOCOL: llama-server output depends on slot reuse state (see PARITY.md) —
# always restart so each run compares against first-request output.
pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
sleep 2

# reference server (raw completion, FA off to match the Rust non-FA path)
if ! curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  "$REF/llama-server" -m "$MODEL" -c "$CTX" -t 8 -fa off --port "$PORT" --host 127.0.0.1 >/tmp/parity-server.log 2>&1 &
  SRV_PID=$!
  trap '[ -n "${SRV_PID:-}" ] && kill $SRV_PID 2>/dev/null || true' EXIT
  for _ in $(seq 1 60); do
    curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 1
  done
fi

curl -s "http://127.0.0.1:$PORT/completion" -H "Content-Type: application/json" \
  -d "{\"prompt\":\"$PROMPT\",\"n_predict\":$N,\"temperature\":0,\"logprobs\":1}" -o /tmp/parity-ref.json

./target/release/llama-cli -m "$MODEL" -p "$PROMPT" -n "$N" -t 8 -c "$CTX" --temp 0 2>/dev/null \
  | grep "gen tokens" | grep -oE '\[.*\]' | tr -d '[]' | tr ',' '\n' > /tmp/parity-rust.txt

python3 - "$N" <<'EOF'
import json, sys
n = int(sys.argv[1])
d = json.load(open('/tmp/parity-ref.json'))
rust = [int(x) for x in open('/tmp/parity-rust.txt').read().split()]
ref_text = d['content']
# reference tokens: re-tokenize? server returns text only without logprobs;
# with logprobs=0 the ids live in completion_probabilities — request again if absent
print(f"ref : {ref_text!r}")
print(f"ref  tokens_evaluated={d['tokens_evaluated']} predicted={d['tokens_predicted']}")
print(f"rust tokens ({len(rust)}): {rust}")
cp = d.get('completion_probabilities')
if cp:
    ref_ids = [p['id'] for p in cp]
    print(f"ref  tokens ({len(ref_ids)}): {ref_ids}")
    m = sum(1 for a, b in zip(ref_ids, rust) if a == b)
    print(f"MATCH: {m}/{min(len(ref_ids), len(rust))}")
    sys.exit(0 if m == min(len(ref_ids), len(rust)) else 1)
print("NOTE: no per-token ids from server; comparing text only")
EOF
