#!/usr/bin/env bash
# SWA long-context reference capture (PARITY.md protocol): a *fresh* reference
# llama-server plus the first /completion request on it. The prompt is
# parity/swa_long_prompt.txt (> 1024 tokens, i.e. longer than gemma4's
# n_swa = 1024) and is sent verbatim, so the server tokenizes it with its own
# tokenizer; the response's tokens_evaluated is the tokenizer cross-check
# against the port's Vocab::tokenize.
#
#   parity/swa_long_capture.sh <model> <ctx> <port> <tag> <n_predict>
#
# Artifacts: /tmp/parity-<tag>-{server.log,ref.json,body.json}
set -euo pipefail

MODEL="${1:?model}"
CTX="${2:?n_ctx}"
PORT="${3:?port}"
TAG="${4:?tag}"
N="${5:-16}"
THREADS="${THREADS:-8}"
REF="${REF:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin}"
cd "$(dirname "$0")/.."

python3 - "$N" <<'PY' >"/tmp/parity-$TAG-body.json"
import json, sys
n = int(sys.argv[1])
text = open("parity/swa_long_prompt.txt", encoding="utf-8").read()
print(json.dumps({"prompt": text, "n_predict": n, "temperature": 0,
                  "logprobs": 20, "cache_prompt": False, "n_keep": 0}))
PY

# only ever kills a lingering server on *this* port (the script's own command
# line contains neither "llama-server" nor the port literal)
pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
sleep 2

"$REF/llama-server" -m "$MODEL" -c "$CTX" -t "$THREADS" \
  --port "$PORT" --host 127.0.0.1 >"/tmp/parity-$TAG-server.log" 2>&1 &
SRV_PID=$!
trap 'kill "$SRV_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 600); do
  curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && break
  sleep 1
done

curl -s "http://127.0.0.1:$PORT/completion" -H "Content-Type: application/json" \
  -d @"/tmp/parity-$TAG-body.json" -o "/tmp/parity-$TAG-ref.json"

kill "$SRV_PID" 2>/dev/null || true
wait "$SRV_PID" 2>/dev/null || true
trap - EXIT

python3 - "$TAG" <<'PY'
import json, sys
tag = sys.argv[1]
r = json.load(open(f"/tmp/parity-{tag}-ref.json"))
print(f"{tag}: tokens_evaluated={r.get('tokens_evaluated')} "
      f"tokens_predicted={r.get('tokens_predicted')} stop={r.get('stop')}")
print(f"{tag}: content={r.get('content','')!r}")
print(f"{tag}: tokens={[c['id'] for c in r.get('completion_probabilities', [])]}")
for i, c in enumerate(r.get("completion_probabilities", [])[:8]):
    print(f"  step {i}: top8 {[(p['id'], round(float(p['logprob']),4)) for p in c['top_logprobs'][:8]]}")
PY
grep -E "SWA KV cache|non-SWA KV cache|n_swa|kv_unified" "/tmp/parity-$TAG-server.log" | head -10