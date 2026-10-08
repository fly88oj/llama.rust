#!/usr/bin/env bash
# run_grammar_parity.sh — token-level GBNF parity: Rust llama-cli (--grammar-file)
# vs the reference llama-server /completion with the same grammar.
#
#   ./run_grammar_parity.sh [grammar.gbnf ...]
#
# PROTOCOL (see PARITY.md): the reference build's output depends on slot state,
# so a fresh server is started and only its *first* request is compared
# (llama-cli of this build hangs in a non-tty, so the embedded server is the
# reference sampling path — common_sampler_sample with grammar_first = false,
# tools/server/server-context.cpp:3856).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF=/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin
MODEL="${MODEL:-/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
PORT="${PORT:-8793}"
PROMPT="${PROMPT:-The capital of France is}"
N="${N:-32}"
CTX=512

GRAMMARS=("$@")
if [ ${#GRAMMARS[@]} -eq 0 ]; then
    GRAMMARS=("$ROOT/parity/grammars/json_simple.gbnf")
fi

pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
sleep 2
cd "$ROOT"

"$REF/llama-server" -m "$MODEL" -c "$CTX" -t 8 -fa off --port "$PORT" --host 127.0.0.1 \
    >/tmp/grammar-parity-server.log 2>&1 &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
    curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 1
done

fail=0
for G in "${GRAMMARS[@]}"; do
    echo "=== $G ==="
    python3 - "$G" "$PROMPT" "$N" <<'EOF'
import json, sys
gbnf = open(sys.argv[1]).read()
req = {"prompt": sys.argv[2], "n_predict": int(sys.argv[3]), "temperature": 0,
       "logprobs": 1, "grammar": gbnf, "cache_prompt": False}
open('/tmp/grammar-req.json', 'w').write(json.dumps(req))
EOF
    curl -s "http://127.0.0.1:$PORT/completion" -H "Content-Type: application/json" \
        -d @/tmp/grammar-req.json -o /tmp/grammar-ref.json

    ./target/release/llama-cli -m "$MODEL" -p "$PROMPT" -n "$N" -t 8 -c "$CTX" \
        --temp 0 -fa off --grammar-file "$G" 2>/dev/null \
        | grep "gen tokens" | grep -oE '\[.*\]' | tr -d '[]' | tr ',' '\n' > /tmp/grammar-rust.txt

    python3 - "$G" <<'EOF'
import json, sys
g = sys.argv[1]
d = json.load(open('/tmp/grammar-ref.json'))
cp = d.get('completion_probabilities') or []
ref_ids = [p['id'] for p in cp]
rust = [int(x) for x in open('/tmp/grammar-rust.txt').read().split()]
print(f"ref  text: {d['content']!r}")
print(f"ref  ids ({len(ref_ids)}): {ref_ids}")
print(f"mine ids ({len(rust)}): {rust}")
n = min(len(ref_ids), len(rust))
m = sum(1 for a, b in zip(ref_ids, rust) if a == b)
print(f"TOKEN MATCH: {m}/{n}")
# grammar conformance of the reference text
try:
    json.loads(d['content'])
    print("ref text parses as JSON: yes")
except Exception as e:
    print(f"ref text parses as JSON: no ({e})")
sys.exit(0 if m == n and n > 0 else 1)
EOF
    [ $? -eq 0 ] || fail=1
done

if [ $fail -eq 0 ]; then
    echo "GRAMMAR PARITY: OK"
else
    echo "GRAMMAR PARITY: MISMATCH"
fi
exit $fail