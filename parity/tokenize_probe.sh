#!/usr/bin/env bash
# Compare /tokenize between the reference and the port for the embedding
# probe strings (add_special on, exactly what /embedding does).
set -u
MODEL="${MODEL:-/home/jeffrey/localai/models/Qwen3-Embedding-0.6B-Q8_0.gguf}"
REF="${REF:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin}"
PORT="${PORT:-8866}"
cd "$(dirname "$0")/.."

stop() { pkill -f "llama-serve[r].*--port $PORT" >/dev/null 2>&1; sleep 1; }

for C in "$@"; do
  echo "### [$C]"
  python3 - "$C" > /tmp/t-body.json <<'EOF'
import json, sys
print(json.dumps({"content": sys.argv[1]}))
EOF
  for SIDE in ref rust; do
    stop
    if [ "$SIDE" = ref ]; then BIN="$REF/llama-server"; else BIN="./target/release/llama-server"; fi
    "$BIN" -m "$MODEL" -c 512 -t 8 --port "$PORT" --host 127.0.0.1 >"/tmp/t-$SIDE.log" 2>&1 &
    for _ in $(seq 1 90); do
      curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
      sleep 1
    done
    curl -s -m 300 "http://127.0.0.1:$PORT/tokenize" \
        -H 'Content-Type: application/json' -d @/tmp/t-body.json >"/tmp/t-$SIDE.json"
  done
  stop
  python3 - <<'EOF'
import json
a = json.load(open('/tmp/t-ref.json')).get('tokens')
b = json.load(open('/tmp/t-rust.json')).get('tokens')
print('ref :', a)
print('rust:', b)
print('SAME' if a == b else 'DIFFER')
EOF
done