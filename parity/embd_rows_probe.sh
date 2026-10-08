#!/usr/bin/env bash
# Probe the per-token embedding rows for short prompts (decoder -fe path):
# for each content string, start fresh reference + port servers (--embeddings
# --pooling none) and compare every returned row bit-for-bit.
set -u
MODEL="${MODEL:-/home/jeffrey/localai/models/Qwen3-Embedding-0.6B-Q8_0.gguf}"
REF="${REF:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin}"
PORT="${PORT:-8865}"
cd "$(dirname "$0")/.."

stop() { pkill -f "llama-serve[r].*--port $PORT" >/dev/null 2>&1; sleep 1; }

for C in "$@"; do
  echo "### content=[$C]"
  python3 - "$C" > /tmp/e-body.json <<'EOF'
import json, sys
print(json.dumps({"content": sys.argv[1], "temperature": 0}))
EOF
  for SIDE in ref rust; do
    stop
    if [ "$SIDE" = ref ]; then BIN="$REF/llama-server"; else BIN="./target/release/llama-server"; fi
    "$BIN" -m "$MODEL" --embeddings --pooling none -c 512 -t 8 -fa ${FA:-off} \
        --port "$PORT" --host 127.0.0.1 >"/tmp/e-$SIDE.log" 2>&1 &
    for _ in $(seq 1 90); do
      curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
      sleep 1
    done
    curl -s -m 300 "http://127.0.0.1:$PORT/embedding" \
        -H 'Content-Type: application/json' -d @/tmp/e-body.json >"/tmp/e-$SIDE.json"
  done
  stop
  python3 - <<'EOF'
import json, struct
ref = json.load(open('/tmp/e-ref.json'))[0]['embedding']
rst = json.load(open('/tmp/e-rust.json'))[0]['embedding']
print('rows:', len(ref), len(rst))
for i in range(min(len(ref), len(rst))):
    bit = sum(1 for a, b in zip(ref[i], rst[i]) if struct.pack('<f', a) == struct.pack('<f', b))
    mx = max(abs(a - b) for a, b in zip(ref[i], rst[i]))
    print(f'row {i}: bit {bit}/{len(ref[i])} max|d| {mx:.3e}')
EOF
done