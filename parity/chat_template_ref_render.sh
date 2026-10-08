#!/usr/bin/env bash
# Reference-only e2e capture: start the pinned reference llama-server on each
# newly-unblocked local model, dump /props' chat_template and /apply-template
# outputs for the three probe bodies, for comparison against the port engine.
set -uo pipefail
REF="${REF:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin}"
OUT="${OUT:-/tmp/mj-e2e}"
mkdir -p "$OUT"
port="${PORT:-8175}"
pkill -f "port $port" 2>/dev/null || true; sleep 1

TOOLS='[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city"]}}}]'

declare -A MODELS
MODELS[gptoss]="/home/jeffrey/.lmstudio/models/lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf"
MODELS[gemma4]="/home/jeffrey/.lmstudio/models/lmstudio-community/gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf"
MODELS[qwen35]="/home/jeffrey/.lmstudio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

for tag in "$@"; do
  m="${MODELS[$tag]}"
  export LD_LIBRARY_PATH="$REF${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  nohup "$REF/llama-server" -m "$m" -c 512 -t 8 --port $port --host 127.0.0.1 \
    >"$OUT/$tag-refonly.log" 2>&1 &
  ok=""
  for _ in $(seq 1 240); do
    curl -s "http://127.0.0.1:$port/health" 2>/dev/null | grep -q '"ok"' && ok=1 && break
    sleep 1
  done
  if [ -z "$ok" ]; then echo "[$tag] ref server failed"; pkill -f "port $port"; continue; fi
  curl -s "http://127.0.0.1:$port/props" | python3 -c "import json,sys; d=json.load(sys.stdin); open('$OUT/$tag.template.jinja','w').write(d.get('chat_template') or ''); print('[$tag] template bytes:', len(d.get('chat_template') or ''))"
  i=0
  i=$((i+1)); curl -s -X POST "http://127.0.0.1:$port/apply-template" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"system","content":"You are terse."},{"role":"user","content":"What is the weather in Tokyo?"}]}' -o "$OUT/$tag-$i-ref.json"
  i=$((i+1)); curl -s -X POST "http://127.0.0.1:$port/apply-template" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"system","content":"You are terse."},{"role":"user","content":"What is the weather in Tokyo? Use the tool."}],"tools":'"$TOOLS"'}' -o "$OUT/$tag-$i-ref.json"
  i=$((i+1)); curl -s -X POST "http://127.0.0.1:$port/apply-template" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Weather in Tokyo?"},{"role":"assistant","content":"","tool_calls":[{"id":"call1","type":"function","function":{"name":"get_weather","arguments":{"city":"Tokyo"}}}]},{"role":"tool","content":"15C sunny","tool_call_id":"call1"},{"role":"user","content":"thanks"}],"tools":'"$TOOLS"'}' -o "$OUT/$tag-$i-ref.json"
  echo "[$tag] captured 3 prompts"
  pkill -f "port $port" 2>/dev/null || true; sleep 1
done
