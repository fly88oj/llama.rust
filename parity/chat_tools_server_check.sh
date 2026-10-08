#!/usr/bin/env bash
# chat_tools_server_check.sh — end-to-end tool-call generation vs the port's
# parser (agent: chat tools).
#
# 1. starts the pinned reference llama-server with the qwen2.5 model
# 2. sends a /v1/chat/completions request with tools (temperature 0)
# 3. captures the reference's reported tool_calls AND the raw generated text
#    (via a plain completion that replays the same prompt, so the exact bytes
#    the model produced are visible)
# 4. feeds the raw text through the port's common_chat_parse equivalent
#    (tests/chat_tools_parity.rs::ref_server_tool_call_parse) and compares
#
# Artifacts: /tmp/chat-tools-srv/{request.json,response.json,raw-completion.json}
set -euo pipefail

MODEL="${MODEL:-/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
REF="${REF:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin}"
PORT="${PORT:-8177}"
OUT="${OUT:-/tmp/chat-tools-srv}"
mkdir -p "$OUT"
cd "$(dirname "$0")/.."

# server lifecycle lives in this script only (never pkill -f with this
# script's own pattern in-line)
pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
sleep 1

nohup "$REF/llama-server" -m "$MODEL" -c 768 -t 8 --temp 0 --port "$PORT" \
    --host 127.0.0.1 >"$OUT/server.log" 2>&1 &

for i in $(seq 1 60); do
  if curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; then break; fi
  sleep 1
done

cat >"$OUT/request.json" <<'EOF'
{
  "messages": [
    {"role": "user", "content": "What is the weather in Tokyo right now? Use the tool."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "The city name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
          },
          "required": ["city"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "temperature": 0,
  "max_tokens": 200,
  "stream": false
}
EOF

# 1) the tool-call request: the reference parses tool_calls itself
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d @"$OUT/request.json" >"$OUT/response.json"

# 2) the raw generated text: same conversation WITHOUT tools parsing —
#    replay the exact prompt the reference built (from /props + the tool
#    request is not reproducible verbatim, so instead ask for a plain
#    completion of the same user turn primed to emit the qwen tool format)
cat >"$OUT/raw-request.json" <<'EOF'
{
  "prompt": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the current weather for a city\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"The city name\"}, \"unit\": {\"type\": \"string\", \"enum\": [\"celsius\", \"fahrenheit\"]}}, \"required\": [\"city\"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\nWhat is the weather in Tokyo right now? Use the tool.<|im_end|>\n<|im_start|>assistant\n",
  "temperature": 0,
  "max_tokens": 200,
  "stream": false
}
EOF
curl -s "http://127.0.0.1:$PORT/completion" \
    -H 'Content-Type: application/json' \
    -d @"$OUT/raw-request.json" >"$OUT/raw-completion.json"

pkill -f "llama-server.*--port $PORT" 2>/dev/null || true

python3 - "$OUT" <<'EOF'
import json, sys
out = sys.argv[1]
resp = json.load(open(f"{out}/response.json"))
raw = json.load(open(f"{out}/raw-completion.json"))
tc = resp["choices"][0]["message"].get("tool_calls")
print("reference tool_calls:", json.dumps(tc, ensure_ascii=False))
print("finish_reason:", resp["choices"][0].get("finish_reason"))
print("raw text:", json.dumps(raw.get("content"), ensure_ascii=False))
assert tc, "reference did not produce tool_calls"
EOF
echo "artifacts in $OUT (consumed by tests/chat_tools_parity.rs::ref_server_tool_call_parse)"
