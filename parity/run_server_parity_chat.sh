#!/usr/bin/env bash
# llama-server parity — the OpenAI-compatible endpoints (`/v1/chat/completions`
# with tools/tool_calls, `/v1/completions`, `/models`, `/slots`, the rerank
# 501s) plus the embeddings endpoints on bge-m3. Same protocol as
# run_server_parity.sh: fresh servers, the same request bodies, field-by-field
# comparison of the responses and the SSE frames (created/id/timings/
# system_fingerprint — and the random tool-call ids — differ by design).
#
# Artifacts: /tmp/srvparity-chat/<tag>-{ref,rust}.json and the .stream captures.
set -euo pipefail

MODEL="${MODEL:-/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
EMBD_MODEL="${EMBD_MODEL:-/home/jeffrey/localai/models/bge-m3-Q8_0.gguf}"
REF="${REF:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin}"
CTX="${CTX:-512}"
THREADS="${THREADS:-8}"
NP="${NP:-4}"
FA="${FA:-off}"
TAG="${TAG:-chat}"
PORT_REF="${PORT_REF:-8151}"
PORT_RUST="${PORT_RUST:-8152}"
EPORT_REF="${EPORT_REF:-8153}"
EPORT_RUST="${EPORT_RUST:-8154}"
OUT="${OUT:-/tmp/srvparity-chat}"
mkdir -p "$OUT"
cd "$(dirname "$0")/.."

kill_ports() {
  for p in "$@"; do pkill -f "llama-server.*--port $p" 2>/dev/null || true; done
  sleep 1
}
kill_ports "$PORT_REF" "$PORT_RUST" "$EPORT_REF" "$EPORT_RUST"

start() { # kind port model extra...
  local kind="$1" port="$2" model="$3"; shift 3
  if [ "$kind" = ref ]; then
    nohup "$REF/llama-server" -m "$model" -c "$CTX" -t "$THREADS" -np "$NP" -fa "$FA" "$@" \
      --port "$port" --host 127.0.0.1 >"$OUT/$TAG-$kind.log" 2>&1 &
  else
    nohup ./target/release/llama-server -m "$model" -c "$CTX" -t "$THREADS" -np "$NP" -fa "$FA" "$@" \
      --port "$port" --host 127.0.0.1 >"$OUT/$TAG-$kind.log" 2>&1 &
  fi
  echo $! > "$OUT/$TAG-$kind.pid"
}
wait_ready() {
  local port="$1"
  for _ in $(seq 1 300); do
    curl -s "http://127.0.0.1:$port/health" 2>/dev/null | grep -q '"ok"' && return 0
    sleep 1
  done
  return 1
}

post() { # port path body out [extra curl args...]
  local P="$1" PATH_="$2" B="$3" O="$4"; shift 4
  curl -s -X POST "http://127.0.0.1:$P$PATH_" -H 'Content-Type: application/json' -d "$B" -o "$O" "$@"
}
post_stream() { # port path body out [extra curl args...]
  local P="$1" PATH_="$2" B="$3" O="$4"; shift 4
  curl -s -N -X POST "http://127.0.0.1:$P$PATH_" -H 'Content-Type: application/json' -d "$B" -o "$O" "$@"
}

CAPTURE_QWEN() { # port kind
  local P="$1" K="$2"
  # --- chat, tools: function calling through the jinja + autoparser path ---
  # (the lazy tool-call grammar + the <tool_call> PEG parser; first request on
  # the fresh server, temperature 0 — the tool call is deterministic)
  post "$P" /v1/chat/completions '{
    "messages":[{"role":"user","content":"What is the weather in Tokyo right now? Use the tool."}],
    "tools":[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string","description":"The city name"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city"]}}}],
    "max_tokens":200,"temperature":0,"cache_prompt":false}' \
    "$OUT/$K-chat-tools.json" -w '%{http_code}' > "$OUT/$K-chat-tools.status"
  # --- chat, tools streaming (the tool_calls deltas of the parse diffs) ---
  post_stream "$P" /v1/chat/completions '{
    "messages":[{"role":"user","content":"What is the weather in Tokyo right now? Use the tool."}],
    "tools":[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string","description":"The city name"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city"]}}}],
    "max_tokens":200,"temperature":0,"cache_prompt":false,"stream":true}' \
    "$OUT/$K-chat-tools.stream"
  # --- chat, two tools + tool_choice required (the eager tool-call grammar) ---
  post "$P" /v1/chat/completions '{
    "messages":[{"role":"user","content":"What is the weather in Tokyo right now? Use the tool."}],
    "tools":[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}},
             {"type":"function","function":{"name":"get_time","description":"Get the current time of a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
    "tool_choice":"required",
    "max_tokens":200,"temperature":0,"cache_prompt":false}' \
    "$OUT/$K-chat-tools2.json" -w '%{http_code}' > "$OUT/$K-chat-tools2.status"
  # --- chat, tools present but a plain-text answer (finish_reason "stop") ---
  # (a short deterministic answer — the port's documented forward numeric tail
  # can flip near-tied logits, so long free-form answers are avoided here)
  post "$P" /v1/chat/completions '{
    "messages":[{"role":"user","content":"What is 2+2? Reply with just the number. Do not use tools."}],
    "tools":[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
    "max_tokens":16,"temperature":0,"cache_prompt":false}' \
    "$OUT/$K-chat-tools-refusal.json" -w '%{http_code}' > "$OUT/$K-chat-tools-refusal.status"
  # --- /v1/chat/completions, non-stream (with system + history) ---
  post "$P" /v1/chat/completions '{
    "messages":[{"role":"system","content":"You are a helpful assistant."},
                {"role":"user","content":"What is the capital of France? Answer in one word."},
                {"role":"assistant","content":"Paris"},
                {"role":"user","content":"And of Germany? One word."}],
    "max_tokens":8,"temperature":0,"cache_prompt":false,"verbose":true}' "$OUT/$K-chat.json"
  # --- chat, streaming ---
  post_stream "$P" /v1/chat/completions '{
    "messages":[{"role":"user","content":"Say hello"}],
    "max_tokens":6,"temperature":0,"cache_prompt":false,"stream":true}' "$OUT/$K-chat.stream"
  # --- chat, streaming with usage ---
  post_stream "$P" /v1/chat/completions '{
    "messages":[{"role":"user","content":"Say hi"}],
    "max_tokens":6,"temperature":0,"cache_prompt":false,"stream":true,
    "stream_options":{"include_usage":true}}' "$OUT/$K-chat-usage.stream"
  # --- chat, logprobs ---
  post "$P" /v1/chat/completions '{
    "messages":[{"role":"user","content":"The capital of France is"}],
    "max_tokens":4,"temperature":0,"cache_prompt":false,"logprobs":true,"top_logprobs":3}' \
    "$OUT/$K-chat-logprobs.json"
  # --- chat, response_format json_schema ---
  post "$P" /v1/chat/completions '{
    "messages":[{"role":"user","content":"Give a person: name Alice, age 30"}],
    "max_tokens":24,"temperature":0,"cache_prompt":false,
    "response_format":{"type":"json_schema","json_schema":{"schema":{
      "type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}}}}}}' \
    "$OUT/$K-chat-schema.json"
  # --- chat, response_format json_object ---
  post "$P" /v1/chat/completions '{
    "messages":[{"role":"user","content":"Give me a JSON with name and age of Alice. Reply JSON only."}],
    "max_tokens":24,"temperature":0,"cache_prompt":false,
    "response_format":{"type":"json_object"}}' "$OUT/$K-chat-jsonobj.json"
  # --- chat, json_schema streaming (the ```json fence is parse-stripped) ---
  post_stream "$P" /v1/chat/completions '{
    "messages":[{"role":"user","content":"Give a person: name Alice, age 30"}],
    "max_tokens":24,"temperature":0,"cache_prompt":false,"stream":true,
    "response_format":{"type":"json_schema","json_schema":{"schema":{
      "type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}}}}}}' \
    "$OUT/$K-chat-schema.stream"
  # --- chat, stop word ---
  post "$P" /v1/chat/completions '{
    "messages":[{"role":"user","content":"Let us count together: one, two, three, four, five."}],
    "max_tokens":16,"temperature":0,"cache_prompt":false,"stop":[" three"]}' "$OUT/$K-chat-stop.json"
  # --- chat, alias route ---
  post "$P" /chat/completions '{
    "messages":[{"role":"user","content":"hi"}],
    "max_tokens":2,"temperature":0,"cache_prompt":false}' "$OUT/$K-chat-alias.json"
  # --- /v1/completions, non-stream + stream ---
  post "$P" /v1/completions '{
    "prompt":"The capital of France is","max_tokens":6,"temperature":0,
    "cache_prompt":false,"verbose":true}' "$OUT/$K-oaicmpl.json"
  post_stream "$P" /v1/completions '{
    "prompt":"The capital of France is","max_tokens":4,"temperature":0,
    "cache_prompt":false,"stream":true}' "$OUT/$K-oaicmpl.stream"
  # --- /models, /slots ---
  curl -s "http://127.0.0.1:$P/v1/models" -o "$OUT/$K-models.json"
  curl -s "http://127.0.0.1:$P/slots"     -o "$OUT/$K-slots.json"
  # --- rerank family (501) ---
  post "$P" /v1/rerank '{"query":"q","documents":["a","b"]}' "$OUT/$K-rerank.json" \
    -w '%{http_code}' > "$OUT/$K-rerank.status"
  # --- embeddings on a decoder server (501) ---
  post "$P" /v1/embeddings '{"input":"hi"}' "$OUT/$K-embd501.json" \
    -w '%{http_code}' > "$OUT/$K-embd501.status"
}

CAPTURE_BGE() { # port kind
  local P="$1" K="$2"
  post "$P" /v1/embeddings '{"input":["hello world","goodbye world"]}' "$OUT/$K-embd-oai.json"
  post "$P" /v1/embeddings '{"input":"single string"}' "$OUT/$K-embd-oai1.json"
  post "$P" /embedding '{"content":"hello world"}' "$OUT/$K-embd-legacy.json"
  post "$P" /embeddings '{"content":"hello world"}' "$OUT/$K-embd-legacy2.json"
  post "$P" /v1/embeddings '{"input":[],"encoding_format":"base64"}' "$OUT/$K-embd-bad.json" \
    -w '%{http_code}' > "$OUT/$K-embd-bad.status"
}

echo "== qwen2.5 pair (chat/OAI endpoints)"
start ref  "$PORT_REF"  "$MODEL"
start rust "$PORT_RUST" "$MODEL"
wait_ready "$PORT_REF"  || { echo "ref (chat) failed to start"; tail -5 "$OUT/$TAG-ref.log"; exit 1; }
wait_ready "$PORT_RUST" || { echo "rust (chat) failed to start"; tail -5 "$OUT/$TAG-rust.log"; exit 1; }
CAPTURE_QWEN "$PORT_REF"  "$TAG-ref"
CAPTURE_QWEN "$PORT_RUST" "$TAG-rust"
kill "$(cat "$OUT/$TAG-ref.pid")"  "$(cat "$OUT/$TAG-rust.pid")"  2>/dev/null || true

echo "== bge-m3 pair (embeddings; reference anchored to -fa off like the port's encoder)"
start ref  "$EPORT_REF"  "$EMBD_MODEL" --embeddings -fa off
start rust "$EPORT_RUST" "$EMBD_MODEL" --embeddings -fa off
wait_ready "$EPORT_REF"  || { echo "ref (bge) failed to start"; tail -5 "$OUT/$TAG-ref.log"; exit 1; }
wait_ready "$EPORT_RUST" || { echo "rust (bge) failed to start"; tail -5 "$OUT/$TAG-rust.log"; exit 1; }
CAPTURE_BGE "$EPORT_REF"  "$TAG-bge-ref"
CAPTURE_BGE "$EPORT_RUST" "$TAG-bge-rust"
kill "$(cat "$OUT/$TAG-ref.pid")"  "$(cat "$OUT/$TAG-rust.pid")"  2>/dev/null || true

python3 parity/server_parity_chat_cmp.py "$TAG" "$OUT"
