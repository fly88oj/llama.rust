#!/usr/bin/env bash
# CLI arch-dispatch parity: a *fresh* reference llama-server and the first
# /completion request on it vs ./target/release/llama-cli (PARITY.md protocol:
# fresh server + first request, temperature 0, cache_prompt=false).
#
# Two modes:
#   tokens:  run_cli_arch_parity.sh tokens <model> <fa> <port> <tag> <prompt> <n>
#   schema:  run_cli_arch_parity.sh schema <model> <fa> <port> <tag> '<json schema>' <n>
#
# `schema` mode sends the schema as the request's "json_schema" field
# (tools/server/server-schema.cpp:252-271 -> json_schema_to_grammar) and passes
# the same schema to the CLI as `-j`, so the comparison is
# "reference server grammar" vs "llama-cli -j grammar" over the same prompt.
#
# Artifacts: /tmp/parity-<tag>-{server.log,ref.json,cli.out,cli.err}
set -euo pipefail

MODE="${1:?mode: tokens|schema}"
MODEL="${2:?model}"
FA="${3:?fa on/off}"
PORT="${4:?port}"
TAG="${5:?tag}"
shift 5

REF="${REF:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin}"
CTX="${CTX:-512}"
THREADS="${THREADS:-8}"
cd "$(dirname "$0")/.."

if [ "$MODE" = tokens ]; then
  PROMPT="${1:?prompt}"
  N="${2:-16}"
  SCHEMA=""
else
  SCHEMA="${1:?json schema}"
  N="${2:-32}"
  PROMPT="Here is a JSON object:"
fi

# only ever kills a lingering server on *this* port (the script's own command
# line contains neither "llama-server" nor the port literal)
pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
sleep 2

"$REF/llama-server" -m "$MODEL" -c "$CTX" -t "$THREADS" -fa "$FA" \
  --port "$PORT" --host 127.0.0.1 >"/tmp/parity-$TAG-server.log" 2>&1 &
SRV_PID=$!
trap 'kill "$SRV_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 300); do
  # /health answers 503 {"error":{"message":"Loading model"}} until the model
  # (and the reference's CPU_REPACK buffers) are ready
  curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && break
  sleep 1
done

python3 parity/cli_parity_cmp.py body "$MODE" "$PROMPT" "$SCHEMA" "$N" \
  >"/tmp/parity-$TAG-body.json"
curl -s "http://127.0.0.1:$PORT/completion" -H "Content-Type: application/json" \
  -d @"/tmp/parity-$TAG-body.json" -o "/tmp/parity-$TAG-ref.json"
# free the reference's ~10 GiB repack before the port's own lazy repack
kill "$SRV_PID" 2>/dev/null || true
wait "$SRV_PID" 2>/dev/null || true
trap - EXIT

CLI_ARGS=(-m "$MODEL" -p "$PROMPT" -n "$N" -t "$THREADS" -c "$CTX" --temp 0 -fa "$FA")
if [ "$MODE" = schema ]; then
  CLI_ARGS+=(-j "$SCHEMA")
fi
LLAMA_RUST_DEBUG=1 ./target/release/llama-cli "${CLI_ARGS[@]}" \
  >"/tmp/parity-$TAG-cli.out" 2>"/tmp/parity-$TAG-cli.err" || true

python3 parity/cli_parity_cmp.py report "$MODE" "$N" "$TAG"