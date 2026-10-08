#!/usr/bin/env bash
# ngram speculative parity: the reference server and the port's server, each
# started FRESH with the same `--spec-type ngram-*` flag (self-drafting, no
# -md), answer the same FIRST request — temperature 0, the STABLE counting
# prompt of speculative_e2e.rs — and the committed streams (the response
# `content`) must match. This is the fresh-server/first-request criterion of
# PARITY.md (the in-process committed-stream-vs-plain-greedy check lives in
# crates/llama/tests/speculative_e2e.rs).
#
# Artifacts: /tmp/srvngram/<type>-{ref,rust}.json + .log
set -euo pipefail

MODEL="${MODEL:-/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
REF="${REF:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin}"
CTX="${CTX:-512}"
THREADS="${THREADS:-8}"
TYPES="${TYPES:-ngram-simple ngram-map-k ngram-map-k4v ngram-mod ngram-cache}"
PROMPT="${PROMPT:-1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}"
N="${N:-28}"
PORT_REF="${PORT_REF:-8151}"
PORT_RUST="${PORT_RUST:-8152}"
OUT="${OUT:-/tmp/srvngram}"
mkdir -p "$OUT"
cd "$(dirname "$0")/.."

run_one() {
  local type="$1"
  local port="$2" bin="$3" tag="$4"
  pkill -f "llama-server.*--port $port" 2>/dev/null || true
  sleep 1
  "$bin" -m "$MODEL" -c "$CTX" -t "$THREADS" -np 1 -fa off \
    --spec-type "$type" --port "$port" --host 127.0.0.1 >"$OUT/$type-$tag.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 300); do
    curl -s "http://127.0.0.1:$port/health" 2>/dev/null | grep -q '"ok"' && break
    sleep 1
  done
  python3 - "$port" "$PROMPT" "$N" "$OUT/$type-$tag.json" <<'PY'
import json, sys, urllib.request
port, prompt, n, out = sys.argv[1:]
body = json.dumps({"prompt": prompt, "n_predict": int(n), "temperature": 0.0,
                   "cache_prompt": False}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=600) as r:
    resp = json.load(r)
open(out, "w").write(json.dumps(resp))
print(f"  {out}: {resp.get('content')!r} (tokens {resp.get('tokens_eogd', 'n/a')}, "
      f"timings predict {resp.get('timings', {}).get('prediction_ms', 'n/a')} ms)")
PY
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

status=0
for type in $TYPES; do
  echo "== $type =="
  echo " reference:"
  run_one "$type" "$PORT_REF" "$REF/llama-server" ref
  echo " port:"
  run_one "$type" "$PORT_RUST" "./target/release/llama-server" rust

  if python3 - "$type" "$OUT" <<'PY'
import json, sys
type, out = sys.argv[1:]
a = json.load(open(f"{out}/{type}-ref.json"))
b = json.load(open(f"{out}/{type}-rust.json"))
ok = a.get("content") == b.get("content")
print(f"{type}: committed stream {'MATCH' if ok else 'MISMATCH'}")
if not ok:
    print(f"  reference: {a.get('content')!r}")
    print(f"  port     : {b.get('content')!r}")
sys.exit(0 if ok else 1)
PY
  then :; else status=1; fi
done

exit $status
