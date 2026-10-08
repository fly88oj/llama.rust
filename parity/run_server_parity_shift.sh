#!/usr/bin/env bash
# llama-server context-shift parity: the reference server and the port's
# server, both started with `--context-shift` on a small `-c`, answer the same
# long-prompt greedy request; the generated text must match token for token
# through every context shift (the truncation semantics are computed by both
# sides, so this is an exact-stream comparison).
#
# protocol (PARITY.md): a *fresh* server and the *first* request on it,
# temperature 0, cache_prompt=false, -fa as given, --context-shift.
#
# `-c 256` (a multiple of 256 — the reference pads `cparams.n_ctx` to 256,
# llama-context.cpp:290): prompt ~220 tokens, n_predict 160 → the slot fills
# at 255 tokens mid-generation and shifts twice (n_keep 0, n_discard = half),
# server-context.cpp:2909-2972.
#
# Artifacts: /tmp/srvparity-shift-<tag>-{ref,rust}.json + .log
set -euo pipefail

MODEL="${MODEL:-/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
REF="${REF:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin}"
CTX="${CTX:-256}"
THREADS="${THREADS:-8}"
NP="${NP:-1}"
FA="${FA:-off}"
N="${N:-260}"
TAG="${TAG:-shift}"
PORT_REF="${PORT_REF:-8161}"
PORT_RUST="${PORT_RUST:-8162}"
OUT="${OUT:-/tmp/srvparity-shift}"
mkdir -p "$OUT"
cd "$(dirname "$0")/.."

bash parity/kill_stale_servers.sh >/dev/null 2>&1 || true
pkill -f "llama-server.*--port $PORT_REF" 2>/dev/null || true
pkill -f "llama-server.*--port $PORT_RUST" 2>/dev/null || true
sleep 1

start_ref() {
  "$REF/llama-server" -m "$MODEL" -c "$CTX" -t "$THREADS" -np "$NP" -fa "$FA" \
    --context-shift --port "$PORT_REF" --host 127.0.0.1 >"$OUT/$TAG-ref.log" 2>&1 &
  REF_PID=$!
}
start_rust() {
  ./target/release/llama-server -m "$MODEL" -c "$CTX" -t "$THREADS" -np "$NP" -fa "$FA" \
    --context-shift --port "$PORT_RUST" --host 127.0.0.1 >"$OUT/$TAG-rust.log" 2>&1 &
  RUST_PID=$!
}
wait_ready() {
  local port="$1"
  for _ in $(seq 1 300); do
    curl -s "http://127.0.0.1:$port/health" 2>/dev/null | grep -q '"ok"' && return 0
    sleep 1
  done
  return 1
}

# ~200-token prompt: long enough that the first shift lands mid-generation
# (must stay below -c so the request is accepted)
PROMPT_FILE="$OUT/prompt.txt"
python3 - "$PROMPT_FILE" <<'PY'
import sys
base = ("The city of Paris has been a center of commerce, science, and art for "
        "centuries. Its streets preserve layers of history, from Roman ruins "
        "beneath the medieval quarters to the grand avenues of the nineteenth "
        "century. Writers and painters came here to work, to argue, and to "
        "publish ideas that would travel far beyond the river banks. ")
open(sys.argv[1], "w").write(base * 3 + "Today the city continues")
PY

BODY=$(python3 - "$PROMPT_FILE" "$N" <<'PY'
import json, sys
print(json.dumps({"prompt": open(sys.argv[1]).read(), "n_predict": int(sys.argv[2]),
                  "temperature": 0.0, "cache_prompt": False, "n_keep": 0, "n_discard": 0}))
PY
)

capture() {
  local port="$1" tag="$2"
  curl -s -X POST "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' \
       -d "$(python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))))' <(echo "$BODY"))" \
       -o "$OUT/$tag-completion.json"
}

start_ref
wait_ready "$PORT_REF"
capture "$PORT_REF" "$TAG-ref"
kill "$REF_PID" 2>/dev/null || true
wait "$REF_PID" 2>/dev/null || true

start_rust
wait_ready "$PORT_RUST"
capture "$PORT_RUST" "$TAG-rust"
kill "$RUST_PID" 2>/dev/null || true
wait "$RUST_PID" 2>/dev/null || true

python3 - "$TAG" "$OUT" <<'PY'
import json, sys

tag, out = sys.argv[1], sys.argv[2]
a = json.load(open(f"{out}/{tag}-ref-completion.json"))
b = json.load(open(f"{out}/{tag}-rust-completion.json"))

ok = True
ca, cb = a.get("content"), b.get("content")
print(f"reference content ({len(ca)} chars): {ca!r}"[:200])
print(f"port      content ({len(cb)} chars): {cb!r}"[:200])
if ca != cb:
    n = 0
    for x, y in zip(ca or "", cb or ""):
        if x != y:
            break
        n += 1
    print(f"FAIL: generated text differs (common prefix {n} chars)")
    ok = False
else:
    print("token/text match: OK")

# non-timing fields: stop type, token counts, truncated
for k in ("stop_type", "tokens_evaluated", "n_prompt_tokens", "truncated", "stop"):
    va, vb = a.get(k), b.get(k)
    if va != vb:
        print(f"DIFF {k}: ref={va!r} port={vb!r}")
        ok = False

# both servers must have actually shifted: the run went past -c (prompt 199 +
# n_predict > 256) and reports truncated; the reference's log carries the
# "slot context shift" warning (SLT_WRN, server-context.cpp:2949)
import re
shifts_ref = len(re.findall(r"slot context shift", open(f"{out}/{tag}-ref.log").read()))
print(f"reference 'slot context shift' events: {shifts_ref}")
if shifts_ref < 1 or not a.get("truncated"):
    print("FAIL: the reference run never shifted — the case is vacuous")
    ok = False
if not b.get("truncated"):
    print("FAIL: the port's run never shifted")
    ok = False

print("RESULT:", "MATCH" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
PY
