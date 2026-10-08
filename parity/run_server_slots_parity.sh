#!/usr/bin/env bash
# /slots save/restore/erase verification (the port's POST /slots/{id} wiring).
#
# Three halves:
#  1. PORT ROUND-TRIP: on the port server, a prompt's processing state is
#     saved, the slot is polluted by another prompt, restored, and the
#     continuation of the ORIGINAL prompt is compared token-for-token against
#     an uninterrupted run — identical tokens = the KV state round-tripped.
#  2. THE SAME ROUND-TRIP on the recurrent arch (mamba2): the /slots endpoint
#     must save/restore the recurrence (llama-memory-recurrent's conv/ssm
#     cells) and the continuation must match BOTH the port's uninterrupted
#     run AND the reference server's own restored continuation (cross-
#     verified on the same model — the reference leg runs below).
#  3. REFERENCE SHAPES: the reference server runs the same
#     save/restore/erase requests; the JSON *shapes* (key sets, semantic
#     fields) are compared against the port's responses field by field.
#
# Artifacts: /tmp/slotstate-*.json
set -uo pipefail
cd "$(dirname "$0")/.."

REF="${REF:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin}"
MODEL="${MODEL:-/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
# a recurrent-state arch: /slots must round-trip its conv/ssm cells
RECUR_MODEL="${RECUR_MODEL:-/tmp/arch-batch5/mamba2-synth.gguf}"
PORT_REF="${PORT_REF:-8161}"
PORT_RUST="${PORT_RUST:-8162}"
CTX="${CTX:-512}"
THREADS="${THREADS:-8}"
OUT="${OUT:-/tmp/slotstate}"
SAVEPATH="${SAVEPATH:-/tmp/slotstate-dir}"
mkdir -p "$OUT" "$SAVEPATH"
rm -f "$SAVEPATH"/*.bin

# the recurrent model is the batch-5 generator's — regenerate when missing
if [ ! -f "$RECUR_MODEL" ]; then
  cargo test --release -p llama --test arch_batch5_e2e arch_batch5_write_synth -- --ignored \
    > "$OUT/recur-synth.log" 2>&1 || { echo "batch-5 synth failed"; tail -5 "$OUT/recur-synth.log"; exit 1; }
fi

PROMPT="The capital of France is"
OTHER="The largest planet in the solar system is"

wait_ready() {
  for _ in $(seq 1 300); do
    curl -s "http://127.0.0.1:$1/health" 2>/dev/null | grep -q '"ok"' && return 0
    sleep 1
  done
  return 1
}

body() { python3 - "$@" <<'PY'
import json, sys
d = {"prompt": sys.argv[1], "n_predict": int(sys.argv[2]), "temperature": 0.0,
     "cache_prompt": sys.argv[3] == "1", "return_tokens": True}
print(json.dumps(d))
PY
}

# ---- 1) the port round-trip -------------------------------------------------
echo "== port: uninterrupted reference run"
./target/release/llama-server -m "$MODEL" -c "$CTX" -t "$THREADS" -np 1 \
  --port "$PORT_RUST" --host 127.0.0.1 --slot-save-path "$SAVEPATH" \
  >"$OUT/port.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
wait_ready "$PORT_RUST" || { echo "port server never ready"; exit 1; }

curl -s "http://127.0.0.1:$PORT_RUST/completion" -H 'Content-Type: application/json' \
  -d "$(body "$PROMPT" 16 0)" -o "$OUT/uninterrupted.json"

echo "== port: interrupted run + save / pollute / restore / continue"
curl -s "http://127.0.0.1:$PORT_RUST/completion" -H 'Content-Type: application/json' \
  -d "$(body "$PROMPT" 8 1)" -o "$OUT/first8.json"
curl -s -X POST "http://127.0.0.1:$PORT_RUST/slots/0?action=save" \
  -H 'Content-Type: application/json' -d '{"filename":"slots-parity.bin"}' -o "$OUT/port-save.json"
# pollute slot 0 with a different prompt
curl -s "http://127.0.0.1:$PORT_RUST/completion" -H 'Content-Type: application/json' \
  -d "$(body "$OTHER" 4 1)" -o "$OUT/pollute.json"
curl -s -X POST "http://127.0.0.1:$PORT_RUST/slots/0?action=restore" \
  -H 'Content-Type: application/json' -d '{"filename":"slots-parity.bin"}' -o "$OUT/port-restore.json"
# continue the ORIGINAL prompt from the restored state: the token-id prompt is
# the saved prefix (the prompt + the first 8 generated tokens), so the whole
# prompt must come from the restored cache (n_prompt_cached == its length) and
# the 8 new tokens must continue the uninterrupted trajectory
python3 - "$PORT_RUST" "$OUT" <<'PY' > "$OUT/continue-body.json"
import json, sys
port, out = sys.argv[1], sys.argv[2]
import urllib.request
def post(path, payload):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))
first8 = json.load(open(f"{out}/first8.json"))
toks = json.load(urllib.request.urlopen(urllib.request.Request(
    f"http://127.0.0.1:{port}/tokenize",
    data=json.dumps({"content": "The capital of France is", "add_special": True}).encode(),
    headers={"Content-Type": "application/json"})))["tokens"]
prefix = toks + (first8.get("tokens") or [])
print(json.dumps({"prompt": prefix, "n_predict": 8, "temperature": 0.0,
                  "cache_prompt": True, "return_tokens": True}))
PY
curl -s "http://127.0.0.1:$PORT_RUST/completion" -H 'Content-Type: application/json' \
  -d @"$OUT/continue-body.json" -o "$OUT/port-continue.json"
curl -s -X POST "http://127.0.0.1:$PORT_RUST/slots/0?action=erase" -o "$OUT/port-erase.json"

# the save/pollute/restore/continue round-trip on one server (the recurrent
# leg runs it on BOTH the port and the reference — port-recur-* / ref-recur-*)
slots_round_trip() {
  local port=$1 prefix=$2
  curl -s "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' \
    -d "$(body "$PROMPT" 16 0)" -o "$OUT/$prefix-uninterrupted.json"
  curl -s "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' \
    -d "$(body "$PROMPT" 8 1)" -o "$OUT/$prefix-first8.json"
  curl -s -X POST "http://127.0.0.1:$port/slots/0?action=save" \
    -H 'Content-Type: application/json' -d '{"filename":"recur.bin"}' -o "$OUT/$prefix-save.json"
  curl -s "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' \
    -d "$(body "$OTHER" 4 1)" -o "$OUT/$prefix-pollute.json"
  curl -s -X POST "http://127.0.0.1:$port/slots/0?action=restore" \
    -H 'Content-Type: application/json' -d '{"filename":"recur.bin"}' -o "$OUT/$prefix-restore.json"
  # continue the ORIGINAL prompt from the restored state: the token-id prompt
  # is the saved prefix (the prompt + the first 8 generated tokens), so the
  # whole prompt must come from the restored cells and the 8 new tokens must
  # continue the uninterrupted trajectory
  python3 - "$port" "$OUT" "$prefix" <<'PY' > "$OUT/$prefix-continue-body.json"
import json, sys
port, out, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
import urllib.request
def post(path, payload):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))
first8 = json.load(open(f"{out}/{prefix}-first8.json"))
toks = json.load(urllib.request.urlopen(urllib.request.Request(
    f"http://127.0.0.1:{port}/tokenize",
    data=json.dumps({"content": "The capital of France is", "add_special": True}).encode(),
    headers={"Content-Type": "application/json"})))["tokens"]
prefix_ids = toks + (first8.get("tokens") or [])
print(json.dumps({"prompt": prefix_ids, "n_predict": 8, "temperature": 0.0,
                  "cache_prompt": True, "return_tokens": True}))
PY
  curl -s "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' \
    -d @"$OUT/$prefix-continue-body.json" -o "$OUT/$prefix-continue.json"
}

# ---- 1.5) the port's recurrent arch: the /slots round-trip -------------------
echo "== port: the recurrent arch saves/restores the conv/ssm cells"
./target/release/llama-server -m "$RECUR_MODEL" -c "$CTX" -t "$THREADS" -np 1 \
  --port $((PORT_RUST + 1)) --host 127.0.0.1 --slot-save-path "$SAVEPATH" \
  >"$OUT/port-recur.log" 2>&1 &
SRV2=$!
if wait_ready $((PORT_RUST + 1)); then
  slots_round_trip $((PORT_RUST + 1)) recur-port
fi
kill $SRV2 2>/dev/null || true
wait $SRV2 2>/dev/null || true
kill $SRV 2>/dev/null || true
wait $SRV 2>/dev/null || true
trap - EXIT

# ---- 2) the reference shapes ------------------------------------------------
echo "== reference: save/restore/erase round-trip for the response shapes"
"$REF/llama-server" -m "$MODEL" -c "$CTX" -t "$THREADS" -np 1 \
  --port "$PORT_REF" --host 127.0.0.1 --slot-save-path "$SAVEPATH" \
  >"$OUT/ref.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
if wait_ready "$PORT_REF"; then
  curl -s "http://127.0.0.1:$PORT_REF/completion" -H 'Content-Type: application/json' \
    -d "$(body "$PROMPT" 8 1)" -o "$OUT/ref-warm.json"
  curl -s -X POST "http://127.0.0.1:$PORT_REF/slots/0?action=save" \
    -H 'Content-Type: application/json' -d '{"filename":"slots-parity.bin"}' -o "$OUT/ref-save.json"
  curl -s "http://127.0.0.1:$PORT_REF/completion" -H 'Content-Type: application/json' \
    -d "$(body "$OTHER" 4 1)" -o "$OUT/ref-pollute.json"
  curl -s -X POST "http://127.0.0.1:$PORT_REF/slots/0?action=restore" \
    -H 'Content-Type: application/json' -d '{"filename":"slots-parity.bin"}' -o "$OUT/ref-restore.json"
  # the identical token-id continuation the port ran
  curl -s "http://127.0.0.1:$PORT_REF/completion" -H 'Content-Type: application/json' \
    -d @"$OUT/continue-body.json" -o "$OUT/ref-continue.json"
  curl -s -X POST "http://127.0.0.1:$PORT_REF/slots/0?action=erase" -o "$OUT/ref-erase.json"
fi
kill $SRV 2>/dev/null || true
wait $SRV 2>/dev/null || true
trap - EXIT

# ---- 2.5) the reference's own recurrent round-trip (the cross-check) --------
echo "== reference: the same recurrent round-trip on mamba2"
"$REF/llama-server" -m "$RECUR_MODEL" -c "$CTX" -t "$THREADS" -np 1 \
  --port "$PORT_REF" --host 127.0.0.1 --slot-save-path "$SAVEPATH" \
  >"$OUT/ref-recur.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
if wait_ready "$PORT_REF"; then
  slots_round_trip "$PORT_REF" recur-ref
fi
kill $SRV 2>/dev/null || true
wait $SRV 2>/dev/null || true
trap - EXIT

python3 parity/server_slots_cmp.py "$OUT"
