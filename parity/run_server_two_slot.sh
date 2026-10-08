#!/usr/bin/env bash
# Two-slot (multi-sequence) server parity: two *interleaved* first requests on
# a fresh server (slot_id 0 and 1) must produce, for each slot, exactly the
# tokens a fresh single-slot server produces for the same prompt — on the
# reference and on the port. The port's engine batches both slots' prompts into
# one multi-sequence decode (watch LLAMA_SERVER_DEBUG=1 in its log).
#
# Artifacts: /tmp/srv2-<tag>-*.json
set -euo pipefail

MODEL="${MODEL:-/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
REF="${REF:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin}"
CTX="${CTX:-1024}"
THREADS="${THREADS:-8}"
N="${N:-16}"
FA="${FA:-off}"
PROMPT_A="${PROMPT_A:-The capital of France is}"
PROMPT_B="${PROMPT_B:-1 + 1 =}"
OUT="${OUT:-/tmp/srv2}"
mkdir -p "$OUT"
cd "$(dirname "$0")/.."

declare -A PORTS=([ref1]=8151 [ref2]=8152 [rust2]=8153)
pkill -f "llama-server.*--port 815" 2>/dev/null || true
sleep 1

req() { # port slot prompt tag
  curl -s -m 120 -X POST "http://127.0.0.1:$1/completion" -H 'Content-Type: application/json' \
    -d "{\"prompt\":\"$3\",\"n_predict\":$N,\"temperature\":0.0,\"cache_prompt\":false,\"id_slot\":$2}" \
    -o "$OUT/$4.json"
}

# 1. the reference with a single slot = the ground truth for one sequence
"$REF/llama-server" -m "$MODEL" -c "$CTX" -t "$THREADS" -np 1 -fa "$FA" \
  --port "${PORTS[ref1]}" --host 127.0.0.1 >"$OUT/ref1.log" 2>&1 &
P1=$!
trap 'kill -9 $P1 2>/dev/null || true' EXIT
for _ in $(seq 1 300); do
  curl -s "http://127.0.0.1:${PORTS[ref1]}/health" 2>/dev/null | grep -q '"ok"' && break
  sleep 1
done
req "${PORTS[ref1]}" 0 "$PROMPT_A" ref_np1_a
req "${PORTS[ref1]}" 0 "$PROMPT_B" ref_np1_b
kill -9 $P1 2>/dev/null || true
trap - EXIT

# 2. the reference with two slots, both requests issued before either finishes
"$REF/llama-server" -m "$MODEL" -c "$CTX" -t "$THREADS" -np 2 -fa "$FA" \
  --port "${PORTS[ref2]}" --host 127.0.0.1 >"$OUT/ref2.log" 2>&1 &
P2=$!
trap 'kill -9 $P2 2>/dev/null || true' EXIT
for _ in $(seq 1 300); do
  curl -s "http://127.0.0.1:${PORTS[ref2]}/health" 2>/dev/null | grep -q '"ok"' && break
  sleep 1
done
req "${PORTS[ref2]}" 0 "$PROMPT_A" ref_np2_a &
RA=$!
req "${PORTS[ref2]}" 1 "$PROMPT_B" ref_np2_b &
RB=$!
# note: a bare `wait` would also wait for the server started above
wait $RA $RB
kill -9 $P2 2>/dev/null || true
trap - EXIT

# 3. the port with two slots, the same interleaved pair
LLAMA_SERVER_DEBUG=1 ./target/release/llama-server -m "$MODEL" -c "$CTX" -t "$THREADS" -np 2 -fa "$FA" \
  --port "${PORTS[rust2]}" --host 127.0.0.1 >"$OUT/rust2.log" 2>&1 &
P3=$!
trap 'kill -9 $P3 2>/dev/null || true' EXIT
for _ in $(seq 1 300); do
  curl -s "http://127.0.0.1:${PORTS[rust2]}/health" 2>/dev/null | grep -q '"ok"' && break
  sleep 1
done
req "${PORTS[rust2]}" 0 "$PROMPT_A" rust_np2_a &
RA=$!
req "${PORTS[rust2]}" 1 "$PROMPT_B" rust_np2_b &
RB=$!
wait $RA $RB
kill -9 $P3 2>/dev/null || true
trap - EXIT

python3 - "$OUT" <<'PY'
import json, sys
d = sys.argv[1]
def load(n):
    return json.load(open(f"{d}/{n}.json"))
a1, b1, a2, b2, ar, br = (load(n) for n in
    ["ref_np1_a", "ref_np1_b", "ref_np2_a", "ref_np2_b", "rust_np2_a", "rust_np2_b"])
ok = True
for label, x, y in [
    ("ref -np2 slot0 vs ref -np1", a2, a1),
    ("ref -np2 slot1 vs ref -np1", b2, b1),
    ("port -np2 slot0 vs ref -np1", ar, a1),
    ("port -np2 slot1 vs ref -np1", br, b1),
    ("port -np2 slot0 vs ref -np2 slot0", ar, a2),
    ("port -np2 slot1 vs ref -np2 slot1", br, b2),
]:
    same_text = x.get("content") == y.get("content")
    same_n = x.get("tokens_predicted") == y.get("tokens_predicted")
    same_slot = x.get("id_slot") == y.get("id_slot")
    same_eval = x.get("tokens_evaluated") == y.get("tokens_evaluated")
    ok &= same_text and same_n and same_eval
    print(f"[{'OK ' if same_text and same_n and same_eval else 'DIFF'}] {label}: "
          f"tokens={x.get('tokens_predicted')}/{y.get('tokens_predicted')} "
          f"evaluated={x.get('tokens_evaluated')}/{y.get('tokens_evaluated')} "
          f"slot={x.get('id_slot')}/{y.get('id_slot')} "
          f"content equal={same_text}")
    if not same_text:
        print(f"        ref : {y.get('content')!r}\n        port: {x.get('content')!r}")
print("RESULT:", "MATCH" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
PY
echo "--- port batching evidence (LLAMA_SERVER_DEBUG=1) ---"
grep -c "n_seqs = \[0, 1\]" "$OUT/rust2.log" || true
grep "n_seqs" "$OUT/rust2.log" | head -4