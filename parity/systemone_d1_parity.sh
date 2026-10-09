#!/usr/bin/env bash
# systemone_d1_parity.sh — `/v1/systemone` e2e of the d1 / d1-omni /
# pplx-decider decision formats (88dcc460d + a657f7e98 + da263e727) vs the
# NEW reference server (c35b66744), the d1-family lane of the systemone
# parity protocol (systemone_parity.sh's laya lane stays 12/12 untouched).
#
# The fixtures are the three SYNTHETIC models of
# crates/tools/llama-server/tests/systemone_d1_parity.rs (`gen`, ignored):
#   * tinylfm2d1-synth.gguf   — causal lfm2 LM + lfm2.decision.type=lfm2-d1
#     (labels at the last prompt token; the d1-3B shape, GPT-2 BPE vocab so
#     every label form is a single token)
#   * tinylfm2d1omni-synth.gguf — the non-causal lfm2 decision head
#     (lfm2.decision.type=lfm2-d1-omni; markers read from the embeddings
#     output; the port drives it on its own stateless decision core — the
#     reference's decode->encode reroute)
#   * tinypplx-synth.gguf     — causal llama LM +
#     llama.decision.type=pplx-decider (the protocol-level carrier of the
#     real qwen35 model: label codes, last-token logits, shared-prompt
#     children are arch-independent; recorded in PARITY.md)
#
# checks per model (mirroring the laya lane's shape):
#   1. the 3-question request: status 200, answers structure, probabilities
#      vs the reference (float tolerance 1e-3 — the d1 parent prompt is a
#      2-batch split like the reference's, residual batch-shape drift is
#      ~1.6e-4), choice == argmax, sums == 1, noul in [0,1], usage exact
#   2. a null state (d1-family only) -> 200 on both sides
#   3. invalid-request matrix -> 400 on both sides
#   4. images -> 501 on both sides (no mmproj wired)
set -uo pipefail
cd "$(dirname "$0")/.."

REF="${REF:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin/llama-server}"
PORT_BIN="${PORT_BIN:-./target/release/llama-server}"
OUT=/tmp/s2t-d1
PORT_REF=8961
PORT_RUST=8962
mkdir -p "$OUT"

# 1. regenerate the fixtures if missing
if [ ! -f "$OUT/tinylfm2d1-synth.gguf" ] || [ ! -f "$OUT/tinylfm2d1omni-synth.gguf" ] || [ ! -f "$OUT/tinypplx-synth.gguf" ]; then
    echo "== generating the synthetic d1 fixtures =="
    parity/limited.sh -- cargo test --release -p llama-server --test systemone_d1_parity gen -- --ignored --nocapture
fi

free -g | awk 'NR==2{print "  (mem available: "$7"G)"}'

cleanup() {
    [ -n "${REF_PID:-}" ] && kill "$REF_PID" 2>/dev/null
    [ -n "${RUST_PID:-}" ] && kill "$RUST_PID" 2>/dev/null
}
trap cleanup EXIT

PASS=0; FAIL=0
check() { if [ "$1" = "0" ]; then echo "MATCH  $2"; PASS=$((PASS+1)); else echo "DIFF   $2"; FAIL=$((FAIL+1)); fi }

post() { # port path body out -> http code
    parity/limited.sh -- curl -s --max-time 120 -X POST "http://127.0.0.1:$1$2" -d "$3" -o "$4" -w '%{http_code}'
}

cat >"$OUT/req.json" <<'EOF'
{"state": "I was charged twice for my order last week and nobody has replied.", "questions": {
  "route": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"billing": "payments and refunds", "shipping": null, "technical": null}},
  "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["can wait", "this week", "today", "right now"]},
  "angry": {"type": "noul", "instructions": "Is the customer angry?"}
}}
EOF

# a 2-option choice (the choice.2 temperature bucket) + a 2-level score
cat >"$OUT/req2.json" <<'EOF'
{"state": "The package arrived damaged.", "questions": {
  "refund": {"type": "choice", "instructions": "Refund or replace?", "criteria": {"refund": null, "replace": null}},
  "bad": {"type": "score", "instructions": "How bad is it?", "criteria": ["fine", "broken"]}
}}
EOF

# the per-request field comparison (python): probabilities/noul agree within
# the tolerance, choice == argmax, sums == 1, usage identical
compare_answers() { # out prefix
    parity/limited.sh -- python3 - "$1" "$2" <<'EOF'
import json, sys
out, pfx = sys.argv[1], sys.argv[2]
a = json.load(open(f"{out}/ref-{pfx}-ans.json"))
b = json.load(open(f"{out}/port-{pfx}-ans.json"))
ok = True
def fail(msg):
    global ok; ok = False; print("  DIFF:", msg)
if a["usage"] != b["usage"]:
    fail(f"usage {a['usage']} vs {b['usage']}")
if a["usage"]["output_tokens"] != 0:
    fail("output_tokens != 0")
for qid in a["answers"]:
    qa, qb = a["answers"][qid], b["answers"][qid]
    if qa["type"] != qb["type"]:
        fail(f"{qid}.type"); continue
    if "noul" in qa:
        if abs(qa["noul"] - qb["noul"]) > 1e-3:
            fail(f"{qid}.noul {qa['noul']} vs {qb['noul']}")
        if not (0.0 <= qb["noul"] <= 1.0):
            fail(f"{qid}.noul out of [0,1]")
        continue
    pa, pb = qa["probabilities"], qb["probabilities"]
    if list(pa.keys()) != list(pb.keys()):
        fail(f"{qid}.probabilities keys {list(pa)} vs {list(pb)}")
    s = 0.0
    for k in pa:
        if abs(pa[k] - pb[k]) > 1e-3:
            fail(f"{qid}.probabilities[{k}] {pa[k]} vs {pb[k]}")
        s += pb[k]
    if abs(s - 1.0) > 1e-4:
        fail(f"{qid}.probabilities sum {s}")
    best = max(pb, key=pb.get)
    if qb.get("choice") is not None and qb["choice"] != best:
        fail(f"{qid}.choice {qb['choice']} != argmax {best}")
    if "score" in qa:
        exp = sum(int(k) * v for k, v in pb.items())
        if abs(qb["score"] - exp) > 1e-3:
            fail(f"{qid}.score {qb['score']} != expectation {exp}")
sys.exit(0 if ok else 1)
EOF
}

for MODEL in "$OUT/tinylfm2d1-synth.gguf" "$OUT/tinylfm2d1omni-synth.gguf" "$OUT/tinypplx-synth.gguf"; do
    NAME=$(basename "$MODEL" .gguf)
    echo
    echo "==== $NAME ===="

    parity/limited.sh -- "$REF" -m "$MODEL" -c 512 -t 4 --port $PORT_REF --host 127.0.0.1 >"$OUT/ref-$NAME.log" 2>&1 &
    REF_PID=$!
    parity/limited.sh -- "$PORT_BIN" -m "$MODEL" -c 512 -t 4 --port $PORT_RUST --host 127.0.0.1 >"$OUT/port-$NAME.log" 2>&1 &
    RUST_PID=$!

    for i in $(seq 1 60); do
        ok1=$(curl -s --max-time 2 "http://127.0.0.1:$PORT_REF/health" | grep -c '"ok"' || true)
        ok2=$(curl -s --max-time 2 "http://127.0.0.1:$PORT_RUST/health" | grep -c '"ok"' || true)
        [ "$ok1" = "1" ] && [ "$ok2" = "1" ] && break
        sleep 1
    done
    if [ "${ok1:-0}" != "1" ] || [ "${ok2:-0}" != "1" ]; then
        echo "DIFF   $NAME: servers did not come up (ref $ok1 / port $ok2)"
        FAIL=$((FAIL+1))
        kill "$REF_PID" "$RUST_PID" 2>/dev/null; wait 2>/dev/null
        continue
    fi

    # ---- the 3-question request --------------------------------------------
    s_ref=$(post $PORT_REF /v1/systemone @"$OUT/req.json" "$OUT/ref-a-ans.json")
    s_port=$(post $PORT_RUST /v1/systemone @"$OUT/req.json" "$OUT/port-a-ans.json")
    [ "$s_ref" = "200" ] && [ "$s_port" = "200" ]; check $? "$NAME: 3-question request 200 (ref $s_ref / port $s_port)"
    compare_answers "$OUT" a; check $? "$NAME: 3-question answers vs reference (float tol 1e-3)"

    # ---- the 2-option/2-level request (the .2 temperature buckets) --------
    s_ref=$(post $PORT_REF /v1/systemone @"$OUT/req2.json" "$OUT/ref-b-ans.json")
    s_port=$(post $PORT_RUST /v1/systemone @"$OUT/req2.json" "$OUT/port-b-ans.json")
    [ "$s_ref" = "200" ] && [ "$s_port" = "200" ]; check $? "$NAME: 2-option request 200 (ref $s_ref / port $s_port)"
    compare_answers "$OUT" b; check $? "$NAME: 2-option answers vs reference (float tol 1e-3)"

    # ---- a null state (accepted by the d1 family only; pplx rejects it
    # like every non-d1 type, server-decision.cpp:147-149) -------------------
    want_null=200
    [ "$NAME" = "tinypplx-synth" ] && want_null=400
    s_ref=$(post $PORT_REF /v1/systemone '{"state": null, "questions": {"q": {"type": "noul", "instructions": "x"}}}' "$OUT/ref-null.json")
    s_port=$(post $PORT_RUST /v1/systemone '{"state": null, "questions": {"q": {"type": "noul", "instructions": "x"}}}' "$OUT/port-null.json")
    [ "$s_ref" = "$want_null" ] && [ "$s_port" = "$want_null" ]; check $? "$NAME: null state -> $want_null (ref $s_ref / port $s_port)"

    # ---- the invalid-request matrix (both 400) -----------------------------
    i=0
    while IFS= read -r body; do
        i=$((i+1))
        s1=$(post $PORT_REF /v1/systemone "$body" "$OUT/ref-bad$i.json")
        s2=$(post $PORT_RUST /v1/systemone "$body" "$OUT/port-bad$i.json")
        [ "$s1" = "400" ] && [ "$s2" = "400" ]; check $? "$NAME: invalid request #$i -> 400 (ref $s1 / port $s2)"
    done <<'BODIES'
{"questions": {"q": {"type": "noul", "instructions": "x"}}}
{"state": "s", "questions": {}}
{"state": "s", "questions": {"q": {"type": "unknown", "instructions": "x"}}}
{"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {}}}}
{"state": "s", "questions": {"q": {"type": "score", "instructions": "x", "criteria": ["only one"]}}}
BODIES

    # ---- images -> 501 (no mmproj wired) ------------------------------------
    img='data:image/png;base64,iVBORw0KGgo='
    s1=$(post $PORT_REF /v1/systemone "{\"state\": \"s\", \"images\": [\"$img\"], \"questions\": {\"q\": {\"type\": \"noul\", \"instructions\": \"x\"}}}" "$OUT/ref-img.json")
    s2=$(post $PORT_RUST /v1/systemone "{\"state\": \"s\", \"images\": [\"$img\"], \"questions\": {\"q\": {\"type\": \"noul\", \"instructions\": \"x\"}}}" "$OUT/port-img.json")
    [ "$s1" = "501" ] && [ "$s2" = "501" ]; check $? "$NAME: images -> 501 (ref $s1 / port $s2)"

    kill "$REF_PID" "$RUST_PID" 2>/dev/null
    wait 2>/dev/null
    REF_PID=""; RUST_PID=""
done

echo
echo "systemone d1-family parity: $PASS MATCH, $FAIL DIFF (of $((PASS+FAIL)))"
[ "$FAIL" = "0" ]
