#!/usr/bin/env bash
# systemone_d1omni_media_parity.sh — the AUDIO media lane of `/v1/systemone`
# on lfm2-d1-omni (a657f7e98's media segment) vs the NEW reference server,
# the companion of systemone_d1_parity.sh's text-only 33/33. The media rows
# ride the mixed token+embd batch into `graph_decision`:
#   request files / input_audio data-URLs
#     -> decision_load_audio (files[]) / the input_audio chat part
#     -> handle_media's base64 decode (engine.rs decision_media_bytes)
#     -> process_mtmd_prompt (engine.rs prepare_decision_media: WAV decode,
#        D1omniPreproc mel, d1omni_a conformer encode)
#     -> the [BOS][audio rows][text] prompt rows with LLAMA_TOKEN_NULL at
#        the chunk positions (server_tokens)
#     -> one mixed batch through build_lfm2_decision_forward (the
#        llm_graph_input_attn_media enc/head masks + the conv neighbor taps
#        + build_inp_embd's set_rows mixed arm)
#
# the fixtures: tinylfm2d1omni-synth.gguf (systemone_d1_parity.sh's) + the
# synthetic d1omni_a audio mmproj of the same test's `gen` — a vision+audio
# mixed file, because the reference's media gate reads
# `has_inp_image = mtmd_support_vision = ctx_v != nullptr`: an audio-only
# mmproj makes the REFERENCE 501 every media request. The vision half is a
# load-only mlp stub (block_count 0, one mm.2.weight), so the reference
# must run with --no-warmup (the stub's ViT is never built; the lane only
# sends audio).
#
# checks:
#   1. the 3-question audio request (input_audio part in a chat state):
#      200 both, answers compared field by field vs the reference
#   2. the same audio via "files": [data:audio/wav;base64,...]: 200 both,
#      answers identical to the reference
#   3. two audio clips -> 400 both ("a request has images or one audio clip,
#      not both")
#   4. an http URL as input_audio -> 400 both ("audio must be base64 data")
#   5. a broken data URL (no comma) -> 400 both
#   6. without --mmproj the port keeps the 501 of the text-only lane
#      (already pinned by systemone_d1_parity.sh; asserted here once more)
# llama-rust-data-home: root of the local model/reference corpus (default: $HOME)
DATA_HOME="${LLAMA_RUST_DATA_HOME:-$HOME}"
set -uo pipefail
cd "$(dirname "$0")/.."

REF="${REF:-${DATA_HOME}/llm/llama.cpp-next/build-rust-ref/bin/llama-server}"
PORT_BIN="${PORT_BIN:-./target/release/llama-server}"
OUT=/tmp/r4o1/s2t-d1omni-media
PORT_REF=8711
PORT_RUST=8712
PORT_RUST_NOMM=8713
mkdir -p "$OUT"

# 1. regenerate the fixtures if missing
if [ ! -f "$OUT/tinylfm2d1omni-synth.gguf" ] || [ ! -f "$OUT/mmproj-d1omni-a-synth.gguf" ]; then
    echo "== generating the d1omni media fixtures =="
    parity/limited.sh -- cargo test --release -p llama-server --test systemone_d1_parity gen -- --ignored --nocapture
    # the gen test writes into /tmp/s2t-d1 — reuse its files
    [ -f /tmp/s2t-d1/tinylfm2d1omni-synth.gguf ] && cp -f /tmp/s2t-d1/tinylfm2d1omni-synth.gguf "$OUT/"
    [ -f /tmp/s2t-d1/mmproj-d1omni-a-synth.gguf ] && cp -f /tmp/s2t-d1/mmproj-d1omni-a-synth.gguf "$OUT/"
fi

# the WAV sample: the audio parity fixture (2.56 s @ 16 kHz PCM —
# gen_audio_fixture.py; the reference decodes it via miniaudio, the port via
# mtmd_audio's WAV reader — bit-equal samples, pinned by
# audio_mtmd_parity2.sh)
WAV=parity/mtmd-fixture-audio.wav
if [ ! -f "$WAV" ]; then
    parity/limited.sh -- python3 parity/gen_audio_fixture.py "$WAV" || {
        echo "DIFF   WAV fixture missing and gen_audio_fixture.py failed"; exit 1; }
fi

free -g | awk 'NR==2{print "  (mem available: "$7"G)"}'

cleanup() {
    [ -n "${REF_PID:-}" ] && kill "$REF_PID" 2>/dev/null
    [ -n "${RUST_PID:-}" ] && kill "$RUST_PID" 2>/dev/null
    [ -n "${RUST2_PID:-}" ] && kill "$RUST2_PID" 2>/dev/null
}
trap cleanup EXIT

PASS=0; FAIL=0
check() { if [ "$1" = "0" ]; then echo "MATCH  $2"; PASS=$((PASS+1)); else echo "DIFF   $2"; FAIL=$((FAIL+1)); fi }

post() { # port path body out -> http code
    parity/limited.sh -- curl -s --max-time 120 -X POST "http://127.0.0.1:$1$2" -d "$3" -o "$4" -w '%{http_code}'
}

# the base64 audio payload
B64=$(base64 -w0 "$WAV")
AUDIO_URL="data:audio/wav;base64,$B64"

# the 3-question request over a chat state with an input_audio part (the
# part is replaced by the media marker in the state the template sees — any
# media must NOT stay inside the state, the template would serialize it)
cat >"$OUT/req-audio.json" <<EOF
{"state": {"messages": [{"role": "user", "content": [
    {"type": "text", "text": "I was charged twice for my order last week and nobody replied."},
    {"type": "input_audio", "input_audio": {"data": "$B64", "format": "wav"}}
  ]}]}, "questions": {
  "route": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"billing": "payments and refunds", "shipping": null, "technical": null}},
  "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["can wait", "this week", "today", "right now"]},
  "angry": {"type": "noul", "instructions": "Is the customer angry?"}
}}
EOF

# the same audio through the files alias, string state
cat >"$OUT/req-files.json" <<EOF
{"state": "The caller says the package arrived damaged.", "files": ["$AUDIO_URL"], "questions": {
  "refund": {"type": "choice", "instructions": "Refund or replace?", "criteria": {"refund": null, "replace": null}},
  "bad": {"type": "score", "instructions": "How bad is it?", "criteria": ["fine", "broken"]}
}}
EOF

# two clips -> 400 both
cat >"$OUT/req-two.json" <<EOF
{"state": "s", "files": ["$AUDIO_URL", "$AUDIO_URL"], "questions": {
  "q": {"type": "noul", "instructions": "x"}}}
EOF

# http url -> 400
cat >"$OUT/req-http.json" <<'EOF'
{"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}},
 "images": ["http://example.com/a.wav"]}
EOF

# broken data url (no comma) -> 400
cat >"$OUT/req-broken.json" <<EOF
{"state": "s", "images": ["data:audio/wav;base64"], "questions": {
  "q": {"type": "noul", "instructions": "x"}}}
EOF

MODEL="$OUT/tinylfm2d1omni-synth.gguf"
MMPROJ="$OUT/mmproj-d1omni-a-synth.gguf"
echo
echo "==== tinylfm2d1omni + d1omni_a mmproj ===="

parity/limited.sh -- "$REF" -m "$MODEL" --mmproj "$MMPROJ" --no-warmup -c 512 -t 4 --port $PORT_REF --host 127.0.0.1 >"$OUT/ref-media.log" 2>&1 &
REF_PID=$!
parity/limited.sh -- "$PORT_BIN" -m "$MODEL" --mmproj "$MMPROJ" -c 512 -t 4 --port $PORT_RUST --host 127.0.0.1 >"$OUT/port-media.log" 2>&1 &
RUST_PID=$!

for i in $(seq 1 60); do
    ok1=$(curl -s --max-time 2 "http://127.0.0.1:$PORT_REF/health" | grep -c '"ok"' || true)
    ok2=$(curl -s --max-time 2 "http://127.0.0.1:$PORT_RUST/health" | grep -c '"ok"' || true)
    [ "$ok1" = "1" ] && [ "$ok2" = "1" ] && break
    sleep 1
done
if [ "${ok1:-0}" != "1" ] || [ "${ok2:-0}" != "1" ]; then
    echo "DIFF   servers did not come up (ref $ok1 / port $ok2) — see $OUT/*.log"
    FAIL=$((FAIL+1))
    exit 1
fi

# the per-request field comparison (python): probabilities/noul vs the
# reference within 1e-3 (the conformer's documented ~6e-6 ULP drift rides
# under it), choice == argmax, sums == 1, usage exact
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
    qa, qb = a["answers"][qid], b["answers"].get(qid)
    if qb is None:
        fail(f"{qid} missing"); continue
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

# ---- 1. the input_audio chat-state request ---------------------------------
s_ref=$(post $PORT_REF /v1/systemone @"$OUT/req-audio.json" "$OUT/ref-a-ans.json")
s_port=$(post $PORT_RUST /v1/systemone @"$OUT/req-audio.json" "$OUT/port-a-ans.json")
[ "$s_ref" = "200" ] && [ "$s_port" = "200" ]; check $? "input_audio chat state -> 200 (ref $s_ref / port $s_port)"
if [ "$s_ref" = "200" ] && [ "$s_port" = "200" ]; then
    compare_answers "$OUT" a; check $? "input_audio answers vs reference (float tol 1e-3)"
else
    check 1 "input_audio answers vs reference (request failed)"
fi

# ---- 2. the files[] alias ---------------------------------------------------
s_ref=$(post $PORT_REF /v1/systemone @"$OUT/req-files.json" "$OUT/ref-b-ans.json")
s_port=$(post $PORT_RUST /v1/systemone @"$OUT/req-files.json" "$OUT/port-b-ans.json")
[ "$s_ref" = "200" ] && [ "$s_port" = "200" ]; check $? "files[] data-url -> 200 (ref $s_ref / port $s_port)"
if [ "$s_ref" = "200" ] && [ "$s_port" = "200" ]; then
    compare_answers "$OUT" b; check $? "files[] answers vs reference (float tol 1e-3)"
else
    check 1 "files[] answers vs reference (request failed)"
fi

# ---- 3. two clips -> 400 ----------------------------------------------------
s_ref=$(post $PORT_REF /v1/systemone @"$OUT/req-two.json" "$OUT/ref-c-two.json")
s_port=$(post $PORT_RUST /v1/systemone @"$OUT/req-two.json" "$OUT/port-c-two.json")
[ "$s_ref" = "400" ] && [ "$s_port" = "400" ]; check $? "two audio clips -> 400 (ref $s_ref / port $s_port)"

# ---- 4. http URL -> 400 -----------------------------------------------------
s_ref=$(post $PORT_REF /v1/systemone @"$OUT/req-http.json" "$OUT/ref-d-http.json")
s_port=$(post $PORT_RUST /v1/systemone @"$OUT/req-http.json" "$OUT/port-d-http.json")
[ "$s_ref" = "400" ] && [ "$s_port" = "400" ]; check $? "http URL as input_audio -> 400 (ref $s_ref / port $s_port)"

# ---- 5. broken data URL -> 400 ----------------------------------------------
s_ref=$(post $PORT_REF /v1/systemone @"$OUT/req-broken.json" "$OUT/ref-e-broken.json")
s_port=$(post $PORT_RUST /v1/systemone @"$OUT/req-broken.json" "$OUT/port-e-broken.json")
[ "$s_ref" = "400" ] && [ "$s_port" = "400" ]; check $? "broken data URL -> 400 (ref $s_ref / port $s_port)"

kill "$REF_PID" "$RUST_PID" 2>/dev/null
wait 2>/dev/null
REF_PID=""; RUST_PID=""

# ---- 6. the port without --mmproj keeps the 501 ----------------------------
parity/limited.sh -- "$PORT_BIN" -m "$MODEL" -c 512 -t 4 --port $PORT_RUST_NOMM --host 127.0.0.1 >"$OUT/port-nomm.log" 2>&1 &
RUST2_PID=$!
for i in $(seq 1 60); do
    ok3=$(curl -s --max-time 2 "http://127.0.0.1:$PORT_RUST_NOMM/health" | grep -c '"ok"' || true)
    [ "$ok3" = "1" ] && break
    sleep 1
done
s_port=$(post $PORT_RUST_NOMM /v1/systemone @"$OUT/req-files.json" "$OUT/port-f-nomm.json")
[ "$s_port" = "501" ]; check $? "port without --mmproj: audio -> 501 (port $s_port)"
kill "$RUST2_PID" 2>/dev/null
wait 2>/dev/null
RUST2_PID=""

echo
echo "systemone d1omni media parity: $PASS MATCH, $FAIL DIFF (of $((PASS+FAIL)))"
[ "$FAIL" = "0" ]
