#!/usr/bin/env bash
# checkpoint_parity.sh — the prompt-cache checkpoint generation/rollback lane
# (the 033df86b6 T2 remainder) vs the NEW reference server (c35b66744): the
# reuse-class assertions of tools/server/tests/unit/test_slot_save.py
# (test_slot_restore_preserves_context_checkpoints), mirrored on BOTH
# servers and compared field by field.
#
# The model: the tinylfm2d1 synthetic (a causal lfm2 hybrid — the shortconv
# half cannot roll back, so `common_context_can_seq_rm` resolves FULL and
# the checkpoint machinery engages; the python test's tinygemma3 rides the
# SWA arm of the same gate, unportable offline). The GPT-2 BPE vocab keeps
# the token counts comparable.
#
# The scenario shape: a multi-turn prompt with `message_delimiters` —
# "USER: " + base + "USER: " + ending. The second user-message boundary is
# the divergence point, so the checkpoint the batch break creates there
# (server-context.cpp:3977-3983) sits exactly below it and turn B rolls
# back to it (the plain KV prefix is unusable on a hybrid: the recurrent
# cell sits at the old end, `pos_min >= pos_min_thold`). This mirrors the
# python test's "Keep the first prompt checkpoint before the divergence
# point" (its swa_server sets n_ubatch = 32 for the same reason).
#
# checks:
#   1. turn A processes fully — prompt_n equal on both servers, content equal
#   2. turn B (same base, different ending) rolls back to the checkpoint:
#      only the ending is processed (prompt_n small), equal on both servers,
#      content equal
#   3. --ctx-checkpoints 0 (the machinery disabled): turn B cannot reuse —
#      the whole prompt re-processes ("forcing full prompt re-processing",
#      server-context.cpp:3765-3767), prompt_n == the full length, still
#      equal on both servers
#   4. the save/restore round trip (test_slot_restore_preserves_context_
#      checkpoints' tail): erase, re-run A, save; an unrelated prompt
#      occupies the slot; restore (n_read == n_written, the SCKP appendix
#      carried); base + B again -> the same small prompt_n as (2)
set -uo pipefail
cd "$(dirname "$0")/.."

REF="${REF:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin/llama-server}"
PORT_BIN="${PORT_BIN:-./target/release/llama-server}"
MODEL="${MODEL:-/tmp/s2t-d1/tinylfm2d1-synth.gguf}"
OUT=/tmp/closd/ckpt-parity
PORT_REF=8971
PORT_RUST=8972
mkdir -p "$OUT/slots"

if [ ! -f "$MODEL" ]; then
    echo "== generating the synthetic d1 fixture =="
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

start_servers() {
    parity/limited.sh -- "$REF" -m "$MODEL" -c 512 -t 4 --port $PORT_REF --host 127.0.0.1 \
        --slot-save-path "$OUT/slots" "$@" >"$OUT/ref.log" 2>&1 &
    REF_PID=$!
    parity/limited.sh -- "$PORT_BIN" -m "$MODEL" -c 512 -t 4 --port $PORT_RUST --host 127.0.0.1 \
        --slot-save-path "$OUT/slots" "$@" >"$OUT/port.log" 2>&1 &
    RUST_PID=$!
    for i in $(seq 1 60); do
        ok1=$(curl -s --max-time 2 "http://127.0.0.1:$PORT_REF/health" | grep -c '"ok"' || true)
        ok2=$(curl -s --max-time 2 "http://127.0.0.1:$PORT_RUST/health" | grep -c '"ok"' || true)
        [ "$ok1" = "1" ] && [ "$ok2" = "1" ] && return 0
        sleep 1
    done
    echo "servers did not come up"; return 1
}

stop_servers() {
    [ -n "${REF_PID:-}" ] && kill "$REF_PID" 2>/dev/null
    [ -n "${RUST_PID:-}" ] && kill "$RUST_PID" 2>/dev/null
    wait 2>/dev/null
    REF_PID=""; RUST_PID=""
}

# post <port> <prompt-suffix-letter A|B|U> <extra json fields> -> "prompt_n content"
# the prompt is the multi-turn shape: "USER: " + BASE + "USER: " + suffix
post_case() {
    local port="$1" case="$2"
    local body
    body=$(python3 - "$case" <<'EOF'
import json, sys
c = sys.argv[1]
base = "Once upon a time there was a little robot who lived in a big city. " * 12
endings = {
    "A": "please finish the first version of this story with a happy ending.",
    "B": "please finish the second version of this story with a sad ending.",
    "U": "an unrelated message that shares nothing with the story above.",
}
body = {
    "prompt": "USER: " + base + "USER: " + endings[c],
    "id_slot": 1,
    "cache_prompt": True,
    "temperature": 0.0,
    "top_k": 1,
    "n_predict": 8,
    # the user-message boundaries (server-context.cpp:4813-4829 +
    # common/chat.cpp:126-165): the checkpoint batch break lands on them.
    # " USER:" keeps the delimiter's tokenization identical inside the
    # prompt stream (GPT-2 BPE merges the leading space into " USER")
    "message_delimiters": [{"role": "user", "delimiter": " USER:"}],
}
print(json.dumps(body))
EOF
)
    printf '%s' "$body" > "$OUT/req.json"
    parity/limited.sh -- curl -s --max-time 120 -X POST "http://127.0.0.1:$port/completion" \
        -d @"$OUT/req.json" -o "$OUT/ans-$port.json" >/dev/null
    python3 - "$OUT/ans-$port.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(d["timings"]["prompt_n"], d.get("content", ""))
EOF
}

# ---- phase 1: the checkpoint rollback (enabled, min-step 0) ----------------
echo "== phase 1: checkpoint-enabled run =="
start_servers --checkpoint-min-step 0 || { FAIL=1; exit 1; }

read pn_r ct_r <<< "$(post_case $PORT_REF A)"
read pn_p ct_p <<< "$(post_case $PORT_RUST A)"
[ -n "$pn_r" ] && [ "$pn_r" = "$pn_p" ] && [ "$pn_r" -gt 100 ]; check $? "turn A: full process, prompt_n equal (ref $pn_r / port $pn_p)"
[ "$ct_r" = "$ct_p" ]; check $? "turn A: content equal"
n_full=$pn_r

read pn_r ct_r <<< "$(post_case $PORT_REF B)"
read pn_p ct_p <<< "$(post_case $PORT_RUST B)"
# only the ending (below the second USER: boundary) is processed — the
# checkpoint rollback reused the base
[ "$pn_r" -lt 40 ] && [ "$pn_p" -lt 40 ]; check $? "turn B: checkpoint reuse, only the ending processed (ref $pn_r / port $pn_p vs n_full $n_full)"
[ "$pn_r" = "$pn_p" ]; check $? "turn B: prompt_n equal (ref $pn_r / port $pn_p)"
[ "$ct_r" = "$ct_p" ]; check $? "turn B: content equal (the rolled-back continuation)"
n_live=$pn_r

# ---- phase 2: the save/restore round trip of the checkpoints ---------------
# (test_slot_restore_preserves_context_checkpoints's tail)
curl -s -X POST "http://127.0.0.1:$PORT_REF/slots/1?action=erase" >/dev/null
curl -s -X POST "http://127.0.0.1:$PORT_RUST/slots/1?action=erase" >/dev/null
post_case $PORT_REF A >/dev/null
post_case $PORT_RUST A >/dev/null

s_ref=$(curl -s -X POST "http://127.0.0.1:$PORT_REF/slots/1?action=save" -d '{"filename": "ckpt_ref.bin"}')
s_port=$(curl -s -X POST "http://127.0.0.1:$PORT_RUST/slots/1?action=save" -d '{"filename": "ckpt_port.bin"}')
# both files carry the SCKP appendix with at least one checkpoint
for f in "$OUT/slots/ckpt_ref.bin" "$OUT/slots/ckpt_port.bin"; do
    python3 -c "
import struct, sys
data = open('$f','rb').read()
off = data.find(struct.pack('<II', 0x504b4353, 1))
assert off > 0, 'no SCKP appendix in $f'
count = struct.unpack_from('<I', data, off+8)[0]
assert count >= 1, 'empty checkpoint appendix in $f'
sys.exit(0)"; check $? "$(basename "$f"): SCKP appendix with checkpoints present"
done
w_ref=$(python3 -c "import json;print(json.loads('''$s_ref''')['n_written'])")
w_port=$(python3 -c "import json;print(json.loads('''$s_port''')['n_written'])")
[ "$w_ref" = "$(stat -c%s "$OUT/slots/ckpt_ref.bin")" ] && [ "$w_port" = "$(stat -c%s "$OUT/slots/ckpt_port.bin")" ]
check $? "save: n_written == file size (ref $w_ref / port $w_port)"

# an unrelated prompt occupies the slot
post_case $PORT_REF U >/dev/null
post_case $PORT_RUST U >/dev/null

rest_ref=$(curl -s -X POST "http://127.0.0.1:$PORT_REF/slots/1?action=restore" -d '{"filename": "ckpt_ref.bin"}')
rest_port=$(curl -s -X POST "http://127.0.0.1:$PORT_RUST/slots/1?action=restore" -d '{"filename": "ckpt_port.bin"}')
rr=$(python3 -c "import json;print(json.loads('''$rest_ref''')['n_read'])")
rp=$(python3 -c "import json;print(json.loads('''$rest_port''')['n_read'])")
[ "$rr" = "$w_ref" ] && [ "$rp" = "$w_port" ]; check $? "restore: n_read == n_written (ref $rr/$w_ref, port $rp/$w_port)"

read pn_r ct_r <<< "$(post_case $PORT_REF B)"
read pn_p ct_p <<< "$(post_case $PORT_RUST B)"
[ "$pn_r" = "$n_live" ] && [ "$pn_p" = "$n_live" ]; check $? "post-restore turn B: prompt_n == n_live (ref $pn_r / port $pn_p vs n_live $n_live)"
[ "$ct_r" = "$ct_p" ]; check $? "post-restore turn B: content equal"

stop_servers
rm -f "$OUT/slots"/*.bin

# ---- phase 3: the machinery disabled (--ctx-checkpoints 0) ----------------
echo
echo "== phase 2: checkpoint-disabled run (--ctx-checkpoints 0) =="
start_servers --ctx-checkpoints 0 || { FAIL=$((FAIL+1)); exit 1; }

read pn_r ct_r <<< "$(post_case $PORT_REF A)"
read pn_p ct_p <<< "$(post_case $PORT_RUST A)"
[ "$pn_r" = "$pn_p" ] && [ "$pn_r" -gt 100 ]; check $? "disabled turn A: full process, prompt_n equal (ref $pn_r / port $pn_p)"

read pn_r ct_r <<< "$(post_case $PORT_REF B)"
read pn_p ct_p <<< "$(post_case $PORT_RUST B)"
# without checkpoints the hybrid memory cannot reuse the diverging prompt:
# both servers re-process everything ("forcing full prompt re-processing")
[ "$pn_r" -gt 100 ] && [ "$pn_p" -gt 100 ] && [ "$pn_r" = "$pn_p" ]
check $? "disabled turn B: full re-process, prompt_n equal (ref $pn_r / port $pn_p)"
[ "$ct_r" = "$ct_p" ]; check $? "disabled turn B: content equal"

stop_servers

echo
echo "checkpoint parity: $PASS MATCH, $FAIL DIFF (of $((PASS+FAIL)))"
[ "$FAIL" = "0" ]
