#!/usr/bin/env bash
# server_badjson_parity.sh — dual-side malformed/invalid JSON body parity.
#
# The reference maps every JSON parse/type/at() error to 400
# invalid_request_error: common_json wraps the backing library's exceptions
# in common_json_error (common/json.cpp:20-27) and ex_wrapper catches that
# type before std::exception (tools/server/server.cpp:65-68). This suite
# pins the status AND the error type (and, for the slots filename path, the
# exact nlohmann message) on both servers, covering:
#   * truncated / non-object / empty bodies on the JSON endpoints
#   * /v1/systemone's decision-model 501 guard, which runs BEFORE the body
#     parse (server-context.cpp:5757-5762)
#   * /slots save|restore "filename" at()/type errors (400 out_of_range.403 /
#     type_error.302 — NOT the 500 wrapper)
#   * /apply-template happy path (prompt byte-equality)
# llama-rust-data-home: root of the local model/reference corpus (default: $HOME)
DATA_HOME="${LLAMA_RUST_DATA_HOME:-$HOME}"
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REF="${REF:-${DATA_HOME}/llm/llama.cpp-next/build-rust-ref/bin/llama-server}"
PORT_BIN="${PORT_BIN:-$ROOT/target/release/llama-server}"
# a model whose memory serializes (the /slots save path needs it)
MODEL="${MODEL:-/tmp/s2t-d1/tinylfm2d1-synth.gguf}"
CHAT_MODEL="${CHAT_MODEL:-${DATA_HOME}/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
OUT="${OUT:-/tmp/r4b/badjson}"
mkdir -p "$OUT/slots"
PORT_REF="${PORT_REF:-8630}"
PORT_RUST="${PORT_RUST:-8631}"

PASS=0; FAIL=0
check() { # <rc> <label>
    if [ "$1" = "0" ]; then PASS=$((PASS+1)); echo "MATCH  $2";
    else FAIL=$((FAIL+1)); echo "DIFF   $2"; fi
}

# ---- pass 1: non-decision chat model (embedding/systemone guards) ----------
start() { # <bin> <port> <model> <log> <extra...>
    parity/limited.sh -- "$1" -m "$3" -c 256 -t 4 --port "$2" --host 127.0.0.1 \
        --slot-save-path "$OUT/slots" "${@:5}" >"$4" 2>&1 &
    echo $!
}
REF_PID=$(start "$REF" "$PORT_REF" "$CHAT_MODEL" "$OUT/ref.log")
RUST_PID=$(start "$PORT_BIN" "$PORT_RUST" "$CHAT_MODEL" "$OUT/port.log")
for i in $(seq 1 120); do
    curl -s --max-time 2 "http://127.0.0.1:$PORT_REF/health" | grep -q '"ok"' && \
    curl -s --max-time 2 "http://127.0.0.1:$PORT_RUST/health" | grep -q '"ok"' && break
    sleep 1
done

python3 - "$PORT_REF" "$PORT_RUST" "$OUT/pass1" <<'EOF'
import json, sys, urllib.request, urllib.error
pr, pp, out = sys.argv[1], sys.argv[2], sys.argv[3]
def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 body.encode(), {"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=30)
        return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
cases = [
    ("/completion", '{"prompt":'),
    ("/completion", '{prompt: "hi"}'),
    ("/completion", ''),
    ("/completion", 'null'),
    ("/completion", '[1,2,3]'),
    ("/v1/chat/completions", '{"messages": ['),
    ("/v1/chat/completions", '{"messages": 5}'),
    ("/v1/chat/completions", '{}'),
    ("/tokenize", '{"content":'),
    ("/detokenize", 'zzz'),
    ("/v1/systemone", '{"state": '),   # 501 decision guard fires before the parse
    ("/apply-template", '{"messages": ['),
    ("/apply-template", '{"messages":[{"role":"user","content":"hi"}]}'),  # 200 + prompt
]
lines = []
for path, body in cases:
    sr, br = post(pr, path, body)
    sp, bp = post(pp, path, body)
    def sig(code, text):
        if code == 200: return "200"
        e = json.loads(text).get("error", {})
        return f"{code}/{e.get('type')}"
    ok = sr == sp and sig(sr, br) == sig(sp, bp)
    extra = ""
    if path == "/apply-template" and sr == 200:
        eq = json.loads(br).get("prompt") == json.loads(bp).get("prompt")
        ok = ok and eq
        extra = f" prompt_equal={eq}"
    lines.append((ok, f"{path} {body[:36]!r}: ref {sig(sr,br)} vs port {sig(sp,bp)}{extra}"))
    if not ok:
        lines.append((None, f"   ref : {br[:160]}"))
        lines.append((None, f"   port: {bp[:160]}"))
with open(out, "w") as f:
    for ok, line in lines:
        f.write(("MATCH  " if ok is True else "DIFF   " if ok is False else "       ") + line + "\n")
EOF
while read -r line; do
    case "$line" in
        MATCH*) check 0 "${line#MATCH  }";;
        DIFF*)  check 1 "${line#DIFF   }";;
    esac
done < "$OUT/pass1"
kill $REF_PID $RUST_PID 2>/dev/null; wait 2>/dev/null

# ---- pass 2: serializable model — /slots filename error shapes -------------
REF_PID=$(start "$REF" "$PORT_REF" "$MODEL" "$OUT/ref2.log")
RUST_PID=$(start "$PORT_BIN" "$PORT_RUST" "$MODEL" "$OUT/port2.log")
for i in $(seq 1 120); do
    curl -s --max-time 2 "http://127.0.0.1:$PORT_REF/health" | grep -q '"ok"' && \
    curl -s --max-time 2 "http://127.0.0.1:$PORT_RUST/health" | grep -q '"ok"' && break
    sleep 1
done
python3 - "$PORT_REF" "$PORT_RUST" "$OUT/pass2" <<'EOF'
import json, sys, urllib.request, urllib.error
pr, pp, out = sys.argv[1], sys.argv[2], sys.argv[3]
def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 body.encode(), {"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=30)
        return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
cases = [
    ("/slots/0?action=save", '{}'),
    ("/slots/0?action=save", '{"filename": 5}'),
    ("/slots/0?action=save", '{"filename": true}'),
    ("/slots/0?action=restore", '{}'),
    ("/slots/0?action=bogus", '{}'),
]
lines = []
for path, body in cases:
    sr, br = post(pr, path, body)
    sp, bp = post(pp, path, body)
    er = json.loads(br).get("error", {}); ep = json.loads(bp).get("error", {})
    ok = sr == sp and er.get("type") == ep.get("type") and er.get("message") == ep.get("message")
    lines.append((ok, f"{path} {body[:26]!r}: ref {sr}/{er.get('message')} vs port {sp}/{ep.get('message')}"))
with open(out, "w") as f:
    for ok, line in lines:
        f.write(("MATCH  " if ok else "DIFF   ") + line + "\n")
EOF
while read -r line; do
    case "$line" in
        MATCH*) check 0 "${line#MATCH  }";;
        DIFF*)  check 1 "${line#DIFF   }";;
    esac
done < "$OUT/pass2"
kill $REF_PID $RUST_PID 2>/dev/null; wait 2>/dev/null

echo
echo "badjson parity: $PASS MATCH, $FAIL DIFF (of $((PASS+FAIL)))"
[ "$FAIL" = "0" ]
