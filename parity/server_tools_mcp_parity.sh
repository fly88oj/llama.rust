#!/usr/bin/env bash
# server_tools_mcp_parity.sh — the server-side tools/MCP/GCP/cors-proxy
# surfaces of the port vs the pinned reference (server-tools.cpp,
# server-mcp.cpp, server-http.cpp:797-923, server-cors-proxy.h).
#
# Comparisons (JSON normalized through python's json.loads):
#   1. /tools GET — the tool-definition exposure (schema + permissions)
#   2. /tools POST — read_file + exec_shell_command round-trips on fixtures
#   3. /tools 403 — the disabled-feature answer without --tools
#   4. MCP — a shell-based MCP server via --mcp-servers-json: the /tools
#      listing gains the bridged tool, and a call round-trips
#   5. GCP compat — AIP_MODE=PREDICTION: /predict chatCompletions alias +
#      the AIP_HEALTH_ROUTE alias
#   6. cors-proxy — POST forwarding to a loopback echo server (the port
#      proxies http:// only; the reference's https is a documented gap)
set -u

MODEL="${1:-/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
REF_BIN_DIR="${REF_BIN_DIR:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin}"
PORT_BIN="${PORT_BIN:-$(dirname "$0")/../target/release/llama-server}"
REF="$REF_BIN_DIR/llama-server"
WORK=$(mktemp -d /tmp/llama-tools-parity.XXXXXX)
PASS=0; FAIL=0

export LD_LIBRARY_PATH="$REF_BIN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cleanup() {
    for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null; done
    [ -z "${KEEP:-}" ] && rm -rf "$WORK"
}
trap cleanup EXIT
PIDS=()

start_server() { # bin, port, extra args...
    local bin="$1" port="$2"; shift 2
    "$bin" -m "$MODEL" --host 127.0.0.1 --port "$port" -c 256 -t 8 "$@" \
        >"$WORK/srv-$port.log" 2>&1 &
    PIDS+=($!)
    for _ in $(seq 1 240); do
        curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && return 0
        sleep 0.5
    done
    echo "server on $port failed to start:" >&2; tail -5 "$WORK/srv-$port.log" >&2
    return 1
}

cmp_json() { # name, ref-file, port-file
    if python3 - "$2" "$3" <<'EOF'
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
sys.exit(0 if a == b else 1)
EOF
    then echo "MATCH  $1"; PASS=$((PASS+1))
    else echo "DIFF   $1"; FAIL=$((FAIL+1)); fi
}

cmp_text() {
    if cmp -s "$2" "$3"; then echo "MATCH  $1"; PASS=$((PASS+1))
    else echo "DIFF   $1"; FAIL=$((FAIL+1)); diff "$2" "$3" | head -4; fi
}

# --- fixtures ---
mkdir -p "$WORK/ctx/sub"
printf 'alpha\nbeta\ngamma\n' > "$WORK/ctx/sub/a.txt"

# --- 1-3: the built-in tools ---
start_server "$REF"  8331 --tools read_file,exec_shell_command,write_file,edit_file,get_info || exit 2
start_server "$PORT_BIN" 8332 --tools read_file,exec_shell_command,write_file,edit_file,get_info || exit 2
sleep 1

curl -s "http://127.0.0.1:8331/tools" > "$WORK/tools-ref.json"
curl -s "http://127.0.0.1:8332/tools" > "$WORK/tools-port.json"
cmp_json "/tools GET schema exposure" "$WORK/tools-ref.json" "$WORK/tools-port.json"

post() { curl -s -X POST "http://127.0.0.1:$1/tools" -H 'content-type: application/json' -d "$2"; }

post 8331 "{\"tool\":\"read_file\",\"params\":{\"path\":\"$WORK/ctx/sub/a.txt\",\"start_line\":2,\"end_line\":3,\"append_loc\":true}}" > "$WORK/rf-ref.json"
post 8332 "{\"tool\":\"read_file\",\"params\":{\"path\":\"$WORK/ctx/sub/a.txt\",\"start_line\":2,\"end_line\":3,\"append_loc\":true}}" > "$WORK/rf-port.json"
cmp_json "/tools POST read_file range" "$WORK/rf-ref.json" "$WORK/rf-port.json"

post 8331 '{"tool":"exec_shell_command","params":{"command":"printf tool-out; echo err 1>&2"}}' > "$WORK/exec-ref.json"
post 8332 '{"tool":"exec_shell_command","params":{"command":"printf tool-out; echo err 1>&2"}}' > "$WORK/exec-port.json"
cmp_json "/tools POST exec_shell_command" "$WORK/exec-ref.json" "$WORK/exec-port.json"

post 8331 '{"tool":"unknown_tool","params":{}}' > "$WORK/unk-ref.json"
post 8332 '{"tool":"unknown_tool","params":{}}' > "$WORK/unk-port.json"
cmp_json "/tools POST unknown tool 404" "$WORK/unk-ref.json" "$WORK/unk-port.json"

post 8331 '{"tool":"read_file","params":{"path":"/nonexistent-llama-parity"}}' > "$WORK/nf-ref.json"
post 8332 '{"tool":"read_file","params":{"path":"/nonexistent-llama-parity"}}' > "$WORK/nf-port.json"
cmp_json "/tools POST read_file stat error" "$WORK/nf-ref.json" "$WORK/nf-port.json"

kill "${PIDS[@]}"; PIDS=(); sleep 1

# --- 3b: without --tools the endpoints answer the 403 ---
start_server "$REF"  8333 || exit 2
start_server "$PORT_BIN" 8334 || exit 2
curl -s "http://127.0.0.1:8333/tools" > "$WORK/403-ref.json"
curl -s "http://127.0.0.1:8334/tools" > "$WORK/403-port.json"
cmp_json "/tools GET disabled 403" "$WORK/403-ref.json" "$WORK/403-port.json"
curl -s "http://127.0.0.1:8333/cors-proxy?url=http://127.0.0.1:1/" > "$WORK/cp403-ref.json"
curl -s "http://127.0.0.1:8334/cors-proxy?url=http://127.0.0.1:1/" > "$WORK/cp403-port.json"
cmp_json "/cors-proxy disabled 403" "$WORK/cp403-ref.json" "$WORK/cp403-port.json"
kill "${PIDS[@]}"; PIDS=(); sleep 1

# --- 4: MCP through a shell "server" ---
MCP_JSON='{"mcpServers":{"shelly":{"command":"/bin/sh","args":["-c","'"$(sed "s/'/''/g" <<'SH'
while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n '"'"'s/.*"id":\([0-9]*\).*/\1/p'"'"')
  method=$(printf '%s' "$line" | sed -n '"'"'s/.*"method":"\([a-z/]*\)".*/\1/p'"'"')
  if [ "$method" = "initialize" ]; then
    printf '"'"'{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2024-11-05"}}\n'"'"' "$id"
  elif [ "$method" = "tools/list" ]; then
    printf '"'"'{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"echo","description":"Echo tool","inputSchema":{"type":"object","properties":{"text":{"type":"string"}}}}]}}\n'"'"' "$id"
  elif [ "$method" = "tools/call" ]; then
    printf '"'"'{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"echoed"}]}}\n'"'"' "$id"
  fi
done
SH
)"'"]}}}'

# the json needs the script embedded as ONE argv string — build it with python
MCP_JSON=$(python3 - <<'PYEOF'
import json
script = r'''
while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9]*\).*/\1/p')
  method=$(printf '%s' "$line" | sed -n 's/.*"method":"\([a-z/]*\)".*/\1/p')
  if [ "$method" = "initialize" ]; then
    printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2024-11-05"}}\n' "$id"
  elif [ "$method" = "tools/list" ]; then
    printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"echo","description":"Echo tool","inputSchema":{"type":"object","properties":{"text":{"type":"string"}}}}]}}\n' "$id"
  elif [ "$method" = "tools/call" ]; then
    printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"echoed"}]}}\n' "$id"
  fi
done
'''
print(json.dumps({"mcpServers": {"shelly": {"command": "/bin/sh", "args": ["-c", script]}}}))
PYEOF
)

start_server "$REF"  8335 --mcp-servers-json "$MCP_JSON" || exit 2
start_server "$PORT_BIN" 8336 --mcp-servers-json "$MCP_JSON" || exit 2
sleep 2

curl -s "http://127.0.0.1:8335/tools" > "$WORK/mcp-ref.json"
curl -s "http://127.0.0.1:8336/tools" > "$WORK/mcp-port.json"
cmp_json "MCP /tools listing (server tool + bridge)" "$WORK/mcp-ref.json" "$WORK/mcp-port.json"

post 8335 '{"tool":"shelly_echo","params":{"text":"hi"}}' > "$WORK/mcpc-ref.json"
post 8336 '{"tool":"shelly_echo","params":{"text":"hi"}}' > "$WORK/mcpc-port.json"
cmp_json "MCP tools/call round-trip" "$WORK/mcpc-ref.json" "$WORK/mcpc-port.json"
kill "${PIDS[@]}"; PIDS=(); sleep 1

# --- 5: GCP compat ---
AIP_MODE=PREDICTION AIP_HTTP_PORT=8337 AIP_HEALTH_ROUTE=/healthz \
    start_server "$REF"  8337 -c 256 || exit 2
AIP_MODE=PREDICTION AIP_HTTP_PORT=8338 AIP_HEALTH_ROUTE=/healthz \
    start_server "$PORT_BIN" 8338 -c 256 || exit 2
sleep 1

curl -s "http://127.0.0.1:8337/healthz" > "$WORK/gcph-ref.txt"
curl -s "http://127.0.0.1:8338/healthz" > "$WORK/gcph-port.txt"
cmp_text "GCP AIP_HEALTH_ROUTE alias" "$WORK/gcph-ref.txt" "$WORK/gcph-port.txt"

PREDICT='{"instances":[{"@requestFormat":"chatCompletions","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":6,"temperature":0,"seed":42}]}'
curl -s -X POST "http://127.0.0.1:8337/predict" -H 'content-type: application/json' -d "$PREDICT" > "$WORK/gcp-ref.json"
curl -s -X POST "http://127.0.0.1:8338/predict" -H 'content-type: application/json' -d "$PREDICT" > "$WORK/gcp-port.json"
# the generated reply must match; the volatile/pre-existing fields of the
# inner /v1/chat/completions answer (created/id/system_fingerprint/timings)
# are normalized away — they are not GCP-surface properties
python3 - "$WORK/gcp-ref.json" "$WORK/gcp-port.json" <<'PYC'
import json, sys
VOLATILE = {"created", "id", "system_fingerprint", "timings"}
def scrub(o):
    if isinstance(o, dict):
        return {k: scrub(v) for k, v in o.items() if k not in VOLATILE}
    if isinstance(o, list):
        return [scrub(v) for v in o]
    return o
a = scrub(json.load(open(sys.argv[1])))
b = scrub(json.load(open(sys.argv[2])))
sys.exit(0 if a == b else 1)
PYC
if [ $? -eq 0 ]; then echo "MATCH  GCP /predict chatCompletions alias"; PASS=$((PASS+1))
else echo "DIFF   GCP /predict chatCompletions alias"; FAIL=$((FAIL+1)); fi

BAD='{"instances":[{"@requestFormat":"noSuchRoute","x":1},{"bad":1}]}'
curl -s -X POST "http://127.0.0.1:8337/predict" -H 'content-type: application/json' -d "$BAD" > "$WORK/gcpb-ref.json"
curl -s -X POST "http://127.0.0.1:8338/predict" -H 'content-type: application/json' -d "$BAD" > "$WORK/gcpb-port.json"
cmp_json "GCP /predict error instances" "$WORK/gcpb-ref.json" "$WORK/gcpb-port.json"
kill "${PIDS[@]}"; PIDS=(); sleep 1

# --- 6: cors-proxy against a loopback echo server ---
python3 - "$WORK" <<'PYEOF' &
import http.server, sys, threading
class Echo(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get('content-length', 0)))
        marker = self.headers.get('x-llama-marker', 'none')
        with open(sys.argv[1] + '/echo-seen.txt', 'wb') as f:
            f.write(body + b'|' + marker.encode())
        self.send_response(200)
        self.send_header('content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'echo:' + body)
    def log_message(self, *a):
        pass
http.server.HTTPServer(('127.0.0.1', 8390), Echo).serve_forever()
PYEOF
ECHO_PID=$!; PIDS+=("$ECHO_PID")
sleep 1

start_server "$REF"  8339 --ui-mcp-proxy || exit 2
start_server "$PORT_BIN" 8340 --ui-mcp-proxy || exit 2

curl -s -X POST "http://127.0.0.1:8339/cors-proxy?url=http://127.0.0.1:8390/echo" \
     -H 'x-llama-server-proxy-header-x-llama-marker: passed' -d 'proxy-body' > "$WORK/cp-ref.txt"
cp "$WORK/echo-seen.txt" "$WORK/cp-seen-ref.txt" 2>/dev/null || printf '(ref)no-capture' > "$WORK/cp-seen-ref.txt"
rm -f "$WORK/echo-seen.txt"
curl -s -X POST "http://127.0.0.1:8340/cors-proxy?url=http://127.0.0.1:8390/echo" \
     -H 'x-llama-server-proxy-header-x-llama-marker: passed' -d 'proxy-body' > "$WORK/cp-port.txt"
cp "$WORK/echo-seen.txt" "$WORK/cp-seen-port.txt" 2>/dev/null || printf '(port)no-capture' > "$WORK/cp-seen-port.txt"

cmp_text "cors-proxy POST body forward" "$WORK/cp-ref.txt" "$WORK/cp-port.txt"
cmp_text "cors-proxy header pass-through (target view)" "$WORK/cp-seen-ref.txt" "$WORK/cp-seen-port.txt"

# the reference's invalid-scheme surface
curl -s -X POST "http://127.0.0.1:8339/cors-proxy?url=ftp://x/" -d '' > "$WORK/cpe-ref.txt"
curl -s -X POST "http://127.0.0.1:8340/cors-proxy?url=ftp://x/" -d '' > "$WORK/cpe-port.txt"
cmp_text "cors-proxy unsupported scheme" "$WORK/cpe-ref.txt" "$WORK/cpe-port.txt"

echo
echo "server tools/mcp/gcp/cors parity: $PASS MATCH, $FAIL DIFF (of $((PASS+FAIL)))"
[ "$FAIL" -eq 0 ]
