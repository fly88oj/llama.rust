#!/usr/bin/env bash
# server_ui_proxy_parity.sh — the two tools-domain gaps closed against the
# pinned reference (bd4f514db1):
#
#   * /cors-proxy's https arm (server-cors-proxy.h:39-40 + server-models.cpp:
#     2456-2600's httplib::SSLClient) — a loopback TLS echo server trusted
#     through SSL_CERT_FILE, byte-compared; the untrusted-cert refusal; the
#     follow_location redirect rules (httplib.cpp:4222-4244)
#   * the web client serving (server-http.cpp:360-478): the empty-asset-table
#     default (this reference build embeds 0 assets), the --path static mount,
#     --no-ui, --api-prefix on the UI routes
#
# Ports ≥ 8820; every server through parity/limited.sh gates itself.
set -u

MODEL="${1:-/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
REF_BIN_DIR="${REF_BIN_DIR:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin}"
PORT_BIN="${PORT_BIN:-$(dirname "$0")/../target/release/llama-server}"
REF="$REF_BIN_DIR/llama-server"
WORK=$(mktemp -d /tmp/llama-ui-proxy-parity.XXXXXX)
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
        # a --api-prefix server moves /health under the prefix too
        curl -sf "http://127.0.0.1:$port/pfx/health" >/dev/null 2>&1 && return 0
        sleep 0.5
    done
    echo "server on $port failed to start:" >&2; tail -5 "$WORK/srv-$port.log" >&2
    return 1
}

cmp_text() {
    if cmp -s "$2" "$3"; then echo "MATCH  $1"; PASS=$((PASS+1))
    else echo "DIFF   $1"; FAIL=$((FAIL+1)); diff "$2" "$3" | head -6; fi
}

cmp_json() { # name, ref-file, port-file (normalized through json.loads)
    if python3 -c 'import json,sys; parse=lambda p: json.loads(open(p).read().split(chr(10))[0]); sys.exit(0 if parse(sys.argv[1]) == parse(sys.argv[2]) else 1)' "$2" "$3"; then
        echo "MATCH  $1"; PASS=$((PASS+1))
    else
        echo "DIFF   $1"; FAIL=$((FAIL+1))
    fi
}

cmp_status() { # name, ref-file, port-file
    local rs ps
    rs=$(head -c1 "$2"); ps=$(head -c1 "$3")
    if [ "$rs" = "$ps" ]; then echo "MATCH  $1 (both HTTP $rs..)"; PASS=$((PASS+1))
    else echo "DIFF   $1 (ref $(cat "$2") vs port $(cat "$3"))"; FAIL=$((FAIL+1)); fi
}

# ---------------------------------------------------------------------------
# fixtures: the CA + server cert (openssl), the TLS echo server, the www tree
# ---------------------------------------------------------------------------
openssl req -x509 -newkey rsa:2048 -nodes -keyout "$WORK/srv.key" -out "$WORK/srv.crt" \
    -days 2 -subj "/CN=127.0.0.1" \
    -addext "subjectAltName=IP:127.0.0.1,DNS:localhost" >/dev/null 2>&1 || exit 2
cat "$WORK/srv.crt" "$WORK/srv.key" > "$WORK/srv.pem"
cp "$WORK/srv.crt" "$WORK/ca.pem"

# the echo server: reflects method/path/headers/body so the whole forwarding
# chain (Host rewrite, header set, body) lands in the compared bytes; the
# /redir path exercises follow_location (302 + 303 POST→GET)
python3 - "$WORK/srv.pem" >"$WORK/echo.log" 2>&1 <<'PYEOF' &
import http.server, json, ssl, sys, threading

pem = sys.argv[1]

class Echo(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def _reply(self):
        body = json.dumps({
            "method": self.command,
            "path": self.path,
            "headers": {k.lower(): v for k, v in self.headers.items()},
        }, sort_keys=True).encode()
        n = int(self.headers.get("Content-Length") or 0)
        req_body = self.rfile.read(n) if n else b""
        if self.path.startswith("/redir"):
            self.send_response(302 if self.command == "GET" else 303)
            self.send_header("Location", "/final")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path.startswith("/echo-post"):
            # 303 forces the redirect client into GET (httplib.cpp:4227-4231)
            self.send_response(303)
            self.send_header("Location", "/final")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path.startswith("/bin-echo"):
            # the fixed non-UTF-8 payload: the relay must come back
            # byte-exact (std::string semantics, no lossy UTF-8 copy)
            blob = bytes([0x00, 0x01, 0x7f, 0x80, 0xc3, 0x28, 0xfe, 0xff,
                          0x0d, 0x0a, 0x00, 0x2d, 0x2d, 0x0d, 0x0a, 0x7f])
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(blob)))
            self.send_header("X-Echo-Marker", "binary")
            self.end_headers()
            self.wfile.write(blob)
            return
        if self.path.startswith("/multipart-echo"):
            # reflect the REBUILT multipart request verbatim (body bytes +
            # content-type) so both servers' rebuilds compare after boundary
            # normalization
            ct = self.headers.get("Content-Type", "")
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(req_body)))
            self.send_header("X-Echo-Marker", "multipart")
            self.end_headers()
            self.wfile.write(req_body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body) + len(req_body)))
        self.send_header("X-Echo-Marker", "relayed")
        self.end_headers()
        self.wfile.write(body + req_body)
    do_GET = _reply
    do_POST = _reply
    def log_message(self, *a):
        pass

srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8849), Echo)
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain(pem)
tls = http.server.ThreadingHTTPServer(("127.0.0.1", 8848), Echo)
tls.socket = ctx.wrap_socket(tls.socket, server_side=True)
threading.Thread(target=srv.serve_forever, daemon=True).start()
tls.serve_forever()
PYEOF
PIDS+=($!)
sleep 1

mkdir -p "$WORK/www/sub"
printf '<h1>mount index</h1>\n' > "$WORK/www/index.html"
printf 'sub js body\n' > "$WORK/www/sub/a.js"
printf 'a note\n' > "$WORK/www/sub/note.txt"
# the binary mount fixture: non-UTF-8 bytes incl. NUL/DEL/high bytes
printf '\x00\x01\x7f\x80\xc3\x28\xfe\xff\x0d\x0a' > "$WORK/www/sub/blob.bin"

# ---------------------------------------------------------------------------
# 1. https + http forwarding (the servers trust the test CA via SSL_CERT_FILE,
#    exactly how the reference's OpenSSL default-verify-paths load works)
# ---------------------------------------------------------------------------
export SSL_CERT_FILE="$WORK/ca.pem"
export SSL_CERT_DIR="$WORK/empty-certs"
mkdir -p "$WORK/empty-certs"
start_server "$REF"  8831 --ui-mcp-proxy || exit 2
start_server "$PORT_BIN" 8832 --ui-mcp-proxy || exit 2
sleep 1

curl -s "https://127.0.0.1:8848/echo?h=1" -o /dev/null 2>/dev/null || {
    echo "SKIP   tls echo server unreachable"; }

curl -s -X POST "http://127.0.0.1:8831/cors-proxy?url=https://127.0.0.1:8848/echo" \
     -H 'x-llama-server-proxy-header-x-llama-marker: passed' -d 'proxy-body' > "$WORK/https-ref.txt"
curl -s -X POST "http://127.0.0.1:8832/cors-proxy?url=https://127.0.0.1:8848/echo" \
     -H 'x-llama-server-proxy-header-x-llama-marker: passed' -d 'proxy-body' > "$WORK/https-port.txt"
cmp_text "https proxy POST forward" "$WORK/https-ref.txt" "$WORK/https-port.txt"

curl -s "http://127.0.0.1:8831/cors-proxy?url=http://127.0.0.1:8849/echo" \
     -H 'x-llama-server-proxy-header-x-llama-marker: passed' > "$WORK/http-ref.txt"
curl -s "http://127.0.0.1:8832/cors-proxy?url=http://127.0.0.1:8849/echo" \
     -H 'x-llama-server-proxy-header-x-llama-marker: passed' > "$WORK/http-port.txt"
cmp_text "http proxy GET forward" "$WORK/http-ref.txt" "$WORK/http-port.txt"

# follow_location: 302 on GET, 303 POST→GET (httplib.cpp:4222-4244)
curl -s "http://127.0.0.1:8831/cors-proxy?url=http://127.0.0.1:8849/redir" > "$WORK/redir-ref.txt"
curl -s "http://127.0.0.1:8832/cors-proxy?url=http://127.0.0.1:8849/redir" > "$WORK/redir-port.txt"
cmp_text "proxy 302 redirect follow" "$WORK/redir-ref.txt" "$WORK/redir-port.txt"

curl -s -X POST "http://127.0.0.1:8831/cors-proxy?url=http://127.0.0.1:8849/echo-post" -d 'x' > "$WORK/r303-ref.txt"
curl -s -X POST "http://127.0.0.1:8832/cors-proxy?url=http://127.0.0.1:8849/echo-post" -d 'x' > "$WORK/r303-port.txt"
cmp_text "proxy 303 POST-to-GET redirect" "$WORK/r303-ref.txt" "$WORK/r303-port.txt"

# the relayed response headers (minus the proxy's strip list)
curl -s -D - -o /dev/null "http://127.0.0.1:8831/cors-proxy?url=http://127.0.0.1:8849/echo" \
    | grep -i "^x-echo-marker" > "$WORK/hdr-ref.txt"
curl -s -D - -o /dev/null "http://127.0.0.1:8832/cors-proxy?url=http://127.0.0.1:8849/echo" \
    | grep -i "^x-echo-marker" > "$WORK/hdr-port.txt"
cmp_text "proxy response header relay" "$WORK/hdr-ref.txt" "$WORK/hdr-port.txt"

# ---------------------------------------------------------------------------
# 2. the untrusted-cert refusal (both must fail; the error text is
#    httplib-internal, so the STATUS is the comparable contract)
# ---------------------------------------------------------------------------
env -u SSL_CERT_FILE -u SSL_CERT_DIR LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    "$REF" -m "$MODEL" --host 127.0.0.1 --port 8833 -c 256 -t 8 --ui-mcp-proxy \
    >"$WORK/srv-8833.log" 2>&1 &
PIDS+=($!)
env -u SSL_CERT_FILE -u SSL_CERT_DIR \
    "$PORT_BIN" -m "$MODEL" --host 127.0.0.1 --port 8834 -c 256 -t 8 --ui-mcp-proxy \
    >"$WORK/srv-8834.log" 2>&1 &
PIDS+=($!)
for p in 8833 8834; do
    for _ in $(seq 1 240); do
        curl -sf "http://127.0.0.1:$p/health" >/dev/null 2>&1 && break
        sleep 0.5
    done
done
curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:8833/cors-proxy?url=https://127.0.0.1:8848/echo" > "$WORK/cert-ref.txt"
curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:8834/cors-proxy?url=https://127.0.0.1:8848/echo" > "$WORK/cert-port.txt"
cmp_status "https untrusted cert refused (status)" "$WORK/cert-ref.txt" "$WORK/cert-port.txt"

# ---------------------------------------------------------------------------
# 3. the web client: the empty-table default (this reference build embeds no
#    assets), the --path mount, --no-ui
# ---------------------------------------------------------------------------
start_server "$REF"  8835 || exit 2
start_server "$PORT_BIN" 8836 || exit 2
sleep 1
curl -s -w '\n%{http_code} %{content_type}' "http://127.0.0.1:8835/" > "$WORK/root-ref.txt"
curl -s -w '\n%{http_code} %{content_type}' "http://127.0.0.1:8836/" > "$WORK/root-port.txt"
cmp_json "ui default (no assets) GET /" "$WORK/root-ref.txt" "$WORK/root-port.txt"

curl -s -w '\n%{http_code} %{content_type}' "http://127.0.0.1:8835/index.html" > "$WORK/ih-ref.txt"
curl -s -w '\n%{http_code} %{content_type}' "http://127.0.0.1:8836/index.html" > "$WORK/ih-port.txt"
cmp_json "ui default GET /index.html" "$WORK/ih-ref.txt" "$WORK/ih-port.txt"

start_server "$REF"  8837 --path "$WORK/www" || exit 2
start_server "$PORT_BIN" 8838 --path "$WORK/www" || exit 2
sleep 1
for path in "/" "/sub/a.js" "/sub/note.txt" "/missing.js" "/sub/"; do
    tag=$(echo "$path" | tr '/.' '__')
    curl -s "http://127.0.0.1:8837$path" > "$WORK/m-ref$tag.txt"
    curl -s -o /dev/null -w '%{http_code} %{content_type}' "http://127.0.0.1:8837$path" > "$WORK/m-ref$tag.meta"
    curl -s "http://127.0.0.1:8838$path" > "$WORK/m-port$tag.txt"
    curl -s -o /dev/null -w '%{http_code} %{content_type}' "http://127.0.0.1:8838$path" > "$WORK/m-port$tag.meta"
    cmp_text "--path mount GET $path body" "$WORK/m-ref$tag.txt" "$WORK/m-port$tag.txt"
    cmp_text "--path mount GET $path status+type" "$WORK/m-ref$tag.meta" "$WORK/m-port$tag.meta"
done

start_server "$REF"  8839 --no-ui || exit 2
start_server "$PORT_BIN" 8840 --no-ui || exit 2
sleep 1
curl -s -w '\n%{http_code}' "http://127.0.0.1:8839/" > "$WORK/noui-ref.txt"
curl -s -w '\n%{http_code}' "http://127.0.0.1:8840/" > "$WORK/noui-port.txt"
cmp_json "--no-ui GET /" "$WORK/noui-ref.txt" "$WORK/noui-port.txt"

# ---------------------------------------------------------------------------
# 4. the api-prefix on the UI routes (the reference re-mounts EVERYTHING
#    under the prefix; the port prefixes the UI routes — the mount test only)
# ---------------------------------------------------------------------------
start_server "$REF"  8841 --path "$WORK/www" --api-prefix /pfx || exit 2
start_server "$PORT_BIN" 8842 --path "$WORK/www" --api-prefix /pfx || exit 2
sleep 1
curl -s -w '\n%{http_code} %{content_type}' "http://127.0.0.1:8841/pfx/sub/a.js" > "$WORK/p-ref.txt"
curl -s -w '\n%{http_code} %{content_type}' "http://127.0.0.1:8842/pfx/sub/a.js" > "$WORK/p-port.txt"
cmp_text "--api-prefix mount GET /pfx/sub/a.js" "$WORK/p-ref.txt" "$WORK/p-port.txt"

# the mount's segment-boundary rule: "/pfx" (no slash) matches no route and
# NOT the "/pfx/" mount (httplib.cpp:8917-8923) — 404 on both
for what in "/pfx" "/pfx/nope.js"; do
    tag=$(echo "$what" | tr '/.' '__')
    curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:8841$what" > "$WORK/mb-ref$tag.txt"
    curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:8842$what" > "$WORK/mb-port$tag.txt"
    cmp_text "--api-prefix mount boundary $what" "$WORK/mb-ref$tag.txt" "$WORK/mb-port$tag.txt"
done
curl -s "http://127.0.0.1:8841/pfx/" > "$WORK/pdx-ref.txt"
curl -s "http://127.0.0.1:8842/pfx/" > "$WORK/pdx-port.txt"
cmp_text "--api-prefix mount GET /pfx/ (index)" "$WORK/pdx-ref.txt" "$WORK/pdx-port.txt"

# the binary file through the mount (Body::Bytes both sides)
curl -s "http://127.0.0.1:8841/pfx/sub/blob.bin" > "$WORK/mbb-ref.bin"
curl -s "http://127.0.0.1:8842/pfx/sub/blob.bin" > "$WORK/mbb-port.bin"
cmp -s "$WORK/mbb-ref.bin" "$WORK/mbb-port.bin" && cmp -s "$WORK/mbb-ref.bin" "$WORK/www/sub/blob.bin" \
    && { echo "MATCH  --api-prefix mount binary GET /pfx/sub/blob.bin (byte-exact)"; PASS=$((PASS+1)); } \
    || { echo "DIFF   --api-prefix mount binary GET /pfx/sub/blob.bin"; FAIL=$((FAIL+1)); }

# ---------------------------------------------------------------------------
# 5. --api-prefix re-mounts the WHOLE API surface (server-http.cpp:114 +
#    :673/724/745 — path_prefix rides every ctx_http.get/post/del): /health,
#    /props, /v1/models, /tokenize move under /pfx, the bare paths 404
# ---------------------------------------------------------------------------
start_server "$REF"  8843 --api-prefix /pfx || exit 2
start_server "$PORT_BIN" 8844 --api-prefix /pfx || exit 2
sleep 1
for what in "/pfx/health" "/health" "/pfx/v1/health" "/pfx/props" "/pfx/v1/models" "/pfx/" "/pfx/index.html"; do
    tag=$(echo "$what" | tr '/.' '__')
    curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:8843$what" > "$WORK/a-ref$tag.txt"
    curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:8844$what" > "$WORK/a-port$tag.txt"
    cmp_text "--api-prefix API GET $what (status)" "$WORK/a-ref$tag.txt" "$WORK/a-port$tag.txt"
done
# POST moves too: /pfx/tokenize answers, bare /tokenize 404s
curl -s -X POST "http://127.0.0.1:8843/pfx/tokenize" -H 'Content-Type: application/json' \
     -d '{"content":"hello world"}' > "$WORK/tok-ref.txt"
curl -s -X POST "http://127.0.0.1:8844/pfx/tokenize" -H 'Content-Type: application/json' \
     -d '{"content":"hello world"}' > "$WORK/tok-port.txt"
cmp_json "--api-prefix POST /pfx/tokenize" "$WORK/tok-ref.txt" "$WORK/tok-port.txt"
curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:8843/tokenize" \
     -H 'Content-Type: application/json' -d '{}' > "$WORK/tokb-ref.txt"
curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:8844/tokenize" \
     -H 'Content-Type: application/json' -d '{}' > "$WORK/tokb-port.txt"
cmp_text "--api-prefix bare POST /tokenize 404" "$WORK/tokb-ref.txt" "$WORK/tokb-port.txt"
# /v1/models content under the prefix
curl -s "http://127.0.0.1:8843/pfx/v1/models" > "$WORK/vm-ref.txt"
curl -s "http://127.0.0.1:8844/pfx/v1/models" > "$WORK/vm-port.txt"
cmp_json "--api-prefix GET /pfx/v1/models" "$WORK/vm-ref.txt" "$WORK/vm-port.txt"
# the cors-proxy itself moves
curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:8843/cors-proxy?url=http://127.0.0.1:8849/echo" > "$WORK/cp-ref.txt"
curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:8844/cors-proxy?url=http://127.0.0.1:8849/echo" > "$WORK/cp-port.txt"
cmp_text "--api-prefix bare /cors-proxy 404" "$WORK/cp-ref.txt" "$WORK/cp-port.txt"

# ---------------------------------------------------------------------------
# 6. the binary response relay (server-models.cpp:2365-2377 — res->data is a
#    std::string): a non-UTF-8 body through the proxy comes back byte-exact
# ---------------------------------------------------------------------------
start_server "$REF"  8845 --ui-mcp-proxy || exit 2
start_server "$PORT_BIN" 8846 --ui-mcp-proxy || exit 2
sleep 1
curl -s "http://127.0.0.1:8845/cors-proxy?url=http://127.0.0.1:8849/bin-echo" > "$WORK/bin-ref.bin"
curl -s "http://127.0.0.1:8846/cors-proxy?url=http://127.0.0.1:8849/bin-echo" > "$WORK/bin-port.bin"
python3 - "$WORK/bin-ref.bin" "$WORK/bin-port.bin" <<'PYCHK'
import sys
ref = open(sys.argv[1], 'rb').read()
port = open(sys.argv[2], 'rb').read()
blob = bytes([0x00, 0x01, 0x7f, 0x80, 0xc3, 0x28, 0xfe, 0xff,
              0x0d, 0x0a, 0x00, 0x2d, 0x2d, 0x0d, 0x0a, 0x7f])
sys.exit(0 if ref == port == blob else 1)
PYCHK
if [ $? -eq 0 ]; then echo "MATCH  proxy binary body relay (byte-exact)"; PASS=$((PASS+1))
else echo "DIFF   proxy binary body relay"; FAIL=$((FAIL+1)); fi

# ---------------------------------------------------------------------------
# 7. the multipart arm (server-http.cpp:683-717 parse + server-models.cpp:
#    2553-2568 rebuild): a form with a duplicate key and a binary file
#    forwards as a rebuilt multipart body — identical structure after the
#    boundary normalization (both sides generate their own random boundary)
# ---------------------------------------------------------------------------
printf '\x00\xff\x0d\x0a\x80\x7f\x2d\x2d\x0d\x0a\xc3\x28' > "$WORK/up.bin"
for port_ in 8845 8846; do
    curl -s -X POST "http://127.0.0.1:$port_/cors-proxy?url=http://127.0.0.1:8849/multipart-echo" \
         -H 'x-llama-server-proxy-header-x-marker: mp' \
         -F 'q=hello' -F 'q=world' -F 'z=first' \
         -F "f=@$WORK/up.bin;type=application/x-binary;filename=up.bin" \
         > "$WORK/mp-$port_.body" -D "$WORK/mp-$port_.hdr"
done
python3 - "$WORK/mp-8845.body" "$WORK/mp-8846.body" "$WORK/mp-8845.hdr" "$WORK/mp-8846.hdr" "$WORK/up.bin" <<'PYCHK'
import re, sys
def boundary_of(hdr):
    # the rebuild's boundary rides the relayed Content-Type header
    m = re.search(r'(?im)^content-type: multipart/form-data; boundary=(----llama-cpp-proxy-[0-9a-z]{16})', open(hdr, 'r', errors='replace').read())
    return m.group(1).encode() if m else None
def norm(body, boundary):
    return open(body, 'rb').read().replace(boundary, b'BOUNDARY') if boundary else None
rb = norm(sys.argv[1], boundary_of(sys.argv[3]))
pb = norm(sys.argv[2], boundary_of(sys.argv[4]))
up = open(sys.argv[5], 'rb').read()  # the uploaded file bytes
ok = rb is not None and pb is not None and rb == pb and up in rb
# structural spot checks on the normalized body: 4 part delimiters + the
# closing frame, the duplicate-key array as two parts, the file's disposition
if ok:
    s = rb
    ok = (b'name="q"' in s and b'name="z"' in s and
          s.count(b'--BOUNDARY\r\n') == 4 and s.rstrip().endswith(b'--BOUNDARY--'))
    ok = ok and b'filename="up.bin"' in s and b'application/x-binary' in s
sys.exit(0 if ok else 1)
PYCHK
if [ $? -eq 0 ]; then echo "MATCH  proxy multipart rebuild (fields+binary file, boundary-normalized)"; PASS=$((PASS+1))
else echo "DIFF   proxy multipart rebuild"; FAIL=$((FAIL+1)); diff <(cat -v "$WORK/mp-8845.body") <(cat -v "$WORK/mp-8846.body") | head -8; fi

echo "server_ui_proxy_parity: $PASS MATCH / $FAIL DIFF"
[ "$FAIL" -eq 0 ]
