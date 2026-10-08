#!/usr/bin/env bash
# Reference-side distribution probe for the falcon synthetic model (the only
# batch-2 arch that diverges from the reference — see PARITY.md).
#
#   parity/falcon_probe.sh            # uses /tmp/arch-batch2/falcon-synth.gguf
#
# One *fresh* reference llama-server, then:
#   1. the standard parity request (greedy, n_predict 16) with logprobs 20 ->
#      /tmp/parity-falcon-off-ref.json and the greedy token ids ->
#      /tmp/arch-batch2/falcon-ref-tokens.txt
#   2. one `prompt = [token ids]` request per line of
#      /tmp/arch-batch2/falcon-probe-contexts.txt (a token-array prompt is
#      evaluated as-is, so the port can be teacher-forced on any context) ->
#      REF lines on stdout
#
# The port side is `cargo test --release -p llama --test arch_batch2_e2e --
# --ignored --nocapture falcon_teacher_force_probe` (prints A/CTX lines); pair
# them with parity/falcon_probe_cmp.py.
set -uo pipefail
cd "$(dirname "$0")/.."

REF="${REF:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin}"
MODEL="${MODEL:-/tmp/arch-batch2/falcon-synth.gguf}"
PORT="${PORT:-8794}"
N="${N:-16}"

pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
sleep 2
"$REF/llama-server" -m "$MODEL" -c 512 -t 8 -fa off --port "$PORT" --host 127.0.0.1 \
  >"/tmp/parity-falcon-probe-server.log" 2>&1 &
SRV_PID=$!
trap 'kill "$SRV_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 300); do
  curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && break
  sleep 1
done

# 1. the standard greedy request, exactly like parity/run_cli_arch_parity.sh
python3 - "$PORT" "$N" <<'PY'
import json, sys, urllib.request
port, n = sys.argv[1], int(sys.argv[2])
body = json.dumps({
    "prompt": "The capital of France is",
    "n_predict": n, "temperature": 0, "logprobs": 20, "n_probs": 20,
    "cache_prompt": False, "top_k": 40, "top_p": 0.95, "min_p": 0.05,
}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                             headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req))
open("/tmp/parity-falcon-probe-ref.json", "w").write(json.dumps(r))
ids = [p["id"] for p in r.get("completion_probabilities", [])]
open("/tmp/arch-batch2/falcon-ref-tokens.txt", "w").write(" ".join(map(str, ids)))
print("ref greedy tokens:", ids)
PY

# 2. one token-array request per context line
if [ -f /tmp/arch-batch2/falcon-probe-contexts.txt ]; then
  python3 - "$PORT" <<'PY'
import json, sys, urllib.request
port = sys.argv[1]
for i, line in enumerate(open("/tmp/arch-batch2/falcon-probe-contexts.txt")):
    toks = [int(x) for x in line.split()]
    if not toks:
        continue
    body = json.dumps({"prompt": toks, "n_predict": 2, "temperature": 0,
                       "logprobs": 20, "n_probs": 20, "cache_prompt": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req))
    cp = r.get("completion_probabilities") or []
    top = cp[0].get("top_logprobs", [])[:5] if cp else []
    print(f"REF {i} n={r.get('tokens_evaluated')} argmax {cp[0]['id'] if cp else -1} top5 "
          + " ".join(f"({e['id']},{e['logprob']:.4f})" for e in top))
PY
fi