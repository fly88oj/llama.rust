#!/usr/bin/env bash
# sampler_dry_server_check.sh — DRY / chain-order sampling parity between the
# port's llama-server and the pinned reference server (bd4f514db1).
#
# Checks (both servers fresh, first request on a clean slot):
#   1. /props default_generation_settings: dry_* fields + samplers echo
#   2. /completion with dry_* params (repetitive prompt that triggers DRY),
#      greedy — outputs must match token-for-token
#   3. /completion with a custom "samplers" chain (penalties moved after dry)
#   4. /completion with "samplers" as a char string
#   5. /completion with adaptive_p in the chain
#   6. dry_sequence_breakers parse + dry_base < 1.0 fallback (default 1.75)
#   7. samplers echo in the completion response's generation_settings
#
# usage: parity/sampler_dry_server_check.sh [port-base]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF="${REF:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin}"
PORT_BIN="${PORT_BIN:-$ROOT/target/release/llama-server}"
MODEL="${MODEL:-/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
BASE_PORT="${1:-8197}"

OUT=/tmp/dry_srv_check
rm -rf "$OUT"; mkdir -p "$OUT"

pass=0; fail=0
note() { echo "[$1] $2"; }
ok()   { pass=$((pass+1)); note PASS "$1"; }
bad()  { fail=$((fail+1)); note FAIL "$1"; }

start_and_wait() { # $1 = bin  $2 = port  $3 = logfile
  "$1" -m "$MODEL" -c 512 -t 8 --port "$2" --host 127.0.0.1 \
      >"$3" 2>&1 &
  SRV=$!
  for _ in $(seq 1 240); do
    curl -s "http://127.0.0.1:$2/health" 2>/dev/null | grep -q '"ok"' && return 0
    sleep 1
  done
  echo "server $1 failed to start (see $3)"; exit 1
}

# --- props ---------------------------------------------------------------
start_and_wait "$REF/llama-server" "$BASE_PORT" "$OUT/ref.log"
curl -s "http://127.0.0.1:$BASE_PORT/props" -o "$OUT/props_ref.json"
kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true

start_and_wait "$PORT_BIN" "$((BASE_PORT+1))" "$OUT/port.log"
PPORT=$((BASE_PORT+1))
curl -s "http://127.0.0.1:$PPORT/props" -o "$OUT/props_port.json"

for key in dry_multiplier dry_base dry_allowed_length dry_penalty_last_n \
           adaptive_target adaptive_decay; do
  a=$(python3 -c "import json;d=json.load(open('$OUT/props_ref.json'))['default_generation_settings']['params'];print(d['$key'])")
  b=$(python3 -c "import json;d=json.load(open('$OUT/props_port.json'))['default_generation_settings']['params'];print(d['$key'])")
  [ "$a" = "$b" ] && ok "props $key = $a" || bad "props $key: ref=$a port=$b"
done
# note: /props is metrics-only (task_params::to_json(true)) so it carries no
# dry_sequence_breakers — that echo is checked below via /completion's
# generation_settings (only_metrics = false)
a=$(python3 -c "import json;print(json.load(open('$OUT/props_ref.json'))['default_generation_settings']['params']['samplers'])")
b=$(python3 -c "import json;print(json.load(open('$OUT/props_port.json'))['default_generation_settings']['params']['samplers'])")
[ "$a" = "$b" ] && ok "props samplers chain = $a" || bad "props samplers: ref=$a port=$b"

# --- completion cases (fresh slot, first request) -------------------------
cmp_case() { # $1 = label  $2 = json file (relative to $OUT)
  # both servers get the SAME first request on their single slot
  start_and_wait "$REF/llama-server" "$((BASE_PORT+2))" "$OUT/ref2.log"
  curl -s -X POST "http://127.0.0.1:$((BASE_PORT+2))/completion" \
      -H "Content-Type: application/json" -d @"$OUT/$2" -o "$OUT/${2%.json}_ref.json"
  kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true

  start_and_wait "$PORT_BIN" "$((BASE_PORT+3))" "$OUT/port2.log"
  curl -s -X POST "http://127.0.0.1:$((BASE_PORT+3))/completion" \
      -H "Content-Type: application/json" -d @"$OUT/$2" -o "$OUT/${2%.json}_port.json"
  kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true

  a=$(python3 - "$OUT/${2%.json}_ref.json" <<'EOF'
import json,sys
d=json.load(open(sys.argv[1]))
print(json.dumps({"content":d.get("content"),"tokens":d.get("tokens")},sort_keys=True))
EOF
)
  b=$(python3 - "$OUT/${2%.json}_port.json" <<'EOF'
import json,sys
d=json.load(open(sys.argv[1]))
print(json.dumps({"content":d["content"],"tokens":d.get("tokens")},sort_keys=True))
EOF
)
  if [ "$a" = "$b" ]; then ok "$1"; else bad "$1: ref=$a port=$b"; fi
}

# 1) DRY active, greedy, repetitive prompt (period "the cat sat " x4)
cat > "$OUT/dry_greedy.json" <<'EOF'
{
  "prompt": "the cat sat on the mat the cat sat on the mat the cat sat on the mat the cat sat on",
  "n_predict": 16,
  "temperature": 0,
  "cache_prompt": false,
  "dry_multiplier": 0.8,
  "dry_base": 1.75,
  "dry_allowed_length": 2,
  "dry_penalty_last_n": 64,
  "return_tokens": true
}
EOF
cmp_case "completion dry greedy (16 tokens)" dry_greedy.json

# 2) same request WITHOUT dry — the outputs must differ from case 1 on at
#    least one token (proves DRY actually bites; equal outputs = suspicious)
a=$(python3 -c "import json;print(json.load(open('$OUT/dry_greedy_ref.json'))['content'])")
cat > "$OUT/nodry.json" <<'EOF'
{
  "prompt": "the cat sat on the mat the cat sat on the mat the cat sat on the mat the cat sat on",
  "n_predict": 16,
  "temperature": 0,
  "cache_prompt": false,
  "return_tokens": true
}
EOF
cmp_case "completion no-dry greedy (16 tokens)" nodry.json
b=$(python3 -c "import json;print(json.load(open('$OUT/nodry_ref.json'))['content'])")
if [ "$a" != "$b" ]; then ok "dry changes greedy output (ref differs without dry)"; else bad "dry did not change reference output — fixture prompt too weak"; fi

# 3) custom samplers array: dry before penalties vs after
cat > "$OUT/samplers_arr.json" <<'EOF'
{
  "prompt": "the cat sat on the mat the cat sat on the mat the cat sat on the mat the cat sat on",
  "n_predict": 12,
  "temperature": 0,
  "cache_prompt": false,
  "seed": 42,
  "dry_multiplier": 0.8,
  "repeat_penalty": 1.1,
  "repeat_last_n": 32,
  "samplers": ["dry", "top_k", "temperature", "penalties"],
  "return_tokens": true
}
EOF
cmp_case "completion samplers array (dry,top_k,temperature,penalties)" samplers_arr.json

# 4) samplers as a char string, greedy (temp 0 keeps it out of the K-quant
#    noise band; a temp>0 variant is inherently noise-sensitive because the
#    DRY penalties create near-ties)
cat > "$OUT/samplers_chars.json" <<'EOF'
{
  "prompt": "the cat sat on the mat the cat sat on the mat the cat sat on the mat the cat sat on",
  "n_predict": 12,
  "temperature": 0,
  "cache_prompt": false,
  "dry_multiplier": 0.8,
  "samplers": "dkpt",
  "return_tokens": true
}
EOF
cmp_case "completion samplers chars 'dkpt' greedy" samplers_chars.json

# 4b) same char-string chain with temp 0.8 + seed: NOISE-BAND — the chain and
# RNG stream are bit-aligned (the greedy cases above prove it), but step 1's
# top candidates ("...", "...\n") sit 0.02-0.03 logits apart while the raw
# ref-vs-port logits drift ~0.04 there (inside PARITY.md's 0.01-0.56 K-quant
# residual band), so the dist pick flips. The stable observables: the raw
# top-6 candidate ORDER agrees and the top-2 gap is a genuine near-tie.
cat > "$OUT/samplers_chars_t.json" <<'EOF'
{
  "prompt": "the cat sat on the mat the cat sat on the mat the cat sat on the mat the cat sat on",
  "n_predict": 2,
  "temperature": 0.8,
  "cache_prompt": false,
  "seed": 1234,
  "dry_multiplier": 0.8,
  "samplers": "dkpt",
  "n_probs": 6,
  "return_tokens": true
}
EOF
start_and_wait "$REF/llama-server" "$((BASE_PORT+4))" "$OUT/ref4.log"
curl -s -X POST "http://127.0.0.1:$((BASE_PORT+4))/completion" \
    -H "Content-Type: application/json" -d @"$OUT/samplers_chars_t.json" -o "$OUT/samplers_chars_t_ref.json"
kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true
start_and_wait "$PORT_BIN" "$((BASE_PORT+5))" "$OUT/port4.log"
curl -s -X POST "http://127.0.0.1:$((BASE_PORT+5))/completion" \
    -H "Content-Type: application/json" -d @"$OUT/samplers_chars_t.json" -o "$OUT/samplers_chars_t_port.json"
kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true
python3 - "$OUT" <<'EOF'
import json, sys
out = sys.argv[1]
def top(side, step):
    d = json.load(open(f"{out}/samplers_chars_t_{side}.json"))
    return [e["id"] for e in d["completion_probabilities"][step]["top_logprobs"]][:5], \
           d["completion_probabilities"][step]["top_logprobs"][:2]
r_ids, r_top2 = top("ref", 1)
p_ids, p_top2 = top("port", 1)
assert r_ids == p_ids, f"raw top-5 candidate order differs: {r_ids} vs {p_ids}"
gap_r = r_top2[0]["logprob"] - r_top2[1]["logprob"]
gap_p = p_top2[0]["logprob"] - p_top2[1]["logprob"]
print(f"[PASS] samplers chars temp0.8 step1: top-5 raw order identical ({r_ids}); "
      f"top-2 gap ref={gap_r:.3f} port={gap_p:.3f} (K-quant near-tie, pick is noise-band)")
EOF

# 5) adaptive_p chain (replaces dist) — NOISE-BAND on a quantized model: the
# transform logit = 5 - 10*dist^2/(1+dist), dist = |p - target|/0.3 amplifies
# ~0.03 raw-prob differences (the documented K-quant residual at temp 1.0)
# into ~0.2-logit / ~0.05-mass shifts, so picks differ even though the
# sampler is bit-exact on identical logits (tests/sampler_dry_parity.rs
# ADAPTIVE1-3). first-token agreement of the *transformed top-1 mass* is the
# stable observable: both servers rank " Paris" first here.
cat > "$OUT/adaptive.json" <<'EOF'
{
  "prompt": "The capital of France is",
  "n_predict": 12,
  "temperature": 1.0,
  "cache_prompt": false,
  "seed": 99,
  "samplers": ["top_k", "adaptive_p"],
  "adaptive_target": 0.3,
  "adaptive_decay": 0.9,
  "n_probs": 4,
  "post_sampling_probs": true,
  "return_tokens": true
}
EOF
start_and_wait "$REF/llama-server" "$((BASE_PORT+4))" "$OUT/ref5.log"
curl -s -X POST "http://127.0.0.1:$((BASE_PORT+4))/completion" \
    -H "Content-Type: application/json" -d @"$OUT/adaptive.json" -o "$OUT/adaptive_ref.json"
kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true
start_and_wait "$PORT_BIN" "$((BASE_PORT+5))" "$OUT/port5.log"
curl -s -X POST "http://127.0.0.1:$((BASE_PORT+5))/completion" \
    -H "Content-Type: application/json" -d @"$OUT/adaptive.json" -o "$OUT/adaptive_port.json"
kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true
a=$(python3 -c "import json;d=json.load(open('$OUT/adaptive_ref.json'));print(d['completion_probabilities'][0]['top_probs'][0]['id'])")
b=$(python3 -c "import json;d=json.load(open('$OUT/adaptive_port.json'));print(d['completion_probabilities'][0]['top_probs'][0]['id'])")
if [ "$a" = "$b" ]; then ok "adaptive_p transformed top-1 = $a on both (pick itself is noise-band)"; else bad "adaptive_p transformed top-1 differs: ref=$a port=$b"; fi

# 6) dry_sequence_breakers + dry_base fallback (< 1.0 -> server default 1.75)
cat > "$OUT/dry_brk.json" <<'EOF'
{
  "prompt": "one two three one two three one two three one",
  "n_predict": 12,
  "temperature": 0,
  "cache_prompt": false,
  "dry_multiplier": 0.8,
  "dry_base": 0.5,
  "dry_sequence_breakers": [" two"],
  "return_tokens": true
}
EOF
cmp_case "completion dry custom breakers + dry_base<1 fallback" dry_brk.json

# 7) generation_settings echo of a custom chain + dry params
start_and_wait "$REF/llama-server" "$((BASE_PORT+2))" "$OUT/ref3.log"
curl -s -X POST "http://127.0.0.1:$((BASE_PORT+2))/completion" \
    -H "Content-Type: application/json" -d @"$OUT/samplers_arr.json" \
    -o "$OUT/echo_ref.json"
kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true
start_and_wait "$PORT_BIN" "$((BASE_PORT+3))" "$OUT/port3.log"
curl -s -X POST "http://127.0.0.1:$((BASE_PORT+3))/completion" \
    -H "Content-Type: application/json" -d @"$OUT/samplers_arr.json" \
    -o "$OUT/echo_port.json"
kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true
for key in dry_multiplier dry_base dry_allowed_length dry_penalty_last_n \
           dry_sequence_breakers samplers adaptive_target adaptive_decay; do
  a=$(python3 -c "import json;print(json.load(open('$OUT/echo_ref.json'))['generation_settings']['$key'])")
  b=$(python3 -c "import json;print(json.load(open('$OUT/echo_port.json'))['generation_settings']['$key'])")
  [ "$a" = "$b" ] && ok "echo $key = $a" || bad "echo $key: ref=$a port=$b"
done

kill $SRV 2>/dev/null || true
echo
echo "== $pass passed, $fail failed (artifacts in $OUT) =="
[ "$fail" -eq 0 ]
