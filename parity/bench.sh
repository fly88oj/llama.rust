#!/usr/bin/env bash
# bench.sh — CPU prefill / generation throughput: Rust port vs reference llama.cpp.
#
# Units follow llama-bench: prefill (pp) and generation (tg) tokens per second,
# reported for a fresh process/server per measurement (no slot reuse).
#
#   Rust side   : ./target/release/llama-cli  (prints " [prompt: N tokens ... t/s]"
#                 and " [gen: N tokens ... t/s]")
#   Reference   : bin/llama-server + one /completion request (the PARITY.md
#                 protocol: fresh server per model, first request only), whose
#                 `timings.prompt_per_second` / `timings.predicted_per_second`
#                 are the same numbers llama-bench prints.
#
# Both sides run with flash attention ON (each implementation's default) and
# 8 threads. The Rust side's lazy CPU_REPACK materialization (~3 s / 9.7 GiB for
# gpt-oss) is paid on the first forward, so one warm-up run precedes the
# measured reps; the reported figure is the best rep (steady state), the
# reference pays the same cost inside its model load.
#
# usage: parity/bench.sh [qwen|gptoss|all] [reps] [n_predict]
set -euo pipefail

WHICH="${1:-all}"
REPS="${2:-2}"
NPRED="${3:-16}"
THREADS="${THREADS:-8}"
CTX="${CTX:-2048}"
PORT="${PORT:-8793}"

REF_BIN="${REF_BIN:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin}"
QWEN="${QWEN:-/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
GPTOSS="${GPTOSS:-/home/jeffrey/.lmstudio/models/lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLI="$ROOT/target/release/llama-cli"

SHORT_PROMPT="The capital of France is"
# 64 tokens with the qwen2.5 tokenizer: the smallest prompt that makes the
# reference (and this port) take the *tiled* flash-attention kernel, since
# llama-cli's prefill batch is capped at 64 (larger prompts panic with
# "batch size N out of (0, 64]" before any kernel runs).
LONG_PROMPT="Attention is a mechanism that allows a model to weigh the relevance of different parts of its input when producing each element of its output. In practice this means that every token computes a score against every other token it is allowed to see, and those scores are normalized into a probability distribution. It matters a lot right now today."

[[ -x "$CLI" ]] || { echo "build first: cargo build --release -p llama-cli" >&2; exit 1; }

# the reference CLI hangs without a tty; the server path is the documented one
start_server() {
  local model="$1" log="$2"
  pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
  sleep 1
  "$REF_BIN/llama-server" -m "$model" -c "$CTX" -t "$THREADS" -fa on \
      --port "$PORT" --host 127.0.0.1 >"$log" 2>&1 &
  SRV_PID=$!
  for _ in $(seq 1 300); do
    # the body check matters: /health answers 503 {"error":"Loading model"}
    # while the model loads, and curl exits 0 on HTTP errors — without the
    # grep the first request fires mid-load (gpt-oss loads ~30 s) and reads
    # the 503 instead of the benchmark
    curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && return 0
    sleep 1
  done
  echo "reference server failed to start (see $log)" >&2
  return 1
}

stop_server() {
  [[ -n "${SRV_PID:-}" ]] && kill "$SRV_PID" 2>/dev/null || true
  pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
  SRV_PID=""
}

# reference: one fresh-server first request -> "prefill_t/s gen_t/s prompt_n"
ref_measure() {
  local model="$1" prompt="$2" log="$3"
  start_server "$model" "$log"
  curl -s -m 1800 "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
      -d "$(python3 - "$prompt" "$NPRED" <<'EOF'
import json, sys
print(json.dumps({"prompt": sys.argv[1], "n_predict": int(sys.argv[2]),
                  "temperature": 0, "cache_prompt": False}))
EOF
)" >"$log.json"
  stop_server
  python3 - "$log.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
t = d.get("timings", {})
print(f'{t.get("prompt_per_second", float("nan")):.1f} '
      f'{t.get("predicted_per_second", float("nan")):.1f} '
      f'{t.get("prompt_n", 0)} {t.get("predicted_n", 0)}')
EOF
}

# rust: best of `reps` runs (after one warm-up) -> "prefill_t/s gen_t/s"
rust_measure() {
  local model="$1" prompt="$2" extra="$3"
  local best_p=0 best_g=0 out
  # load-time warm-up: raise the warm-repack budget so gpt-oss (~9.7 GiB)
  # fully materializes at load like the reference's CPU_REPACK buffer
  # (default is a test-friendly 4 GiB — see ggml::compute::warm_repack)
  export LLAMA_RUST_REPACK_WARM_MAX_MB="${LLAMA_RUST_REPACK_WARM_MAX_MB:-16384}"
  for r in $(seq 0 "$REPS"); do
    out="$("$CLI" -m "$model" -p "$prompt" -n "$NPRED" -t "$THREADS" -c "$CTX" \
            --temp 0 --ignore-eos -fa on $extra 2>/dev/null | tr '\n' ' ')"
    local p g
    p="$(sed -n 's/.*\[prompt: [0-9]* tokens in [0-9.]*[a-z]* = \([0-9.]*\) t\/s\].*/\1/p' <<<"$out")"
    g="$(sed -n 's/.*\[gen: [0-9]* tokens in [0-9.]*s = \([0-9.]*\) t\/s\].*/\1/p' <<<"$out")"
    [[ -z "$p" || -z "$g" ]] && { echo "rust run failed: $out" >&2; return 1; }
    # run 0 is the warm-up (repack materialization); compare reps 1..N
    if awk "BEGIN{exit !($p > $best_p)}"; then best_p="$p"; fi
    if awk "BEGIN{exit !($g > $best_g)}"; then best_g="$g"; fi
  done
  echo "$best_p $best_g"
}

table=""
run_case() {
  local label="$1" model="$2" prompt="$3"
  echo "=== $label ===" >&2
  local rp="" rg="" fp="" fg="" fn="" pn=""
  read -r rp rg <<<"$(rust_measure "$model" "$prompt" "" || echo '')"
  read -r fp fg fn pn <<<"$(ref_measure "$model" "$prompt" "/tmp/bench-ref-${label//[^a-z0-9]/_}" || echo '')"
  [[ -z "$rp" ]] && rp=0 && rg=0
  [[ -z "$fp" ]] && fp=0 && fg=0 && fn=0 && pn=0
  printf '%-28s %10s %10s %10s %10s\n' "$label" "mine pp" "ref pp" "mine tg" "ref tg" >&2
  printf '%-28s %10s %10s %10s %10s\n' "$label" "$rp" "$fp" "$rg" "$fg" >&2
  table+="$(printf '%-28s %10.1f %10.1f %6.2fx   %10.1f %10.1f %6.2fx   %5s/%s' \
      "$label" "$rp" "$fp" "$(awk "BEGIN{print $rp/$fp}")" "$rg" "$fg" "$(awk "BEGIN{print $rg/$fg}")" "$pn" "$fn")"$'\n'
}

if [[ "$WHICH" == "qwen" || "$WHICH" == "all" ]]; then
  run_case "qwen2.5-0.5b pp5" "$QWEN" "$SHORT_PROMPT"
  run_case "qwen2.5-0.5b pp64" "$QWEN" "$LONG_PROMPT"
fi
if [[ "$WHICH" == "gptoss" || "$WHICH" == "all" ]]; then
  run_case "gpt-oss-20b pp5" "$GPTOSS" "$SHORT_PROMPT"
  run_case "gpt-oss-20b pp64" "$GPTOSS" "$LONG_PROMPT"
fi

echo
printf '%-28s %10s %10s %8s   %10s %10s %8s   %s\n' \
  case "mine pp t/s" "ref pp t/s" "ratio" "mine tg t/s" "ref tg t/s" "ratio" "ref pp/tg n"
echo "$table"