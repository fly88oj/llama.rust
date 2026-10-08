#!/usr/bin/env bash
# Server arch sweep — the batch 1-14 wiring of crates/tools/llama-server's
# forward_weights. Per cell (one synthetic GGUF of an arch batch, or a real
# local model for the archs that have no synthetic file):
#
#   1. PORT-ONLY mode (default): the port server vs the port CLI on the same
#      file — fresh ./target/release/llama-server + first /completion
#      (temperature 0, n as given) vs LLAMA_RUST_DEBUG=1 llama-cli; the 16+
#      greedy tokens must be identical. This is the cheap complete check the
#      task runs on EVERY wired arch.
#   2. Reference mode (WITH_REF=1): a fresh *reference* server answers the
#      identical request first, and parity/server_arch_cmp.py compares the FULL
#      response (0 differing fields expected; logprobs in the PARITY.md band).
#
#   WITH_REF=1 parity/run_server_arch_parity.sh <cell...>   # reference sweep
#   parity/run_server_arch_parity.sh <cell...>              # port-vs-port check
#
# Cell names follow parity/arch_batch_parity.sh's per-batch lists (the same
# name->file mapping); the batch is selected exactly like arch_batch_parity.sh
# (ARCH_BATCH2=1 ... ARCH_BATCH14=1, default batch 1). The REAL_* variables
# override the batch-1 real-model cells (gptoss / gemma4 / granite-hybrid /
# lfm2 / qwen35).
#
# Artifacts: /tmp/srvarch-<cell>-{ref,rust}.json, -server{,-rust}.log, -cli.out
set -uo pipefail
cd "$(dirname "$0")/.."

REF="${REF:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin}"
PORT="${PORT:-8793}"
CTX="${CTX:-512}"
THREADS="${THREADS:-8}"
OUT="${OUT:-/tmp/srvarch}"
mkdir -p "$OUT"

PROMPT="The capital of France is"
N="${N:-16}"

# ---- the batch table (mirrors arch_batch_parity.sh's DIR/DEFAULT/mapping) --
declare -A BATCHDIR=(
  [1]=/tmp/arch-batch [2]=/tmp/arch-batch2 [3]=/tmp/arch-batch3
  [4]=/tmp/arch-batch4 [5]=/tmp/arch-batch5 [6]=/tmp/arch-batch6
  [6b]=/tmp/arch-batch6b [7]=/tmp/arch-batch7 [8]=/tmp/arch-batch8
  [9]=/tmp/arch-batch9 [10]=/tmp/arch-batch10 [11a]=/tmp/arch-batch11a
  [11b]=/tmp/arch-batch11b [12]=/tmp/arch-batch12 [13]=/tmp/arch-batch13
  [14]=/tmp/arch-batch14
)
BATCH="${ARCH_BATCH:-1}"
DIR="${BATCHDIR[$BATCH]:-/tmp/arch-batch}"

model_of() { # cell -> file (the same special cases arch_batch_parity.sh maps)
  local cell="$1"
  # batch-13's flat name differs from batch-5's for the same cell name —
  # discriminate on the batch first
  case "$BATCH::$cell" in
    13::nemotron-h-moe) echo "$DIR/nemotron_h_moe-synth.gguf"; return ;;
  esac
  case "$cell" in
    baichuan13)       echo "$DIR/baichuan-synth.gguf" ;;
    baichuan7)        echo "$DIR/baichuan-synth-7b.gguf" ;;
    mptfull)          echo "$DIR/mpt-synth-full.gguf" ;;
    ernie-nosh)       echo "$DIR/ernie4_5-moe-synth-nosh.gguf" ;;
    granite-moe)      echo "$DIR/granitemoe-synth.gguf" ;;
    seed-oss)         echo "$DIR/seed_oss-synth.gguf" ;;
    ernie4-5-moe)     echo "$DIR/ernie4_5-moe-synth.gguf" ;;
    jamba-moe)        echo "$DIR/jamba-synth-moe.gguf" ;;
    nemotron-h)       echo "$DIR/nemotron_h-synth.gguf" ;;
    nemotron-h-moe)   echo "$DIR/nemotron_h-synth-moe.gguf" ;;
    deepseek2-lite)   echo "$DIR/deepseek2-synth-lite.gguf" ;;
    deepseek2-legacy) echo "$DIR/deepseek2-synth-legacy.gguf" ;;
    deepseek2-ocr)    echo "$DIR/deepseek2-ocr-synth.gguf" ;;
    grok-dense)       echo "$DIR/grok-synth-dense.gguf" ;;
    deci-mixed)       echo "$DIR/deci-synth-mixed.gguf" ;;
    cohere2moe-sep-ln) echo "$DIR/cohere2moe-synth-sep-ln.gguf" ;;
    qwen3next-legacy)  echo "$DIR/qwen3next-synth-legacy.gguf" ;;
    kimi-linear-legacy) echo "$DIR/kimi-linear-synth-legacy.gguf" ;;
    smallthinker-swa)  echo "$DIR/smallthinker-synth-swa.gguf" ;;
    nanbeige-loops2)   echo "$DIR/nanbeige-synth-loops2.gguf" ;;
    hrm-text)          echo "$DIR/hrm_text-synth.gguf" ;;
    laguna-full)       echo "$DIR/laguna-synth-full.gguf" ;;
    maple-allswa)      echo "$DIR/maple-synth-allswa.gguf" ;;
    # batch 13 (b13_model of arch_batch_parity.sh)
    ernie4-5)          echo "$DIR/ernie4_5-synth.gguf" ;;
    glm-dsa-shared)    echo "$DIR/glm-dsa-synth-shared.gguf" ;;
    mistral3-temp)     echo "$DIR/mistral3-synth-temp.gguf" ;;
    glm4-mrope)        echo "$DIR/glm4-synth-mrope.gguf" ;;
    exaone4-swa)       echo "$DIR/exaone4-synth-swa.gguf" ;;
    llama4-noswa)      echo "$DIR/llama4-synth-noswa.gguf" ;;
    gptoss)           echo "${GPTOSS:-/home/jeffrey/.lmstudio/models/lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf}" ;;
    gemma4)           echo "${GEMMA4:-/home/jeffrey/.lmstudio/models/lmstudio-community/gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf}" ;;
    granite-hybrid)   echo "${GRANITE_H:-/home/jeffrey/.lmstudio/models/unsloth/granite-4.0-h-tiny-GGUF/granite-4.0-h-tiny-Q4_K_M.gguf}" ;;
    lfm2)             echo "${LFM2:-/home/jeffrey/.lmstudio/models/LiquidAI/LFM2-8B-A1B-GGUF/LFM2-8B-A1B-Q4_K_M.gguf}" ;;
    qwen35)           echo "${QWEN35:-/home/jeffrey/.lmstudio/models/lmstudio-community/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf}" ;;
    *)                echo "$DIR/$cell-synth.gguf" ;;
  esac
}

# the recurrent / long-context cells want more tokens (arch_batch_parity.sh's
# per-batch N)
cell_n() {
  case "$1" in
    mamba|mamba2|jamba|jamba-moe|nemotron-h|nemotron-h-moe|falcon-h1|plamo2|\
    granite-hybrid|lfm2|qwen35) echo 48 ;;
    rwkv6|rwkv6qwen2|rwkv7|arwkv7|gemma3n) echo 48 ;;
    *) echo "$N" ;;
  esac
}

wait_ready() {
  for _ in $(seq 1 600); do
    curl -s "http://127.0.0.1:$1/health" 2>/dev/null | grep -q '"ok"' && return 0
    sleep 1
  done
  return 1
}

run_ref() { # model fa n tag -> <tag>-ref.json
  local model="$1" fa="$2" n="$3" tag="$4"
  "$REF/llama-server" -m "$model" -c "$CTX" -t "$THREADS" -fa "$fa" \
    --port "$PORT" --host 127.0.0.1 >"$OUT/$tag-server.log" 2>&1 &
  local srv=$!
  trap 'kill "$srv" 2>/dev/null || true' EXIT
  wait_ready "$PORT" || { echo "  ref server never became ready (see $OUT/$tag-server.log)"; return 1; }
  python3 - "$PROMPT" "$n" >"$OUT/$tag-body.json" <<'PY'
import json, sys
print(json.dumps({"prompt": sys.argv[1], "n_predict": int(sys.argv[2]),
                  "temperature": 0.0, "cache_prompt": False, "return_tokens": True}))
PY
  curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
    -d @"$OUT/$tag-body.json" -o "$OUT/$tag-ref.json"
  kill "$srv" 2>/dev/null || true
  wait "$srv" 2>/dev/null || true
  trap - EXIT
}

run_rust() { # model fa n tag -> <tag>-rust.json + <tag>-cli.out
  local model="$1" fa="$2" n="$3" tag="$4"
  ./target/release/llama-server -m "$model" -c "$CTX" -t "$THREADS" -fa "$fa" \
    --port "$PORT" --host 127.0.0.1 >"$OUT/$tag-server-rust.log" 2>&1 &
  local srv=$!
  trap 'kill "$srv" 2>/dev/null || true' EXIT
  wait_ready "$PORT" || { echo "  port server never became ready (see $OUT/$tag-server-rust.log)"; return 1; }
  [ -f "$OUT/$tag-body.json" ] || python3 - "$PROMPT" "$n" >"$OUT/$tag-body.json" <<'PY'
import json, sys
print(json.dumps({"prompt": sys.argv[1], "n_predict": int(sys.argv[2]),
                  "temperature": 0.0, "cache_prompt": False, "return_tokens": True}))
PY
  curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
    -d @"$OUT/$tag-body.json" -o "$OUT/$tag-rust.json"
  kill "$srv" 2>/dev/null || true
  wait "$srv" 2>/dev/null || true
  trap - EXIT
  # port CLI: the same file, greedy, llama-cli's debug format
  LLAMA_RUST_DEBUG=1 ./target/release/llama-cli -m "$model" -p "$PROMPT" -n "$n" \
    -t "$THREADS" -c "$CTX" --temp 0 -fa "$fa" \
    >"$OUT/$tag-cli.out" 2>"$OUT/$tag-cli.err" || true
}

# the port-vs-port CLI token check (llama-cli's `gen tokens:` debug line)
check_cli() {
  python3 - "$1" "$OUT" <<'PY'
import json, re, sys
tag, out = sys.argv[1], sys.argv[2]
try:
    srv = json.load(open(f"{out}/{tag}-rust.json"))
except Exception as e:
    print(f"  port server response unreadable: {e}"); sys.exit(1)
cli_txt = open(f"{out}/{tag}-cli.out").read()
m = re.search(r"gen tokens: \[(.*?)\]", cli_txt)
if not m:
    print(f"  port CLI produced no `gen tokens:` line (see {out}/{tag}-cli.err)"); sys.exit(1)
srv_toks, cli_toks = srv.get("tokens") or [], [int(x) for x in m.group(1).split(",")]
if srv_toks == cli_toks:
    print(f"  port-server == port-CLI: OK ({len(cli_toks)} tokens)")
    sys.exit(0)
n = min(len(srv_toks), len(cli_toks))
first = next((i for i in range(n) if srv_toks[i] != cli_toks[i]), -1)
print(f"  port-server != port-CLI: first divergence at token {first} "
      f"(server {len(srv_toks)} vs cli {len(cli_toks)} tokens)")
sys.exit(1)
PY
}

FAIL=0
for cell in "$@"; do
  model="$(model_of "$cell")"
  if [ ! -f "$model" ]; then
    echo "=== $cell: $model missing — run the batch's generator test first" >&2
    FAIL=1
    continue
  fi
  n="$(cell_n "$cell")"
  for fa in ${FA:-off on}; do
    tag="$cell-$fa"
    echo "=== $cell -fa $fa (n=$n)"
    if [ -n "${WITH_REF:-}" ]; then
      run_ref "$model" "$fa" "$n" "$tag" || { FAIL=1; continue; }
    fi
    run_rust "$model" "$fa" "$n" "$tag" || { FAIL=1; continue; }
    if [ -n "${WITH_REF:-}" ]; then
      python3 parity/server_arch_cmp.py "$tag" "$OUT" || FAIL=1
    fi
    check_cli "$tag" || FAIL=1
  done
done

echo
if [ "$FAIL" -eq 0 ]; then
  echo "server arch sweep: all cells matched"
else
  echo "server arch sweep: at least one cell diverged (see above)"
fi
exit "$FAIL"
