#!/usr/bin/env bash
# bench_matrix.sh — CPU performance matrix: llama.rust (port) vs llama.cpp (ref).
#
# Instrument: llama-bench on BOTH sides (the port's llama-bench CLI/CSV is
# byte-identical to the reference's), so both sides run the exact same test
# definitions, warm-up and rep protocol under identical flags.
#
#   ref  : $LLAMA_RUST_DATA_HOME/llm/llama.cpp-next/build-rust-ref  (aa94f2086, AVX512+REPACK, CPU only)
#   port : ./target/release/llama-bench                     (HEAD, release —
#          compare against the pinned reference at the commit you built)
#
# Matrix: 6 models x {pp512, pp128, tg128} @ -t 8 -fa on -r 3
#         + probes: -t 32 scaling (qwen-0.5b, gemma-12b Q4_0), -fa off (qwen-0.5b)
#
# Fairness protocol:
#   * every run goes through parity/limited.sh -c 3200% (48G memory cap, but
#     NO cpu quota — the default 800% CFS-throttles -t 8 spin-poll threads and
#     inflates stddev; first run of this script was invalidated by that)
#   * page cache pre-warmed per model (cat > /dev/null) so neither side pays cold I/O
#   * side order alternates per model (drift cancellation)
#   * port runs get LLAMA_RUST_REPACK_WARM_MAX_MB=16384 so CPU_REPACK
#     materializes at load exactly like the reference's (bench.sh precedent)
#   * memory guard: a model is skipped if available mem < its threshold
#
# usage: parity/bench_matrix.sh [matrix|probes|all]
# llama-rust-data-home: root of the local model/reference corpus (default: $HOME)
DATA_HOME="${LLAMA_RUST_DATA_HOME:-$HOME}"
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REF_BIN="${REF_BIN:-${DATA_HOME}/llm/llama.cpp-next/build-rust-ref/bin/llama-bench}"
PORT_BIN="${PORT_BIN:-$ROOT/target/release/llama-bench}"
OUT="$ROOT/parity/bench_matrix/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
LIMITED="$ROOT/parity/limited.sh"
PHASE="${1:-all}"

# label|model|min_available_G
MODELS=(
  "qwen2.5-0.5b-q4km|${DATA_HOME}/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf|20"
  "qwen2.5-7b-q4km|${DATA_HOME}/localai/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf|24"
  "gemma-4-12b-q4km|${DATA_HOME}/.lmstudio/models/lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf|24"
  "gemma-4-12b-q40|${DATA_HOME}/.lmstudio/models/lmstudio-community/gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf|24"
  "lfm2-8b-a1b-q4km|${DATA_HOME}/.lmstudio/models/LiquidAI/LFM2-8B-A1B-GGUF/LFM2-8B-A1B-Q4_K_M.gguf|24"
  "gpt-oss-20b-mxfp4|${DATA_HOME}/.lmstudio/models/lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf|30"
)
# label|model|extra llama-bench args (probes)
PROBES=(
  "qwen0.5b-t32|${DATA_HOME}/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf|-t 32 -fa on -p 512 -n 128"
  "gemma12b-q40-t32|${DATA_HOME}/.lmstudio/models/lmstudio-community/gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf|-t 32 -fa on -p 512 -n 128"
  "qwen0.5b-faoff|${DATA_HOME}/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf|-t 8 -fa off -p 512 -n 128"
)

avail_g() { free -g | awk '/^Mem:/{print $7}'; }

# warm the page cache so neither side pays cold I/O on mmap (incl. shards)
prewarm() {
  local f="$1" part
  cat "$f" > /dev/null 2>&1 || true
  if [[ "$(basename "$f")" == *-of-* ]]; then
    local stem="${f%-of-*}"
    for part in "$stem"-of-*.gguf; do [[ -f "$part" ]] && cat "$part" > /dev/null 2>&1 || true; done
  fi
}

run_side() { # side label model csv_out extra...
  local side="$1" label="$2" model="$3" out="$4"; shift 4
  local bin rc
  bin="$REF_BIN"; [[ "$side" == port ]] && bin="$PORT_BIN"
  if [[ "$side" == port ]]; then
    LLAMA_RUST_REPACK_WARM_MAX_MB=16384 \
      bash "$LIMITED" -c 3200% -- "$bin" -m "$model" -r 3 -o csv "$@" >"$out" 2>"$out.err"
  else
    bash "$LIMITED" -c 3200% -- "$bin" -m "$model" -r 3 -o csv "$@" >"$out" 2>"$out.err"
  fi
  rc=$?
  if [[ $rc -ne 0 || ! -s "$out" ]]; then
    echo "    !! $side $label rc=$rc (see $out.err)" >&2
    tail -3 "$out.err" >&2
    return 1
  fi
  return 0
}

idx=0
if [[ "$PHASE" != "probes" ]]; then
for spec in "${MODELS[@]}"; do
  IFS='|' read -r label model minavail <<<"$spec"
  idx=$((idx+1))
  echo "=== [$idx/${#MODELS[@]}] $label ===" >&2
  av="$(avail_g)"
  if (( av < minavail )); then
    echo "    SKIP: available ${av}G < ${minavail}G" >&2
    continue
  fi
  echo "    prewarming page cache ($(du -h "$model" | cut -f1))..." >&2
  prewarm "$model"
  # alternate side order to cancel ambient drift
  if (( idx % 2 == 1 )); then order=(ref port); else order=(port ref); fi
  for side in "${order[@]}"; do
    echo "    $side: pp512/pp128/tg128 @ -t 8 -fa on" >&2
    run_side "$side" "$label" "$model" "$OUT/${label}_${side}.csv" \
      -t 8 -fa on -p 512,128 -n 128
  done
done
fi

if [[ "$PHASE" != "matrix" ]]; then
probe_idx=0
for spec in "${PROBES[@]}"; do
  IFS='|' read -r label model extra <<<"$spec"
  echo "=== probe $label ($extra) ===" >&2
  av="$(avail_g)"
  if (( av < 24 )); then
    echo "    SKIP: available ${av}G < 24G" >&2
    continue
  fi
  prewarm "$model"
  # alternate side order across probes like the matrix phase (drift cancel)
  probe_idx=$((probe_idx + 1))
  if (( probe_idx % 2 == 1 )); then
    run_side ref  "$label" "$model" "$OUT/${label}_ref.csv"  $extra
    run_side port "$label" "$model" "$OUT/${label}_port.csv" $extra
  else
    run_side port "$label" "$model" "$OUT/${label}_port.csv" $extra
    run_side ref  "$label" "$model" "$OUT/${label}_ref.csv"  $extra
  fi
done
fi

echo
echo "=== aggregating $OUT ===" >&2
python3 - "$OUT" <<'EOF'
import csv, glob, os, sys

out_dir = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(out_dir, "*.csv"))):
    name = os.path.basename(path)[:-4]
    label, side = name.rsplit("_", 1)
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(dict(
                label=label, side=side,
                p=int(r["n_prompt"]), n=int(r["n_gen"]),
                t=int(r["n_threads"]), fa=r["flash_attn"],
                avg=float(r["avg_ts"]), sd=float(r["stddev_ts"])))

def test_name(r):
    if r["p"] and not r["n"]: return f'pp{r["p"]}'
    if r["n"] and not r["p"]: return f'tg{r["n"]}'
    return f'pp{r["p"]}+tg{r["n"]}'

groups = {}
for r in rows:
    groups.setdefault((r["label"], r["t"], r["fa"], test_name(r)), {})[r["side"]] = r

hdr = f'{"model":<22}{"thr":>4}{"fa":>5}{"test":>9}{"ref t/s":>11}{"±sd":>7}{"port t/s":>11}{"±sd":>7}{"ratio":>8}'
print(hdr); print("-" * len(hdr))
for key in sorted(groups, key=lambda k: (k[0], k[1], k[3])):
    label, t, fa, test = key
    g = groups[key]
    ref, port = g.get("ref"), g.get("port")
    if not ref or not port:
        side = "ref" if ref else "port"
        print(f'{label:<22}{t:>4}{fa:>5}{test:>9}   MISSING {side}')
        continue
    ratio = port["avg"] / ref["avg"] if ref["avg"] else float("nan")
    print(f'{label:<22}{t:>4}{fa:>5}{test:>9}'
          f'{ref["avg"]:>11.1f}{ref["sd"]:>7.1f}'
          f'{port["avg"]:>11.1f}{port["sd"]:>7.1f}{ratio:>8.2f}x')
EOF
echo "raw csv: $OUT"
