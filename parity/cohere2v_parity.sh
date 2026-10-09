#!/usr/bin/env bash
# cohere2v_parity.sh — the cohere2 vision chain (siglip tower + llava-uhd
# square tiles + swapped-swiglu projector, upstream 50a6c5cf7) at synthetic
# scale, both ends (protocol: batches 19/20 audio projectors — the port writes
# the synthetic mmproj, the pinned reference accepts it, embeddings compare
# bitwise). No real cohere2 mmproj exists locally, so the synthetic file IS
# the fixture (honest record: this validates everything except real-weight
# numerics, which no local artifact can).
#
#  1. the port writes the synthetic cohere2v mmproj + dumps its encoder
#     output for the raw `cb` bitmap at tile size (MTMD_FA=off and on)
#  2. the reference llama-mtmd-debug loads the SAME file and dumps its
#     embeddings via MTMD_DEBUG_EMBEDDINGS — both attention paths
#  3. python bitwise compare (f32 bits, like mtmd_parity.sh)
#  4. the reference's `-p preproc` log gives the slicing geometry
#     (tile count + sizes); the port's unit test asserts the same
#
# Text model: qwen2.5-0.5b (only used by llama-mtmd-debug to init the vocab;
# the cohere2 markers are absent from it — lookup_token returns
# LLAMA_TOKEN_NULL, same as the port).
#
# Evidence snapshots of the last green run: parity/ground-truth/cohere2v/
# (synth mmproj + both ref/port dumps + ref logs; the dir is gitignored like
# every reference dump — see .gitignore parity/ground-truth/).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF=/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin
TEXT="${TEXT:-/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
OUT=/tmp/closv
MMPROJ="$OUT/mmproj-cohere2v-synth.gguf"
cd "$ROOT"

mkdir -p "$OUT"

# 1. the port side: synth file + encoder dumps (both attention paths)
MTMD_COHERE2V_MMPROJ="$MMPROJ" MTMD_FA=off MTMD_CB_SIZE=64 MTMD_CB_OUT="$OUT/port_cb_cohere2v_64_faoff.bin" \
    cargo test -p llama --release --lib clip_cb_parity_dump_cohere2v -- --ignored --nocapture \
    >"$OUT/port_dump_faoff.log" 2>&1 || { tail -20 "$OUT/port_dump_faoff.log"; exit 1; }
MTMD_COHERE2V_MMPROJ="$MMPROJ" MTMD_FA=on MTMD_CB_SIZE=64 MTMD_CB_OUT="$OUT/port_cb_cohere2v_64_faon.bin" \
    cargo test -p llama --release --lib clip_cb_parity_dump_cohere2v -- --ignored --nocapture \
    >"$OUT/port_dump_faon.log" 2>&1 || { tail -20 "$OUT/port_dump_faon.log"; exit 1; }

# 2. the reference accepts the same synthetic mmproj and dumps embeddings
status=0
for tag in faoff faon; do
    faopt=""; [ "$tag" == "faoff" ] && faopt="-fa off"
    MTMD_DEBUG_EMBEDDINGS="$OUT/ref_cb_cohere2v_64_$tag.bin" \
        "$REF/llama-mtmd-debug" -m "$TEXT" --mmproj "$MMPROJ" -p encode -n 64 \
        --image cb -t 8 $faopt >"$OUT/ref_encode_$tag.log" 2>&1
    rc=$?
    echo "reference llama-mtmd-debug [encode/$tag] exit: $rc"
    if [ $rc -ne 0 ]; then
        echo "== reference REJECTED the synthetic cohere2v projector =="
        tail -25 "$OUT/ref_encode_$tag.log"
        status=1
        continue
    fi
    grep -q "cohere2v" "$OUT/ref_encode_$tag.log" \
        && echo "  banner: $(grep -m1 'projector:' "$OUT/ref_encode_$tag.log")"
done

# 3. bitwise compare (the reference dump names swap: ref faoff vs port off)
if [ $status -eq 0 ]; then
    python3 - "$OUT" <<'PYEOF'
import struct, sys
out = sys.argv[1]

def load(path):
    d = open(path, 'rb').read()
    n_tokens, n_embd = struct.unpack('<ii', d[:8])
    assert len(d) - 8 == n_tokens * n_embd * 4, (path, len(d), n_tokens, n_embd)
    return n_tokens, n_embd, struct.unpack(f'<{n_tokens*n_embd}f', d[8:])

for tag in ("faoff", "faon"):
    port = f"{out}/port_cb_cohere2v_64_{tag}.bin"
    rt, re_, rv = load(f"{out}/ref_cb_cohere2v_64_{tag}.bin")
    pt, pe, pv = load(port)
    assert (rt, re_) == (pt, pe), f"shape mismatch: ref {rt}x{re_} vs port {pt}x{pe}"
    diff = sum(1 for a, b in zip(rv, pv) if a != b)  # f32 bit compare
    n = len(rv)
    if diff == 0:
        print(f"[cohere2v/{tag}] BIT-EXACT: {n} f32 values identical ({pt} tokens x {pe} embd)")
    else:
        maxd = max(abs(a - b) for a, b in zip(rv, pv))
        print(f"[cohere2v/{tag}] NOT bit-exact: {diff}/{n} differ, max |delta| = {maxd:.3e}")
        sys.exit(1)
PYEOF
    [ $? -ne 0 ] && status=1
fi

# 4. the reference's preprocessor log: the llava-uhd square-tile geometry
#    (tile 64, max 4 tiles): 150x150 -> 4 tiles of 64x64 + overview;
#    50x50 -> overview only
"$REF/llama-mtmd-debug" --verbose -m "$TEXT" --mmproj "$MMPROJ" -p preproc -n 150 \
    --image cb -t 8 >"$OUT/ref_preproc_150.log" 2>&1
"$REF/llama-mtmd-debug" --verbose -m "$TEXT" --mmproj "$MMPROJ" -p preproc -n 50 \
    --image cb -t 8 >"$OUT/ref_preproc_50.log" 2>&1
echo "== reference load banner + preproc geometry =="
grep -aE "load_hparams: (n_merge|image_size|preproc_tiles)" "$OUT/ref_preproc_150.log" | sed 's/^/  /' 
grep -h "entry .* has nx=" "$OUT/ref_preproc_150.log" | sed 's/^/  150x150: /'
grep -h "batch_f32 with .* entries" "$OUT/ref_preproc_150.log" | sed 's/^/  150x150: /'
grep -h "batch_f32 with .* entries" "$OUT/ref_preproc_50.log" | sed 's/^/  50x50:   /'
grep -h "entry .* has nx=" "$OUT/ref_preproc_50.log" | sed 's/^/  50x50:   /'

# 5. the port asserts the same geometry in-process
cargo test -p llama --release --lib cohere2v 2>&1 | tail -2 | tee "$OUT/port_unit.log"
grep -q "test result: ok" "$OUT/port_unit.log" || status=1

exit $status
