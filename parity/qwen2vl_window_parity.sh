#!/usr/bin/env bash
# qwen2vl_window_parity.sh — the qwen2.5-vl window-attention arm (the A2
# handoff's "qwen2vl use_window_attn 整臂") at synthetic scale, both ends:
# the port writes a synthetic qwen2.5vl mmproj with clip.vision.n_wa_pattern
# = 2 (layers 1,3 full / 0,2 windowed; 128px cb bitmap → 16x16 patch grid →
# merged 8x8 tokens → grid_window 7 → 4 windows 7x7/7x1/1x7/1x1), the pinned
# reference accepts the same file and both dump embeddings (MTMD_DEBUG_
# EMBEDDINGS) for a bitwise compare — the cohere2v_parity.sh protocol.
#
# Covers: the models/qwen2vl.cpp graph arm (separate qkv + window-major
# get_rows reorder + per-layer window mask + raster restore), the
# clip.cpp:5037-5110 set_inputs arm (window_idx / inv_window_idx /
# block-diagonal mask / remapped M-RoPE positions) and the fixed
# clip.vision.n_wa_pattern key read (clip-impl.h:71) — FA off and on.
#
# Text model: qwen2.5-0.5b (only used by llama-mtmd-debug to init the vocab;
# its projection_dim 896 matches the synthetic mm.2).
# llama-rust-data-home: root of the local model/reference corpus (default: $HOME)
DATA_HOME="${LLAMA_RUST_DATA_HOME:-$HOME}"
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF=${DATA_HOME}/llm/llama.cpp-next/build-rust-ref/bin
TEXT="${TEXT:-${DATA_HOME}/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
OUT=/tmp/r4o3/qwen2vl
MMPROJ="$OUT/mmproj-qwen25vl-synth.gguf"
cd "$ROOT"

mkdir -p "$OUT"

# 1. the port side: synth file + encoder dumps (both attention paths)
MTMD_QWEN25VL_MMPROJ="$MMPROJ" MTMD_FA=off MTMD_CB_SIZE=128 MTMD_CB_OUT="$OUT/port_cb_qwen25vl_128_faoff.bin" \
    cargo test -p llama --release --lib clip_cb_parity_dump_qwen25vl -- --ignored --nocapture \
    >"$OUT/port_dump_faoff.log" 2>&1 || { tail -20 "$OUT/port_dump_faoff.log"; exit 1; }
MTMD_QWEN25VL_MMPROJ="$MMPROJ" MTMD_FA=on MTMD_CB_SIZE=128 MTMD_CB_OUT="$OUT/port_cb_qwen25vl_128_faon.bin" \
    cargo test -p llama --release --lib clip_cb_parity_dump_qwen25vl -- --ignored --nocapture \
    >"$OUT/port_dump_faon.log" 2>&1 || { tail -20 "$OUT/port_dump_faon.log"; exit 1; }

# 2. the reference accepts the same synthetic mmproj and dumps embeddings
status=0
for tag in faoff faon; do
    faopt=""; [ "$tag" == "faoff" ] && faopt="-fa off"
    MTMD_DEBUG_EMBEDDINGS="$OUT/ref_cb_qwen25vl_128_$tag.bin" \
        "$REF/llama-mtmd-debug" -m "$TEXT" --mmproj "$MMPROJ" -p encode -n 128 \
        --image cb -t 8 $faopt >"$OUT/ref_encode_$tag.log" 2>&1
    rc=$?
    echo "reference llama-mtmd-debug [encode/$tag] exit: $rc"
    if [ $rc -ne 0 ]; then
        echo "== reference REJECTED the synthetic qwen2.5vl projector =="
        tail -25 "$OUT/ref_encode_$tag.log"
        status=1
        continue
    fi
    grep -q "n_wa_pattern" "$OUT/ref_encode_$tag.log" \
        && echo "  hparams: $(grep -m1 'n_wa_pattern' "$OUT/ref_encode_$tag.log")"
done

# 3. bitwise compare (f32 bits)
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
    rt, re_, rv = load(f"{out}/ref_cb_qwen25vl_128_{tag}.bin")
    pt, pe, pv = load(f"{out}/port_cb_qwen25vl_128_{tag}.bin")
    assert (rt, re_) == (pt, pe), f"shape mismatch: ref {rt}x{re_} vs port {pt}x{pe}"
    diff = sum(1 for a, b in zip(rv, pv) if a != b)  # f32 bit compare
    n = len(rv)
    if diff == 0:
        print(f"[qwen25vl-window/{tag}] BIT-EXACT: {n} f32 values identical ({pt} tokens x {pe} embd)")
    else:
        maxd = max(abs(a - b) for a, b in zip(rv, pv))
        print(f"[qwen25vl-window/{tag}] NOT bit-exact: {diff}/{n} differ, max |delta| = {maxd:.3e}")
        sys.exit(1)
PYEOF
    [ $? -ne 0 ] && status=1
fi

# 4. the port's unit tests: the window-payload math + the toy-scale encode
cargo test -p llama --release --lib qwen25vl 2>&1 | tail -2 | tee "$OUT/port_unit.log"
grep -q "test result: ok" "$OUT/port_unit.log" || status=1

exit $status
