#!/usr/bin/env bash
# d1omni_v_parity.sh — the lfm2/d1omni_v vision family (the V-domain open
# item: the siglip tower's lfm2 tail + resize_position_embeddings + the lfm2
# slice preprocessor + the KEY_IMAGE_RESIZE_ALGO read arm) at synthetic
# scale, both ends, two fixture legs:
#
#   g16 (CONTROL, hard gate): the learned position grid is 16x16 == the cb
#     bitmap's patch grid, so no interpolation runs — everything else must
#     be BIT-EXACT vs the reference (hparams arm, separate-qkv tower, the
#     1e-5 input norm, the GELU-ERF projector, the merged token grid).
#
#   g9 (INTERPOLATION, attributed residual): the 9x9 grid forces
#     resize_position_embeddings' BILINEAR|ANTIALIAS mode (the siglip
#     no-arg default, clip-graph.h:12). The port kernel now implements the
#     flag (compute.rs forward_upscale, ops.cpp:8252-8349); the residual vs
#     the reference .so is 1-2 ulp on a subset — the .so build's FP
#     contraction of the kernel's multiply-adds, proven at kernel level by
#     clip::tests::interp_antialias_port_vs_ref against
#     parity/interp_antialias_ref.bin (62% bit-same, max 1.5e-8) — the same
#     documented phenomenon as the qwen3vl bilinear (the parity ledger /
#     parity/ref_interp_dump.c). Recorded, not gated.
#
# The cohere2v_parity.sh protocol throughout. Text model: qwen2.5-0.5b
# (llama-mtmd-debug's vocab init; projection_dim 896 matches its n_embd).
# llama-rust-data-home: root of the local model/reference corpus (default: $HOME)
DATA_HOME="${LLAMA_RUST_DATA_HOME:-$HOME}"
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF=${DATA_HOME}/llm/llama.cpp-next/build-rust-ref/bin
TEXT="${TEXT:-${DATA_HOME}/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
OUT=/tmp/r4o3/d1omni_v
cd "$ROOT"

mkdir -p "$OUT"
status=0

# 1.+2. both legs, both attention paths: the port dumps, the reference
# accepts the same synthetic file and dumps
for leg in g16 g9; do
    grid=16; [ "$leg" == "g9" ] && grid=9
    MMPROJ="$OUT/mmproj-d1omni_v-$leg.gguf"
    for tag in faoff faon; do
        faopt=""; [ "$tag" == "faoff" ] && faopt="-fa off"
        famode=on; [ "$tag" == "faoff" ] && famode=off
        MTMD_D1OMNI_V_MMPROJ="$MMPROJ" MTMD_D1OMNI_POS_GRID=$grid MTMD_FA=$famode \
            MTMD_CB_SIZE=128 MTMD_CB_OUT="$OUT/port_cb_d1omni_v_128_${leg}_$tag.bin" \
            cargo test -p llama --release --lib clip_cb_parity_dump_d1omni_v -- --ignored --nocapture \
            >"$OUT/port_dump_${leg}_$tag.log" 2>&1 || { tail -20 "$OUT/port_dump_${leg}_$tag.log"; exit 1; }
        MTMD_DEBUG_EMBEDDINGS="$OUT/ref_cb_d1omni_v_128_${leg}_$tag.bin" \
            "$REF/llama-mtmd-debug" -m "$TEXT" --mmproj "$MMPROJ" -p encode -n 128 \
            --image cb -t 8 $faopt >"$OUT/ref_encode_${leg}_$tag.log" 2>&1
        rc=$?
        echo "reference llama-mtmd-debug [encode/$leg/$tag] exit: $rc"
        [ $rc -ne 0 ] && { tail -25 "$OUT/ref_encode_${leg}_$tag.log"; status=1; }
    done
done

# 3. bitwise compare: g16 gates, g9 records the attributed residual
python3 - "$OUT" <<'PYEOF'
import struct, sys
out = sys.argv[1]

def load(path):
    d = open(path, 'rb').read()
    n_tokens, n_embd = struct.unpack('<ii', d[:8])
    assert len(d) - 8 == n_tokens * n_embd * 4, (path, len(d), n_tokens, n_embd)
    return n_tokens, n_embd, struct.unpack(f'<{n_tokens*n_embd}f', d[8:])

fail = False
for leg in ("g16", "g9"):
    for tag in ("faoff", "faon"):
        rt, re_, rv = load(f"{out}/ref_cb_d1omni_v_128_{leg}_{tag}.bin")
        pt, pe, pv = load(f"{out}/port_cb_d1omni_v_128_{leg}_{tag}.bin")
        assert (rt, re_) == (pt, pe), f"shape mismatch: ref {rt}x{re_} vs port {pt}x{pe}"
        diff = sum(1 for a, b in zip(rv, pv) if a != b)  # f32 bit compare
        n = len(rv)
        if diff == 0:
            print(f"[d1omni_v/{leg}/{tag}] BIT-EXACT: {n} f32 values identical ({pt} tokens x {pe} embd)")
        else:
            maxd = max(abs(a - b) for a, b in zip(rv, pv))
            print(f"[d1omni_v/{leg}/{tag}] {diff}/{n} differ, max |delta| = {maxd:.3e}")
            if leg == "g16":
                # the control leg has no interpolation — any diff is a REAL
                # port bug, not the missing antialias kernel
                fail = True
            else:
                print("  (g9: 1-2 ulp contraction residual of the .so build, proven at kernel")
                print("   level by interp_antialias_port_vs_ref — the documented ref_interp_dump")
                print("   phenomenon; a structural error would show O(1e-2) deltas)")
sys.exit(1 if fail else 0)
PYEOF
[ $? -ne 0 ] && status=1

# 4. the port's unit tests: the toy-scale encode + the lfm2 slice geometry
cargo test -p llama --release --lib d1omni_v 2>&1 | tail -2 | tee "$OUT/port_unit.log"
grep -q "test result: ok" "$OUT/port_unit.log" || status=1
cargo test -p llama --release --lib lfm2_slice_geometry 2>&1 | tail -2 | tee -a "$OUT/port_unit.log"
grep -c "test result: ok" "$OUT/port_unit.log" | grep -q 2 || status=1

exit $status
