#!/usr/bin/env bash
# audio_mtmd_parity.sh — reference acceptance + bit-exact embedding parity of
# the synthetic whisper-enc family projectors (agent: AUDIO/whisper-enc).
#
# 1. the port writes six synthetic whisper-family mmprojs
#    (tests/mtmd_audio_synthetic.rs::write_synthetic_mmproj):
#    qwen2a / ultravox / voxtral / meralion / glma / musicflamingo
# 2. the pinned reference llama-mtmd-cli loads each (--mmproj) and runs a full
#    audio turn on parity/mtmd-fixture-audio.wav — proving the file layout
#    (metadata + tensor names + shapes) is exactly what the reference expects.
#    Each arch runs twice: with the default flash-attn (AUTO -> enabled on
#    CPU) and with -fa off (the soft_max_ext path of build_attn).
# 3. the reference's audio embeddings are captured via MTMD_DEBUG_EMBEDDINGS
#    and compared BIT-EXACTLY (f32::to_bits) against the port's whisper-enc
#    graph output (models/whisper-enc.cpp:3-137) on the same mel chunk.
#
# Text model: a small local instruct model (qwen2.5-0.5b, n_embd 896 — the
# synthetic projector output width).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF=/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin
TEXT="${TEXT:-/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
WAV="$ROOT/parity/mtmd-fixture-audio.wav"
OUT=/tmp/mtmd-audio-synth
ARCHS="${ARCHS:-qwen2a ultravox voxtral meralion glma musicflamingo}"
cd "$ROOT"

# 1. the synthetic mmprojs + the port's embedding dumps
if [ "${1:-}" == "--regen" ] || [ ! -f "$OUT/port-default-qwen2a.bin" ]; then
    cargo test -p llama --test mtmd_audio_synthetic -- --nocapture 2>&1 | tail -3
fi

python3 parity/gen_audio_fixture.py

# 2. the reference accepts each arch and dumps its audio embeddings
status=0
for arch in $ARCHS; do
    MMPROJ="$OUT/mmproj-audio-synth-$arch.gguf"
    if [ ! -f "$MMPROJ" ]; then
        echo "== $arch: synthetic mmproj missing ($MMPROJ)"; status=1; continue
    fi
    # 2a. default (flash attn AUTO -> enabled on CPU) and 2b. -fa off
    MTMD_DEBUG_EMBEDDINGS="$OUT/ref-default-$arch.bin" \
        "$REF/llama-mtmd-cli" -m "$TEXT" --mmproj "$MMPROJ" \
        --audio "$WAV" -p "Describe the sound." -n 8 --temp 0 -t "$(nproc)" \
        >"$OUT/ref_mtmd_cli-$arch.log" 2>&1
    rc=$?
    MTMD_DEBUG_EMBEDDINGS="$OUT/ref-faoff-$arch.bin" \
        "$REF/llama-mtmd-cli" -m "$TEXT" --mmproj "$MMPROJ" \
        --audio "$WAV" -p "Describe the sound." -n 8 --temp 0 -fa off -t "$(nproc)" \
        >>"$OUT/ref_mtmd_cli-$arch.log" 2>&1
    rc_off=$?
    echo "reference llama-mtmd-cli [$arch] exit: $rc (fa on) / $rc_off (-fa off)"
    if [ $rc -ne 0 ] || [ $rc_off -ne 0 ]; then
        echo "== reference REJECTED the synthetic $arch projector =="
        tail -20 "$OUT/ref_mtmd_cli-$arch.log"
        status=1
        continue
    fi
    if [ ! -f "$OUT/ref-default-$arch.bin" ] || [ ! -f "$OUT/ref-faoff-$arch.bin" ]; then
        echo "== $arch: no reference embedding dump =="; status=1; continue
    fi
    # 3. bit-exact comparison against the port's dump (both attention paths)
    for tag in default faoff; do
        python3 - "$arch" "$tag" "$OUT/ref-$tag-$arch.bin" "$OUT/port-$tag-$arch.bin" <<'PYEOF'
import struct, sys
arch, tag, ref_path, port_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

def load(path):
    d = open(path, 'rb').read()
    n_tokens, n_embd = struct.unpack('<ii', d[:8])
    assert len(d) - 8 == n_tokens * n_embd * 4, (path, len(d), n_tokens, n_embd)
    return n_tokens, n_embd, struct.unpack(f'<{n_tokens*n_embd}f', d[8:])

rt, re_, rv = load(ref_path)
pt, pe, pv = load(port_path)
assert (rt, re_) == (pt, pe), f"shape mismatch: ref {rt}x{re_} vs port {pt}x{pe}"
diff = sum(1 for a, b in zip(rv, pv) if a != b)  # f32 bit compare (NaN-safe ==)
n = len(rv)
if diff == 0:
    print(f"[{arch}/{tag}] BIT-EXACT: {n} f32 values identical ({pt} tokens x {pe} embd)")
else:
    maxd = max(abs(a - b) for a, b in zip(rv, pv))
    print(f"[{arch}/{tag}] NOT bit-exact: {diff}/{n} differ, max |delta| = {maxd:.3e}")
    sys.exit(1)
PYEOF
        [ $? -ne 0 ] && status=1
    done
done

# 4. the port-side test re-run picks the ref dumps up and asserts bit-exactness
if [ $status -eq 0 ]; then
    cargo test -p llama --test mtmd_audio_synthetic 2>&1 | tail -2 | tee /tmp/audio-parity-test.log
    grep -q "test result: ok" /tmp/audio-parity-test.log || status=1
fi

if [ $status -eq 0 ]; then
    echo "ACCEPTED + BIT-EXACT: the reference loads and runs every port-written"
    echo "whisper-family projector (both attention paths), and the embeddings"
    echo "match the port bit-for-bit"
fi
exit $status
