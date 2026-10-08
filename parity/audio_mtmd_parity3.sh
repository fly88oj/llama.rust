#!/usr/bin/env bash
# audio_mtmd_parity3.sh — audio-encoder round 4: the remaining encoder graphs
# (granite_speech / gemma4a / parakeet / mimo_audio / qwen3tts_spkenc /
# pockettts_spkenc) built with the port's GGUF writer, accepted and run by the
# pinned reference llama-mtmd-cli, embeddings compared bit-exactly against the
# port (agent: AUDIO4).
#
#  1. the port writes the synthetic mmprojs (tests/mtmd_audio_synth3.rs)
#  2. the reference llama-mtmd-cli loads each (--mmproj) and runs a full audio
#     turn on the rate-matched fixture — 16 kHz archs on mtmd-fixture-audio.wav,
#     the 24 kHz archs (mimo/qwen3tts/pockettts) on mtmd-fixture-audio-24k.wav
#     (a rate-matched file is a passthrough in the reference miniaudio path —
#     no resampler). Each arch runs twice: default flash-attn and -fa off.
#  3. the reference's audio embeddings are captured via MTMD_DEBUG_EMBEDDINGS
#     and compared against the port's encoder graph output bit-exactly
#     (f32::to_bits).
#
# Text model: qwen2.5-0.5b (n_embd 896) drives all six.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF=/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin
TEXT="${TEXT:-/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
WAV16="$ROOT/parity/mtmd-fixture-audio.wav"
WAV24="$ROOT/parity/mtmd-fixture-audio-24k.wav"
OUT=/tmp/mtmd-audio-synth
ARCHS="${ARCHS:-granite_speech gemma4a parakeet mimo_audio qwen3tts_spkenc pockettts_spkenc}"
cd "$ROOT"

wav_for() {
    case "$1" in
        granite_speech|gemma4a|parakeet) echo "$WAV16" ;;
        *) echo "$WAV24" ;;
    esac
}

# 1. the synthetic mmprojs + the port's embedding dumps
#    (regenerate whenever any dump is missing or older than clip.rs)
need_regen=0
for a in $ARCHS; do
    if [ ! -f "$OUT/port-default-$a.bin" ] || [ -n "$(find "$ROOT/crates/llama/src/clip.rs" -newer "$OUT/port-default-$a.bin" 2>/dev/null)" ]; then
        need_regen=1
    fi
done
if [ "${1:-}" == "--regen" ] || [ "$need_regen" == "1" ]; then
    MTMD_IGNORE_REF=1 cargo test -p llama --test mtmd_audio_synth3 -- --nocapture 2>&1 | tail -3
fi

python3 parity/gen_audio_fixture.py

# 2. the reference accepts each arch and dumps its audio embeddings
status=0
for arch in $ARCHS; do
    MMPROJ="$OUT/mmproj-audio-synth-$arch.gguf"
    if [ ! -f "$MMPROJ" ]; then
        echo "== $arch: synthetic mmproj missing ($MMPROJ)"; status=1; continue
    fi
    WAV="$(wav_for "$arch")"
    MTMD_DEBUG_EMBEDDINGS="$OUT/ref-default-$arch.bin" \
        "$REF/llama-mtmd-cli" -m "$TEXT" --mmproj "$MMPROJ" \
        --audio "$WAV" -p "Describe the sound." -n 8 --temp 0 -t "$(nproc)" \
        >"$OUT/ref_mtmd_cli3-$arch.log" 2>&1
    rc=$?
    MTMD_DEBUG_EMBEDDINGS="$OUT/ref-faoff-$arch.bin" \
        "$REF/llama-mtmd-cli" -m "$TEXT" --mmproj "$MMPROJ" \
        --audio "$WAV" -p "Describe the sound." -n 8 --temp 0 -fa off -t "$(nproc)" \
        >>"$OUT/ref_mtmd_cli3-$arch.log" 2>&1
    rc_off=$?
    echo "reference llama-mtmd-cli [$arch] exit: $rc (fa on) / $rc_off (-fa off)"
    if [ $rc -ne 0 ] || [ $rc_off -ne 0 ]; then
        echo "== reference REJECTED the synthetic $arch projector =="
        tail -20 "$OUT/ref_mtmd_cli3-$arch.log"
        status=1
        continue
    fi
    # 3. comparison against the port's dump (both attention paths)
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

# 4. the port-side test re-run picks the ref dumps up and asserts the same
if [ $status -eq 0 ]; then
    cargo test -p llama --test mtmd_audio_synth3 2>&1 | tail -2 | tee /tmp/audio-parity3-test.log
    grep -q "test result: ok" /tmp/audio-parity3-test.log || status=1
fi

exit $status
