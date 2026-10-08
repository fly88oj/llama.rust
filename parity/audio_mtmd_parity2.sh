#!/usr/bin/env bash
# audio_mtmd_parity2.sh — audio-encoder round 2: the non-whisper encoders
# (qwen3a / gemma4ua / lfm2a) built with the port's GGUF writer, accepted and
# run by the pinned reference llama-mtmd-cli, embeddings compared against the
# port (agent: AUDIO3).
#
#  1. the port writes the synthetic mmprojs (tests/mtmd_audio_synth2.rs)
#  2. the reference llama-mtmd-cli loads each (--mmproj) and runs a full audio
#     turn on parity/mtmd-fixture-audio.wav — proving the file layout
#     (metadata + tensor names + shapes) is exactly what the reference
#     expects. Each arch runs twice: default flash-attn and -fa off.
#  3. the reference's audio embeddings are captured via MTMD_DEBUG_EMBEDDINGS
#     and compared against the port's encoder graph output:
#       qwen3a   / gemma4ua — BIT-EXACT (f32::to_bits), both attention paths
#       lfm2a    — reference-accepted, embeddings agree to <=1e-4 (a ULP-scale
#                  accumulation difference in the conformer conv module,
#                  documented in PARITY.md; NOT yet bit-exact)
#
# Text model: qwen2.5-0.5b (n_embd 896). lfm2a's n_mmproj_embd comes from the
# position-embedding width (clip.cpp:6005), which the synthetic carries as the
# text width — the same 896 model drives all three.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF=/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin
TEXT="${TEXT:-/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
WAV="$ROOT/parity/mtmd-fixture-audio.wav"
OUT=/tmp/mtmd-audio-synth
ARCHS="${ARCHS:-qwen3a gemma4ua lfm2a}"
cd "$ROOT"

# 1. the synthetic mmprojs + the port's embedding dumps
if [ "${1:-}" == "--regen" ] || [ ! -f "$OUT/port-default-qwen3a.bin" ]; then
    cargo test -p llama --test mtmd_audio_synth2 -- --nocapture 2>&1 | tail -3
fi

python3 parity/gen_audio_fixture.py

# 2. the reference accepts each arch and dumps its audio embeddings
status=0
for arch in $ARCHS; do
    MMPROJ="$OUT/mmproj-audio-synth-$arch.gguf"
    if [ ! -f "$MMPROJ" ]; then
        echo "== $arch: synthetic mmproj missing ($MMPROJ)"; status=1; continue
    fi
    MTMD_DEBUG_EMBEDDINGS="$OUT/ref-default-$arch.bin" \
        "$REF/llama-mtmd-cli" -m "$TEXT" --mmproj "$MMPROJ" \
        --audio "$WAV" -p "Describe the sound." -n 8 --temp 0 -t "$(nproc)" \
        >"$OUT/ref_mtmd_cli2-$arch.log" 2>&1
    rc=$?
    MTMD_DEBUG_EMBEDDINGS="$OUT/ref-faoff-$arch.bin" \
        "$REF/llama-mtmd-cli" -m "$TEXT" --mmproj "$MMPROJ" \
        --audio "$WAV" -p "Describe the sound." -n 8 --temp 0 -fa off -t "$(nproc)" \
        >>"$OUT/ref_mtmd_cli2-$arch.log" 2>&1
    rc_off=$?
    echo "reference llama-mtmd-cli [$arch] exit: $rc (fa on) / $rc_off (-fa off)"
    if [ $rc -ne 0 ] || [ $rc_off -ne 0 ]; then
        echo "== reference REJECTED the synthetic $arch projector =="
        tail -20 "$OUT/ref_mtmd_cli2-$arch.log"
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
elif arch == "lfm2a":
    maxd = max(abs(a - b) for a, b in zip(rv, pv))
    if maxd < 1e-4:
        print(f"[{arch}/{tag}] NEAR-EXACT (documented): {diff}/{n} differ, max |delta| = {maxd:.3e}")
    else:
        print(f"[{arch}/{tag}] OUT OF BAND: {diff}/{n} differ, max |delta| = {maxd:.3e}")
        sys.exit(1)
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
    cargo test -p llama --test mtmd_audio_synth2 2>&1 | tail -2 | tee /tmp/audio-parity2-test.log
    grep -q "test result: ok" /tmp/audio-parity2-test.log || status=1
fi

exit $status
