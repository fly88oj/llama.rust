#!/usr/bin/env bash
# audio_flac_parity.sh — Task 3 (FLAC): the port's minimal FLAC decoder vs
# the reference's miniaudio/dr_flac path. ffmpeg encodes the 16 kHz fixture
# into FLACs (mono/stereo, 16/24-bit, several block sizes); the pinned
# reference llama-mtmd-cli decodes each with a bit-exact synthetic mmproj
# (gemma4ua, round 2 — raw-waveform, sensitive to every PCM bit), and the
# port does the same; MTMD_DEBUG_EMBEDDINGS dumps are compared bit-exactly.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF=/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin
TEXT="${TEXT:-/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
OUT=/tmp/mtmd-flac
cd "$ROOT"

python3 parity/gen_audio_fixture.py

# the port side: the cargo test decodes + encodes each fixture with the
# synthetic gemma4ua mmproj and dumps its embeddings
MTMD_DUMP_FLAC_EMBD_DIR="$OUT" cargo test -p llama --test mtmd_flac -- --nocapture 2>&1 | tail -2

status=0
declare -A ARGS=(
    [mono16]="-ac 1 -ar 16000"
    [mono16-b192]="-ac 1 -ar 16000 -blocksize 192"
    [mono16-b4096]="-ac 1 -ar 16000 -blocksize 4096"
    [mono16-l8]="-ac 1 -ar 16000 -compression_level 8"
    [mono24]="-ac 1 -ar 16000 -sample_fmt s32"
    [stereo16]="-ac 2 -ar 16000"
)
for tag in mono16 mono16-b192 mono16-b4096 mono16-l8 mono24 stereo16; do
    FLAC="$OUT/fixture-$tag.flac"
    # 1. the reference decodes the FLAC through miniaudio/dr_flac
    MTMD_DEBUG_EMBEDDINGS="$OUT/ref-$tag.bin" \
        "$REF/llama-mtmd-cli" -m "$TEXT" --mmproj "$OUT/mmproj-gemma4ua.gguf" \
        --audio "$FLAC" -p "Describe." -n 4 --temp 0 -t 8 \
        >"$OUT/ref-$tag.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        tail -5 "$OUT/ref-$tag.log"; status=1; continue
    fi
    python3 - "$tag" "$OUT/ref-$tag.bin" "$OUT/port-$tag.bin" <<'PYEOF'
import struct, sys
tag, rp, pp = sys.argv[1], sys.argv[2], sys.argv[3]
def load(p):
    d = open(p, 'rb').read()
    n, e = struct.unpack('<ii', d[:8])
    return n, e, struct.unpack(f'<{n*e}f', d[8:])
rn, re_, rv = load(rp); pn, pe, pv = load(pp)
assert (rn, re_) == (pn, pe), f"shape {rn}x{re_} vs {pn}x{pe}"
diff = sum(1 for a, b in zip(rv, pv) if a != b)
if diff == 0:
    print(f"[{tag}] BIT-EXACT: {len(rv)} embedding values identical ({rn} tokens)")
else:
    print(f"[{tag}] NOT bit-exact: {diff}/{len(rv)} differ")
    sys.exit(1)
PYEOF
    [ $? -ne 0 ] && status=1
done
exit $status
