#!/usr/bin/env bash
# tts_pipeline_parity.sh — the TTS pipeline LAYER e2e against the pinned
# reference (bd4f514db1): the port's tests/tts_pipeline_e2e.rs drives
# Qwen3TtsGenPipeline / PocketttsGenPipeline (mtmd.rs's port of
# mtmd-helper-gen.cpp) over a synthetic dflash trunk + the proven gen
# mmprojs, and this script hands the same files + the same inputs to the
# reference through parity/ref_tts_pipeline.cpp — the exact
# mtmd_helper_gen_audio_* calls tools/tts/tts.cpp makes.
#
# Compared per pipeline × {pcm, wav}:
#   * the per-step trunk hidden states (prompt-final row first) — the
#     h_state chain through the injection graph
#   * the PCM (floats, numeric band) and the WAV (bytes, exact — the RIFF
#     header + PCM16 conversion)
#   * the trunk's final sequence state (llama_state_seq_get_data, seq 0) —
#     the K/V the injection wrote, the position-bookkeeping cross-check
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFBIN=/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin
OUT=/tmp/tts2-pipe
cd "$ROOT"

# build the probe if missing
if [ ! -x parity/ref_tts_pipeline ] || [ parity/ref_tts_pipeline.cpp -nt parity/ref_tts_pipeline ]; then
    # explicit .so paths: this box's ld mis-resolves the -l names
    # (libllama/libggml's double-l) — the files themselves link fine
    g++ -O2 -std=c++17 -o parity/ref_tts_pipeline parity/ref_tts_pipeline.cpp \
        -I/home/jeffrey/llm/llama.cpp-pinned/include \
        -I/home/jeffrey/llm/llama.cpp-pinned/tools/mtmd \
        -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
        -Wl,-rpath,"$REFBIN" \
        "$REFBIN/libmtmd.so.0" "$REFBIN/libllama.so.0" \
        "$REFBIN/libggml.so.0" "$REFBIN/libggml-base.so.0" \
        "$REFBIN/libggml-cpu.so.0" || exit 1
fi

# 1. regenerate the port-side fixtures + dumps (writes $OUT/{trunk,mmproj}*.gguf,
#    port-*-{pcm,wav}.bin, q3t-sampled.bin)
./parity/limited.sh -- cargo test -p llama --test tts_pipeline_e2e -- --nocapture 2>&1 | tail -3

status=0

Q3T_PROMPT="The quick brown fox jumps over the lazy dog."
PT_PROMPT="Pocket speech rolls several short words into one steady clause and then keeps rolling through more words before the first question mark finally lands right here? A second clause then follows the same easy rhythm and adds its own words until its own ending arrives at the very last line."

run_ref() { # tag mode outtype prompt sampled-file
    local tag="$1" mode="$2" outty="$3" prompt="$4" sampled="$5"
    local extra=()
    [ -n "$sampled" ] && extra+=("$OUT/$sampled")
    if ! "$ROOT/parity/ref_tts_pipeline" \
         "$OUT/trunk-$tag.gguf" "$OUT/mmproj-$([ "$tag" = q3t ] && echo qwen3tts-gen || echo pockettts-gen).gguf" \
         "$mode" "$outty" "$prompt" "en" 42 20 0.9 "$OUT/ref-$tag" \
         "${extra[@]+"${extra[@]}"}" >"$OUT/ref-$tag-$outty.log" 2>&1; then
        echo "== $tag [$outty]: reference REJECTED (exit $?) =="
        tail -5 "$OUT/ref-$tag-$outty.log"
        status=1
    else
        echo "reference accepted $tag [$outty]: $(grep -o '[0-9]* steps' "$OUT/ref-$tag-$outty.log" | head -1)"
    fi
}

run_ref q3t qwen3tts pcm  "$Q3T_PROMPT" q3t-sampled.bin
run_ref q3t qwen3tts wav  "$Q3T_PROMPT" q3t-sampled.bin
run_ref pt  pockettts pcm "$PT_PROMPT" ""
run_ref pt  pockettts wav "$PT_PROMPT" ""

# ---------------------------------------------------------------------------
# comparison — h rows + PCM at a numeric band, WAV bytes exact, state exact
# ---------------------------------------------------------------------------
python3 - <<'PY'
import struct, sys, os

OUT = "/tmp/tts2-pipe"

def read_pcm_dump(path):
    d = open(path, "rb").read()
    off = 0
    def u32():
        nonlocal off
        v = struct.unpack_from("<I", d, off)[0]; off += 4; return v
    n_prompt = u32()           # port-side info (the reference has no accessor)
    n_h, h_len = u32(), u32()
    hs = list(struct.unpack_from("<" + "f" * (n_h * h_len), d, off)); off += 4 * n_h * h_len
    n_pcm = u32()
    pcm = list(struct.unpack_from("<" + "f" * n_pcm, d, off)); off += 4 * n_pcm
    state_len = u32()
    state = d[off:off + state_len]; off += state_len
    return dict(n_prompt=n_prompt, n_h=n_h, h_len=h_len, hs=hs, pcm=pcm, state=state, rest=len(d)-off)

def band(name, r, p, what, lim=2e-3):
    bad = 0
    if len(r) != len(p):
        print(f"{name}: {what} length {len(r)} vs {len(p)}"); return 1
    if not r:
        print(f"{name}: {what} empty"); return 1
    md = max(abs(a - b) for a, b in zip(r, p))
    ex = sum(1 for a, b in zip(r, p) if struct.pack("<f", a) == struct.pack("<f", b))
    print(f"{name}: {what} n={len(r)} bitexact={ex}/{len(r)} max|d|={md:.3e}")
    if md > lim:
        print(f"{name}: {what} exceeds band"); bad = 1
    return bad

bad = 0
for tag in ("q3t", "pt"):
    rp, pp = f"{OUT}/ref-{tag}-pcm.bin", f"{OUT}/port-{tag}-pcm.bin"
    if not (os.path.exists(rp) and os.path.exists(pp)):
        print(f"{tag}: SKIP (dump missing)"); bad = 1; continue
    r, p = read_pcm_dump(rp), read_pcm_dump(pp)
    print(f"{tag}: n_prompt(port)={p['n_prompt']} n_h ref={r['n_h']} port={p['n_h']} "
          f"h_len ref={r['h_len']} port={p['h_len']}")
    if r["n_h"] != p["n_h"] or r["h_len"] != p["h_len"]:
        print(f"{tag}: shape mismatch"); bad = 1; continue
    bad |= band(tag, r["hs"], p["hs"], "h-states")
    bad |= band(tag, r["pcm"], p["pcm"], "pcm")
    if r["state"] == p["state"]:
        print(f"{tag}: trunk state bit-exact ({len(r['state'])} bytes)")
    else:
        if len(r["state"]) != len(p["state"]):
            print(f"{tag}: trunk state size {len(r['state'])} vs {len(p['state'])}"); bad = 1
        else:
            rs = struct.unpack("<" + "H" * (len(r["state"]) // 2), r["state"])
            ps = struct.unpack("<" + "H" * (len(p["state"]) // 2), p["state"])
            diff = sum(1 for a, b in zip(rs, ps) if a != b)
            print(f"{tag}: trunk state differs in {diff}/{len(rs)} u16 words")
            bad = 1

    rw, pw = f"{OUT}/ref-{tag}-wav.bin", f"{OUT}/port-{tag}-wav.bin"
    if os.path.exists(rw) and os.path.exists(pw):
        rwb = open(rw, "rb").read(); pwb = open(pw, "rb").read()
        if rwb == pwb:
            print(f"{tag}: wav byte-exact ({len(rwb)} bytes)")
        else:
            rr = struct.unpack_from("<II", rwb, 0); pr = struct.unpack_from("<II", pwb, 0)
            print(f"{tag}: wav DIFFERS (header ref={rr} port={pr}, {len(rwb)} vs {len(pwb)} bytes)")
            bad = 1

sys.exit(bad)
PY
rc=$?
[ $rc -ne 0 ] && status=1
echo "tts_pipeline_parity: exit $status (comparison rc $rc)"
exit $status
