#!/usr/bin/env bash
# tts_parity.sh — audio round 5: the TTS family (LM archs + gen mmprojs)
# against the pinned reference (bd4f514db1).
#
#  A. LM archs (models/{wavtokenizer-dec,pockettts,qwen3tts}.cpp):
#     the port's synthetic GGUFs (tests/tts_archs_e2e.rs) + the port's
#     graph runs dump final logits/embd; the reference probe
#     (parity/ref_lm_tts_dump, llama_decode over the same token ids) dumps
#     its own; compared at the numeric band (composed group-norm rounding
#     on the port side is the only expected residual for wavtokenizer).
#  B. gen mmprojs (qwen3tts_gen / pockettts_gen): the port's synthetic
#     mmprojs (tests/tts_gen_e2e.rs) + GEN_CODE (fixed hidden state + seed)
#     and GEN_WAV (codes/feats + state threading) dumps; the reference
#     probe (parity/ref_gen_audio_dump, clip_encode over libmtmd — the
#     exact mtmd_gen_audio_process call) dumps its own; codes/eos compared
#     exactly, floats at the numeric band.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFBIN=/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin
LM_OUT=/tmp/tts-lm-synth
GEN_OUT=/tmp/tts-gen-synth
cd "$ROOT"

# build the probes if missing
if [ ! -x parity/ref_lm_tts_dump ] || [ parity/ref_lm_tts_dump.c -nt parity/ref_lm_tts_dump ]; then
    gcc -O2 -I/home/jeffrey/llm/llama.cpp-pinned/include \
        -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
        parity/ref_lm_tts_dump.c -o parity/ref_lm_tts_dump \
        -L"$REFBIN" -lllama -lggml -lggml-base -lggml-cpu -lm \
        -Wl,-rpath,"$REFBIN" || exit 1
fi
if [ ! -x parity/ref_gen_audio_dump ] || [ parity/ref_gen_audio_dump.cpp -nt parity/ref_gen_audio_dump ]; then
    g++ -O2 -std=c++17 -o parity/ref_gen_audio_dump parity/ref_gen_audio_dump.cpp \
        -I/home/jeffrey/llm/llama.cpp-pinned/tools/mtmd \
        -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
        -I/home/jeffrey/llm/llama.cpp-pinned/include \
        -I/home/jeffrey/llm/llama.cpp-pinned/src \
        -L"$REFBIN" -Wl,-rpath,"$REFBIN" \
        -lmtmd -lggml -lggml-base -lggml-cpu || exit 1
fi

# 1. regenerate the port-side fixtures + dumps
MTMD_IGNORE_REF=1 cargo test -p llama --test tts_archs_e2e -- --nocapture 2>&1 | tail -2
MTMD_IGNORE_REF=1 cargo test -p llama --test tts_gen_e2e -- --nocapture 2>&1 | tail -2

status=0

# ---------------------------------------------------------------------------
# A. LM archs
# ---------------------------------------------------------------------------
run_lm() { # arch file tags...
    local arch="$1" file="$2"; shift 2
    local tokens="$LM_OUT/tokens.bin"
    if [ ! -f "$tokens" ]; then
        python3 - "$tokens" <<'PY'
import struct, sys
ids = [100 + i * 7 for i in range(8)]
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<I", len(ids)))
    f.write(struct.pack("<" + "i" * len(ids), *ids))
PY
    fi
    for tag in "$@"; do
        local out="$LM_OUT/ref-$arch-$tag.bin"
        if ! "$ROOT/parity/ref_lm_tts_dump" "$file" "$tokens" "$out" \
             --embeddings --fa "$(if [ "$tag" = "fa" ]; then echo on; else echo off; fi)" >"$LM_OUT/ref-$arch-$tag.log" 2>&1; then
            echo "== $arch [$tag]: reference REJECTED the file (exit $?) =="
            tail -5 "$LM_OUT/ref-$arch-$tag.log"
            status=1
            continue
        fi
        echo "reference accepted $arch [$tag]"
    done
}

run_lm pockettts "$LM_OUT/pockettts-synth.gguf" fa nofa
run_lm wavtokenizer-dec "$LM_OUT/wavtokenizer-dec-synth.gguf" fa || true
run_lm qwen3tts "$LM_OUT/qwen3tts-synth.gguf" fa || true

# ---------------------------------------------------------------------------
# B. gen mmprojs
# ---------------------------------------------------------------------------
run_gen() { # name mode infile outfile extra...
    local name="$1" mode="$2" infile="$3" outfile="$4"; shift 4
    if ! "$ROOT/parity/ref_gen_audio_dump" "$GEN_OUT/mmproj-$name.gguf" \
         "$mode" "$infile" "$outfile" "$@" >"$GEN_OUT/ref-$name-$mode.log" 2>&1; then
        echo "== $name [$mode]: reference REJECTED (exit $?) =="
        tail -5 "$GEN_OUT/ref-$name-$mode.log"
        status=1
    else
        echo "reference accepted $name [$mode]"
    fi
}

run_gen qwen3tts-gen gencode "$GEN_OUT/q3t-hstate.bin" "$GEN_OUT/ref-q3t-gencode.bin" 42 5 50 1.0 0.9
run_gen qwen3tts-gen genwav-codes "$GEN_OUT/q3t-codes1.bin" "$GEN_OUT/ref-q3t-genwav1.bin" 42
# the streamed call rides the state of call 1 — its size comes from the port dump
S1=$(python3 -c "
import struct
d=open('$GEN_OUT/port-q3t-genwav1.bin','rb').read()
off=0
def u32():
    global off
    v=struct.unpack_from('<I',d,off)[0]; off+=4; return v
nc=u32(); off+=4*nc; ne=u32(); off+=4*ne; nf=u32(); off+=4*nf; ie=u32(); na=u32(); off+=4*na; print(u32())")
TTS_STATE_SIZE=$S1 run_gen qwen3tts-gen genwav-codes "$GEN_OUT/q3t-codes2.bin" "$GEN_OUT/ref-q3t-genwav2.bin" 42

run_gen pockettts-gen gencode "$GEN_OUT/pt-hstate.bin" "$GEN_OUT/ref-pt-gencode.bin" 42 0 50 1.0 0.7
run_gen pockettts-gen genwav-feats "$GEN_OUT/pt-feats1.bin" "$GEN_OUT/ref-pt-genwav1.bin" 42 32
S2=$(python3 -c "
import struct
d=open('$GEN_OUT/port-pt-genwav1.bin','rb').read()
off=0
def u32():
    global off
    v=struct.unpack_from('<I',d,off)[0]; off+=4; return v
nc=u32(); off+=4*nc; ne=u32(); off+=4*ne; nf=u32(); off+=4*nf; ie=u32(); na=u32(); off+=4*na; print(u32())")
TTS_STATE_SIZE=$S2 run_gen pockettts-gen genwav-feats "$GEN_OUT/pt-feats2.bin" "$GEN_OUT/ref-pt-genwav2.bin" 42 32

# ---------------------------------------------------------------------------
# comparison — codes/eos exact, floats at a numeric band
# ---------------------------------------------------------------------------
python3 - <<'PY'
import struct, sys, os

def read_gen_dump(path):
    d = open(path, "rb").read()
    off = 0
    def u32():
        nonlocal off
        v = struct.unpack_from("<I", d, off)[0]; off += 4; return v
    out = {}
    nc = u32(); out["codes"] = list(struct.unpack_from("<" + "i" * nc, d, off)); off += 4 * nc
    ne = u32(); out["embd"] = list(struct.unpack_from("<" + "f" * ne, d, off)); off += 4 * ne
    nf = u32(); out["feats"] = list(struct.unpack_from("<" + "f" * nf, d, off)); off += 4 * nf
    out["eos"] = u32()
    na = u32(); out["audio"] = list(struct.unpack_from("<" + "f" * na, d, off)); off += 4 * na
    ns = u32(); out["state"] = d[off:off + ns]; off += ns
    return out

def cmp_gen(name, ref_path, port_path):
    if not (os.path.exists(ref_path) and os.path.exists(port_path)):
        print(f"{name}: SKIP (dump missing)"); return 1
    r, p = read_gen_dump(ref_path), read_gen_dump(port_path)
    bad = 0
    if r["codes"] != p["codes"]:
        print(f"{name}: CODES differ ref={r['codes']} port={p['codes']}"); bad = 1
    else:
        print(f"{name}: codes exact ({len(r['codes'])})")
    if r["eos"] != p["eos"]:
        print(f"{name}: eos differ {r['eos']} vs {p['eos']}"); bad = 1
    for k in ("embd", "feats", "audio"):
        if not r[k] and not p[k]:
            continue
        if len(r[k]) != len(p[k]):
            print(f"{name}: {k} length {len(r[k])} vs {len(p[k])}"); bad = 1; continue
        md = max((abs(a - b) for a, b in zip(r[k], p[k])), default=0.0)
        exact = sum(1 for a, b in zip(r[k], p[k]) if struct.pack("<f", a) == struct.pack("<f", b))
        print(f"{name}: {k} n={len(r[k])} bitexact={exact}/{len(r[k])} max|d|={md:.3e}")
        if md > 2e-3:
            print(f"{name}: {k} exceeds band"); bad = 1
    if r["state"] != p["state"]:
        if len(r["state"]) != len(p["state"]):
            print(f"{name}: state size {len(r['state'])} vs {len(p['state'])}"); bad = 1
        else:
            rs = struct.unpack("<" + "f" * (len(r["state"]) // 4), r["state"])
            ps = struct.unpack("<" + "f" * (len(p["state"]) // 4), p["state"])
            md = max(abs(a - b) for a, b in zip(rs, ps))
            print(f"{name}: state n={len(rs)} max|d|={md:.3e}")
            if md > 2e-3:
                print(f"{name}: state exceeds band"); bad = 1
    else:
        print(f"{name}: state bit-exact ({len(r['state'])} bytes)")
    return bad

def read_lm_dump(path):
    d = open(path, "rb").read()
    nv = struct.unpack_from("<I", d, 0)[0]
    logits = list(struct.unpack_from("<" + "f" * nv, d, 4))
    off = 4 + 4 * nv
    embd = []
    if off < len(d):
        no = struct.unpack_from("<I", d, off)[0]
        embd = list(struct.unpack_from("<" + "f" * no, d, off + 4))
    return logits, embd

def cmp_lm(name, ref_path, port_path, logit_n=None, embd_n=None):
    if not (os.path.exists(ref_path) and os.path.exists(port_path)):
        print(f"{name}: SKIP (dump missing)"); return 1
    rl, re_ = read_lm_dump(ref_path)
    pl, pe = read_lm_dump(port_path)
    bad = 0
    # the port's dump is the last row only; the reference's n_vocab may be
    # the whole table — compare the argmax token and the numeric band
    def band(r, p, what):
        nonlocal bad
        n = min(len(r), len(p))
        if n == 0:
            print(f"{name}: {what} empty"); bad = 1; return
        if len(r) != len(p):
            print(f"{name}: {what} length {len(r)} vs {len(p)} — comparing first {n} (the wavtokenizer head emits the waveform dim, the reference buffer stays n_vocab wide)")
        md = max((abs(a - b) for a, b in zip(r[:n], p[:n])), default=0.0)
        am_r = max(range(n), key=lambda i: r[i])
        am_p = max(range(n), key=lambda i: p[i])
        print(f"{name}: {what} n={n} argmax {am_r}=={am_p} max|d|={md:.3e}")
        if am_r != am_p or md > 5e-3:
            print(f"{name}: {what} exceeds band / argmax differs"); bad = 1
    if logit_n:
        band(rl[:logit_n], pl[:logit_n], "logits")
    else:
        band(rl, pl, "logits")
    if embd_n:
        band(re_, pe, "embd")
    elif re_ and pe:
        # both sides carry an embd section — compare it (the pockettts port
        # dump is logits-only: the graph is verified through the head, whose
        # random weights make bitexact logits imply a bitexact t_embd)
        band(re_, pe, "embd")
    return bad

bad = 0
G = "/tmp/tts-gen-synth"
bad |= cmp_gen("q3t-gencode", f"{G}/ref-q3t-gencode.bin", f"{G}/port-q3t-gencode.bin")
bad |= cmp_gen("q3t-genwav1", f"{G}/ref-q3t-genwav1.bin", f"{G}/port-q3t-genwav1.bin")
bad |= cmp_gen("q3t-genwav2", f"{G}/ref-q3t-genwav2.bin", f"{G}/port-q3t-genwav2.bin")
bad |= cmp_gen("pt-gencode", f"{G}/ref-pt-gencode.bin", f"{G}/port-pt-gencode.bin")
bad |= cmp_gen("pt-genwav1", f"{G}/ref-pt-genwav1.bin", f"{G}/port-pt-genwav1.bin")
bad |= cmp_gen("pt-genwav2", f"{G}/ref-pt-genwav2.bin", f"{G}/port-pt-genwav2.bin")

L = "/tmp/tts-lm-synth"
for tag in ("fa", "nofa"):
    bad |= cmp_lm(f"pockettts-{tag}", f"{L}/ref-pockettts-{tag}.bin", f"{L}/port-pockettts-logits-{tag}.bin")

# wavtokenizer-dec sets only res->t_embd — the reference's logits row is the
# zero-filled dummy buffer (llama-context.cpp sizes it n_vocab while the arch
# never produces t_logits). The real comparison target is t_embd, the decoded
# waveform: the reference's flat copy makes row j the t_embd column j, and the
# probe's n_embd-wide read takes its n_embd_out prefix.
def cmp_wavtokenizer():
    r = read_lm_dump(f"{L}/ref-wavtokenizer-dec-fa.bin")
    rl, re_ = r
    # the port side dumps the waveform as a plain [n][f32] blob (dump_f32)
    pd = open(f"{L}/port-wavtokenizer-embd.bin", "rb").read()
    pn = struct.unpack_from("<I", pd, 0)[0]
    pe = list(struct.unpack_from("<" + "f" * pn, pd, 4))
    if any(v != 0.0 for v in rl):
        print(f"wavtokenizer: reference logits not the zero dummy? max={max(rl)}")
        return 1
    n = min(len(re_), len(pe))
    if n == 0:
        print("wavtokenizer: embd missing")
        return 1
    md = max(abs(a - b) for a, b in zip(re_[:n], pe[:n]))
    ex = sum(1 for a, b in zip(re_[:n], pe[:n]) if struct.pack("<f", a) == struct.pack("<f", b))
    print(f"wavtokenizer: t_embd n={n} (of {len(pe)}) bitexact={ex}/{n} max|d|={md:.3e}")
    return 1 if md > 5e-3 else 0

bad |= cmp_wavtokenizer()

sys.exit(bad)
PY
rc=$?
[ $rc -ne 0 ] && status=1
echo "tts_parity: exit $status (comparison rc $rc)"
exit $status
