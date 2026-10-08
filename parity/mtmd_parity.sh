#!/usr/bin/env bash
# Vision + multimodal parity: port (llama::clip / llama::mtmd) vs the reference
# mtmd binaries (bd4f514db1). See PARITY.md's mtmd section for the numbers.
#
# Usage: parity/mtmd_parity.sh [--quick]
#   --quick   reuse the existing reference dumps under /tmp instead of
#             re-running the reference binaries (they take minutes)
#
# Models (local):
#   TEXT=/home/jeffrey/.lmstudio/models/Jackrong/Qwen3.8-27B-MTP-GGUF/Qwen3.8-27B-MTP-Q4_K_M.gguf
#   MMPROJ=/home/jeffrey/.lmstudio/models/Jackrong/Qwen3.8-27B-MTP-GGUF/mmproj-F32.gguf
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REF=/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin
TEXT=${TEXT:-/home/jeffrey/.lmstudio/models/Jackrong/Qwen3.8-27B-MTP-GGUF/Qwen3.8-27B-MTP-Q4_K_M.gguf}
MMPROJ=${MMPROJ:-/home/jeffrey/.lmstudio/models/Jackrong/Qwen3.8-27B-MTP-GGUF/mmproj-F32.gguf}
PORT_CLI="$ROOT/target/release/llama-mtmd-cli"
QUICK=${1:-}

cd "$ROOT"

# ---------------------------------------------------------------------------
# 0. fixture images (identical bytes for both sides)
# ---------------------------------------------------------------------------
python3 parity/gen_mtmd_fixture.py

# ---------------------------------------------------------------------------
# 1. encoder-only parity: the `cb` pattern of llama-mtmd-debug fed straight to
#    clip_image_encode (no preprocessing, no image file decoding)
# ---------------------------------------------------------------------------
if [ "$QUICK" != "--quick" ]; then
    echo "== reference: cb 448 (FA off) =="
    MTMD_DEBUG_EMBEDDINGS=/tmp/ref_cb_448_noFA.bin \
        "$REF/llama-mtmd-debug" -m "$TEXT" --mmproj "$MMPROJ" -p encode -n 448 \
        --image cb -t 8 -fa off > /tmp/ref_cb_448_noFA.log 2>&1
    echo "== reference: cb 448 (FA auto -> enabled) =="
    MTMD_DEBUG_EMBEDDINGS=/tmp/ref_cb_448.bin \
        "$REF/llama-mtmd-debug" -m "$TEXT" --mmproj "$MMPROJ" -p encode -n 448 \
        --image cb -t 8 > /tmp/ref_cb_448.log 2>&1
fi
echo "== port: cb 448 =="
MTMD_FA=off MTMD_CB_SIZE=448 \
    cargo test -p llama --release --lib clip::tests::clip_cb_parity_dump -- --ignored --nocapture \
    > /tmp/rust_cb_noFA.log 2>&1
cp /tmp/rust_cb_448.bin /tmp/rust_cb_off.bin
MTMD_FA=on MTMD_CB_SIZE=448 \
    cargo test -p llama --release --lib clip::tests::clip_cb_parity_dump -- --ignored --nocapture \
    > /tmp/rust_cb_on.log 2>&1
cp /tmp/rust_cb_448.bin /tmp/rust_cb_on.bin

# ---------------------------------------------------------------------------
# 2. end-to-end parity: PNG decode + preprocessing + encoder + text model
# ---------------------------------------------------------------------------
if [ "$QUICK" != "--quick" ]; then
    echo "== reference: llama-mtmd-cli on parity/mtmd-fixture.png =="
    MTMD_DEBUG_EMBEDDINGS=/tmp/ref_fixture_embd.bin \
        "$REF/llama-mtmd-cli" -m "$TEXT" --mmproj "$MMPROJ" \
        --image parity/mtmd-fixture.png -p "Describe this image." -n 24 \
        --temp 0 -fa off -t 8 > /tmp/ref_fixture.log 2>&1
fi
echo "== port: llama-mtmd-cli on parity/mtmd-fixture.png =="
"$PORT_CLI" -m "$TEXT" --mmproj "$MMPROJ" --image parity/mtmd-fixture.png \
    -p "Describe this image." -n 24 --temp 0 -fa off -t 8 \
    --dump-embd /tmp/rust_fixture_embd.bin > /tmp/rust_fixture.log 2>&1
echo "== port, with the reference's own embeddings injected (text-path control) =="
"$PORT_CLI" -m "$TEXT" --mmproj "$MMPROJ" --image parity/mtmd-fixture.png \
    -p "Describe this image." -n 24 --temp 0 -fa off -t 8 \
    --embd-file /tmp/ref_fixture_embd.bin > /tmp/rust_fixture_refembd.log 2>&1

# ---------------------------------------------------------------------------
# 3. report
# ---------------------------------------------------------------------------
python3 - "$ROOT" <<'PY'
import math, re, struct, sys

def load(p):
    b = open(p, "rb").read()
    nt, ne = struct.unpack("<ii", b[:8])
    return nt, ne, struct.unpack("<%df" % ((len(b) - 8) >> 2), b[8:])

def cmp(ref, port, label):
    nt, ne, a = load(ref)
    nt2, ne2, b = load(port)
    assert (nt, ne) == (nt2, ne2), f"{label}: shape mismatch {(nt, ne)} vs {(nt2, ne2)}"
    nbit = sum(1 for x, y in zip(a, b) if struct.pack("<f", x) == struct.pack("<f", y))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
    d = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    cos = sum(x * y for x, y in zip(a, b)) / (na * nb)
    print(f"  {label}: {nt}x{ne}; bitwise {nbit}/{len(a)}; "
          f"L2 rel {d/na:.3e}; cos {cos:.10f}; max abs {max(abs(x-y) for x,y in zip(a,b)):.4g}")

print("== vision embeddings (reference vs port) ==")
try:
    cmp("/tmp/ref_cb_448_noFA.bin", "/tmp/rust_cb_off.bin", "cb 448, FA off")
    cmp("/tmp/ref_cb_448.bin", "/tmp/rust_cb_on.bin", "cb 448, FA on ")
    cmp("/tmp/ref_fixture_embd.bin", "/tmp/rust_fixture_embd.bin", "fixture PNG (preproc+encode)")
except OSError as e:
    print("  skipped:", e)

def gen_text(path, ref):
    try:
        txt = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return "(missing %s)" % path
    if ref:
        lines = txt.split("\n")
        idx = [i for i, l in enumerate(lines) if re.match(r"^\d+\.\d+\.\d+\.\d+ [IWED] ", l)]
        return "\n".join(lines[idx[-1] + 1:]).split("EXIT=")[0].strip() if idx else "?"
    m = re.search(r'text = ("(?:[^"\\]|\\.)*")', txt)
    if not m:
        return "?"
    # the CLI logs the text as a Rust literal: unescape it
    return m.group(1)[1:-1].replace("\\n", "\n").replace('\\"', '"')

print("== end-to-end generation (greedy, -n 24, -fa off) ==")
r = gen_text("/tmp/ref_fixture.log", True)
p = gen_text("/tmp/rust_fixture.log", False)
q = gen_text("/tmp/rust_fixture_refembd.log", False)
print("  reference        :", r)
print("  port             :", p)
print("  port + ref embd  :", q)
# longest common prefix in characters
n = 0
for x, y in zip(r, p):
    if x != y:
        break
    n += 1
print(f"  common prefix: {n} chars")
print("  text-path control (same divergence with the reference embeddings?) :",
      "YES" if p == q else "no")
PY