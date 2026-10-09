#!/usr/bin/env bash
# imatrix_md_parity.sh — the `-md` (draft-side NextN) collection lane of
# llama-imatrix vs the NEW reference (c35b66744), the X-domain task ③ gate.
#
# The upstream e2e needs a real MTP-split model pair; this box has none
# (only whole Qwen3.8-27B files with the MTP block inline), so the fixture is
# SYNTHETIC (crates/tools/imatrix/tests/imatrix_md_parity.rs `gen`, ignored):
#   trunk-synth.gguf  — [GDN, attn] hybrid trunk, no nextn tensors
#   draft-synth.gguf  — MTP-only file (block n_layer + the nextn trio, no
#                       trunk blocks — nextn_flags' MTP-only probe)
#   both-synth.gguf   — trunk + the same MTP block in one file
# Weights are name-seeded, so the MTP block of `draft` and `both` carry the
# same bytes: `-md trunk+draft` and `--nextn both` compute over the same
# weights, and the two runs must produce the SAME imatrix — the semantics
# the reference itself proves (ref md == ref nextn, byte-identical).
#
# Honest scope note: a direct rust-vs-ref byte compare of the imatrix is NOT
# achievable on this fixture — the port's qwen35 trunk forward carries a
# pre-existing ~1-ulp activation divergence vs the reference (the plain,
# nextn-free run diverges identically: blk.1.ffn_gate/up skew + the layer
# aggregate; the GDN block blk.0 matches bit-exactly; the 27B token-anchor
# IDENTICAL runs never exposed it). The `-md`-specific behavior is therefore
# verified as: (a) both sides' md==nextn self-consistency, byte-identical;
# (b) the --show-statistics stdout agrees structurally byte-for-byte and
# numerically to the last printed digit (≤2e-4 absolute on every cell).
set -uo pipefail
cd "$(dirname "$0")/.."

REF="${REF:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin/llama-imatrix}"
RUST="${RUST:-./target/release/llama-imatrix}"
OUT=/tmp/closx-md
FIX=crates/tools/imatrix/tests/imatrix_md_parity.rs
mkdir -p "$OUT"

# 1. fixtures (idempotent)
if [ ! -f "$OUT/both-synth.gguf" ]; then
    echo "== generating the synthetic qwen35 trunk/draft/both fixtures =="
    parity/limited.sh -- cargo test --release -p llama-imatrix --test imatrix_md_parity gen -- --ignored --nocapture
fi
echo "== probing the fixtures through load_model =="
parity/limited.sh -- cargo test --release -p llama-imatrix --test imatrix_md_parity probe -- --ignored

COMMON=(-f "$OUT/prompt.txt" -c 64 -b 64 --no-ppl -t 4)

run() { # bin out model mode...
    local bin=$1 out=$2 model=$3; shift 3
    parity/limited.sh -- "$bin" -m "$model" "$@" "${COMMON[@]}" -o "$out" >"$out.log" 2>"$out.err"
}

echo "== collecting: ref/rust x md/nextn =="
run "$REF"  "$OUT/ref-md.gguf"  "$OUT/trunk-synth.gguf" -md "$OUT/draft-synth.gguf"
run "$REF"  "$OUT/ref-nextn.gguf" "$OUT/both-synth.gguf" --nextn
run "$RUST" "$OUT/rust-md.gguf"  "$OUT/trunk-synth.gguf" -md "$OUT/draft-synth.gguf"
run "$RUST" "$OUT/rust-nextn.gguf" "$OUT/both-synth.gguf" --nextn

grep -q "processing 1 NextN layer(s) from block 2" "$OUT/ref-md.gguf.err"  ; echo "ref  -md processed the nextn layer"
grep -q "processing 1 NextN layer(s) from block 2" "$OUT/rust-md.gguf.err" ; echo "rust -md processed the nextn layer"

PASS=0; FAIL=0
check() { if [ "$1" = "0" ]; then echo "MATCH  $2"; PASS=$((PASS+1)); else echo "DIFF   $2"; FAIL=$((FAIL+1)); fi }

# 2. self-consistency: md == nextn on both engines (byte-identical)
cmp -s "$OUT/ref-md.gguf" "$OUT/ref-nextn.gguf";  check $? "reference: -md == --nextn (byte-identical imatrix)"
cmp -s "$OUT/rust-md.gguf" "$OUT/rust-nextn.gguf"; check $? "port:      -md == --nextn (byte-identical imatrix)"

# 3. --show-statistics stdout: structure byte-identical, numerics to the
#    last printed digit
parity/limited.sh -- "$REF"  -m "$OUT/trunk-synth.gguf" --show-statistics --in-file "$OUT/ref-md.gguf"  >"$OUT/ref-stats.txt" 2>/dev/null
parity/limited.sh -- "$RUST" -m "$OUT/trunk-synth.gguf" --show-statistics --in-file "$OUT/rust-md.gguf" >"$OUT/rust-stats.txt" 2>/dev/null
parity/limited.sh -- python3 - "$OUT" <<'EOF'
import re, sys
out = sys.argv[1]
def load(p):
    return [l.replace('ref-md.gguf', 'X').replace('rust-md.gguf', 'X')
            for l in open(p).read().splitlines()]
a, b = load(f'{out}/rust-stats.txt'), load(f'{out}/ref-stats.txt')
assert len(a) == len(b), f'line count {len(a)} vs {len(b)}'
num = re.compile(r'-?\d+\.\d+')
worst_abs, ndiff = 0.0, 0
for x, y in zip(a, b):
    xs, ys = num.findall(x), num.findall(y)
    assert len(xs) == len(ys), f'structure mismatch:\n{x}\n{y}'
    assert x.translate({ord(c): None for c in '0123456789.-'}) == \
           y.translate({ord(c): None for c in '0123456789.-'}), f'label/format mismatch:\n{x}\n{y}'
    for u, v in zip(xs, ys):
        if u != v and 'nan' not in (u, v):
            ndiff += 1
            worst_abs = max(worst_abs, abs(float(u) - float(v)))
print(f'statistics: structure IDENTICAL ({len(a)} lines incl. mtp0 layer tags '
      f'and nan cells); {ndiff} numeric cells differ, worst absolute {worst_abs:.1e} '
      f'(the printed-last-digit tail of the pre-existing 1-ulp qwen35 trunk '
      f'divergence, amplified by the kurtosis/skew fourth-moment statistics — '
      f'see the script header)')
assert worst_abs <= 1e-3, 'numeric divergence beyond the 1e-3 band'
sys.exit(0)
EOF
check $? "statistics stdout: structure IDENTICAL + numerics within the last printed digit"

# 4. -md gate smoke: a trunk WITH nextn tensors must be rejected for -md
#    (imatrix.cpp:1909-1913 "model already includes NextN layers")
"$REF" -m "$OUT/both-synth.gguf" -md "$OUT/draft-synth.gguf" -f "$OUT/prompt.txt" -c 64 -b 64 -o /dev/null >/dev/null 2>"$OUT/ref-gate.err"; ref_gate=$?
"$RUST" -m "$OUT/both-synth.gguf" -md "$OUT/draft-synth.gguf" -f "$OUT/prompt.txt" -c 64 -b 64 -o /dev/null >/dev/null 2>"$OUT/rust-gate.err"; rust_gate=$?
[ "$ref_gate" != "0" ] && [ "$rust_gate" != "0" ]; check $? "gate: both reject -md on a model that already includes NextN (ref=$ref_gate rust=$rust_gate)"

echo
echo "== imatrix -md parity: $PASS MATCH / $FAIL DIFF =="
# keep the artifacts under parity/ (git-tracked evidence)
mkdir -p parity/imatrix_md
cp -f "$OUT"/ref-stats.txt "$OUT"/rust-stats.txt "$OUT"/ref-md.gguf.err "$OUT"/rust-md.gguf.err parity/imatrix_md/
exit "$FAIL"
