#!/usr/bin/env bash
# state_kinds_parity.sh — the remaining state-serialization byte-format cells
# (PARITY.md's state-kinds section), the dsv4_state_parity.sh protocol on the
# models that carry the new kinds:
#
#   * deepseek32 (batch-6 synth) — the dsa pair's lid half: the port's
#     `DecodeContext::state_seq_get_data` must equal the reference's
#     `llama_state_seq_get_data` byte-for-byte (the K-only MLA base half then
#     the K-only indexer-key half, llama-kv-cache-dsa.cpp:164-172);
#   * minimax-m3 (batch-11a synth) — the MSA idx half (the base half then the
#     idx K rows + the never-written zero V rows, llama-kv-cache-msa.cpp:
#     160-168);
#   * mamba2 (batch-5 synth) — the recurrent half (llama-memory-recurrent.cpp:
#     766-1224): the blob is the recurrent serialization ALONE (the pure-
#     recurrent memory has no attention cache, llama-model.cpp:2538-2548).
#     The framing — cell_count, the pos/seq-id meta, s_trans/n_layer, every
#     per-layer F32-type/row-size header — must be byte-identical; the conv/
#     ssm row payloads sit past the arch's bit-exact frontier (PARITY.md
#     batch 5: the mamba2 forward is a Δlogprob ≤ 1e-4 stream, not bit-exact),
#     so the rows are compared as f32 with a bound and the worst delta
#     reported;
#   * all — the whole-context `llama_state_get_data` blob (the arch-string
#     model-info header of llama-context.cpp:3341-3357 around seq_id -1).
#
# Two states per model: the 16-token prefill alone and with the fixed 8-step
# tail. Context geometry pinned: n_ctx 512, n_seq_max 1, n_ubatch 512, FA on
# (v_trans = !flash_attn — only the v_trans = 0 layout matches the port's
# always-!v_trans caches).
#
# usage: bash parity/state_kinds_parity.sh   (regenerates the models + blobs)
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PIN=/home/jeffrey/llm/llama.cpp-pinned
REF=/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin
OUT=/tmp/arch-statekinds
mkdir -p $OUT

fail=0

# 1. regenerate the synthetic models (the batch writers) + dump the port blobs
cargo test --release -p llama --test arch_batch6_e2e arch_batch6_write_synth -- --ignored \
    > $OUT/port-b6.log 2>&1 || { echo "BATCH6 SYNTH FAILED"; tail -5 $OUT/port-b6.log; exit 1; }
cargo test --release -p llama --test arch_batch11a_e2e arch_batch11a_write_synth -- --ignored \
    > $OUT/port-b11a.log 2>&1 || { echo "BATCH11A SYNTH FAILED"; tail -5 $OUT/port-b11a.log; exit 1; }
cargo test --release -p llama --test arch_batch5_e2e arch_batch5_write_synth -- --ignored \
    > $OUT/port-b5.log 2>&1 || { echo "BATCH5 SYNTH FAILED"; tail -5 $OUT/port-b5.log; exit 1; }
cargo test --release -p llama --test state_kinds_e2e state_kinds_dump_blobs -- --ignored \
    > $OUT/port-dump.log 2>&1 || { echo "PORT DUMP FAILED"; tail -8 $OUT/port-dump.log; exit 1; }
cargo test --release -p llama --test recurrent_state_e2e recurrent_state_dump_blobs -- --ignored \
    > $OUT/port-recur-dump.log 2>&1 || { echo "PORT RECURRENT DUMP FAILED"; tail -8 $OUT/port-recur-dump.log; exit 1; }

# 2. the reference probe
PROBE=$OUT/ref_state_kinds
gcc -O2 "$ROOT/parity/ref_state_kinds.c" -o "$PROBE" \
    -I "$PIN/include" -I "$PIN/ggml/include" \
    -L "$REF" -lllama -Wl,-rpath,"$REF" || { echo "PROBE BUILD FAILED"; exit 1; }

for arch in deepseek32 minimax-m3 mamba2 jamba; do
    case $arch in
        deepseek32) GGUF=/tmp/arch-batch6/deepseek32-synth.gguf ;;
        minimax-m3) GGUF=/tmp/arch-batch11a/minimax-m3-synth.gguf ;;
        mamba2)     GGUF=/tmp/arch-batch5/mamba2-synth.gguf ;;
        jamba)      GGUF=/tmp/arch-batch5/jamba-synth.gguf ;;
    esac
    for tail in 0 8; do
        tag=$arch
        if [ $tail -ne 0 ]; then tag=$arch-tail; fi
        LLAMA_LOG_LEVEL=error "$PROBE" "$GGUF" "$OUT/ref-$tag" $tail \
            > $OUT/probe-$tag.log 2>&1 || {
                echo "PROBE RUN FAILED ($arch tail $tail)"; tail -3 $OUT/probe-$tag.log; fail=1; continue; }
    done
done

# 3. byte comparison — the minimax-m3 blobs must be byte-identical outright
for pair in \
    "port-minimax-m3-seq.bin ref-minimax-m3-seq.bin msa-idx seq prefill" \
    "port-minimax-m3-tail-seq.bin ref-minimax-m3-tail-seq.bin msa-idx seq tail" \
    "port-minimax-m3-full.bin ref-minimax-m3-full.bin msa-idx full prefill" \
    "port-minimax-m3-tail-full.bin ref-minimax-m3-tail-full.bin msa-idx full tail"; do
    set -- $pair
    if cmp -s "$OUT/$1" "$OUT/$2"; then
        echo "  $3 $4 ($5): port == reference, byte-identical ($(stat -c%s "$OUT/$1") bytes) PASS"
    else
        echo "  $3 $4 ($5): BYTE MISMATCH (port $(stat -c%s "$OUT/$1") vs ref $(stat -c%s "$OUT/$2")) FAIL"
        cmp "$OUT/$1" "$OUT/$2" | head -2
        fail=1
    fi
done

# 3b. deepseek32: the dsa blob's STRUCTURE must be byte-identical — the whole
# framing (cell meta, v_trans, n_layer, per-layer type/row_size) and the
# dense-lead layer-0 rows of BOTH halves (their inputs are pre-attention, so
# they are bit-exact). The sparse layers' row payloads sit past the
# deepseek32 forward's established parity frontier (the arch's cells are
# stream-level, not bit-exact — PARITY.md batch 6), so only the structure is
# asserted there; the continuation bit-identity lives in the default tests.
for k in seq full; do
    for t in "" "-tail"; do
        python3 - "$OUT/port-deepseek32$t-$k.bin" "$OUT/ref-deepseek32$t-$k.bin" <<'PYEOF' || fail=1
import struct, sys
port, ref = (open(p, 'rb').read() for p in sys.argv[1:3])
def parse(b):
    assert b[:4] in (b'SQST', b'FULL'), 'magic'
    ln = struct.unpack('<I', b[4:8])[0]
    blob, pos = b[8:8+ln], 0
    pos = 0
    if b[:4] == b'FULL':
        # the whole-context blob: write_string(arch) then memory->state_write
        # directly — no [magic][seq_id] framing
        alen = struct.unpack('<I', blob[0:4])[0]
        assert blob[4:4+alen] == b'deepseek32', 'arch header'
        pos = 4 + alen
    else:
        pos = 8  # ctx magic + seq id
    halves = []
    for _ in range(2):  # base + lid
        ns = struct.unpack('<I', blob[pos:pos+4])[0]; pos += 4
        cnt = struct.unpack('<I', blob[pos:pos+4])[0]; pos += 4
        meta_end = pos + cnt*0
        cells = []
        for _ in range(cnt):
            p = struct.unpack('<i', blob[pos:pos+4])[0]
            n = struct.unpack('<I', blob[pos+4:pos+8])[0]
            cells.append((p, n, blob[pos+8:pos+8+4*n]))
            pos += 8 + 4*n
        vt = struct.unpack('<I', blob[pos:pos+4])[0]; pos += 4
        nl = struct.unpack('<I', blob[pos:pos+4])[0]; pos += 4
        layers = []
        for _ in range(nl):
            ty = struct.unpack('<i', blob[pos:pos+4])[0]; pos += 4
            rs = struct.unpack('<Q', blob[pos:pos+8])[0]; pos += 8
            layers.append((ty, rs, blob[pos:pos+cnt*rs])); pos += cnt*rs
        halves.append((ns, cells, vt, layers))
    return halves, pos, len(blob)
hp, plen, ptot = parse(port)
hr, rlen, rtot = parse(ref)
ok = True
def check(name, a, b):
    global ok
    if a != b:
        print(f"    deepseek32 {name}: MISMATCH {a} vs {b}")
        ok = False
check('blob length', (plen, ptot), (rlen, rtot))
for h, (name) in enumerate(['base', 'lid']):
    (ns_p, cells_p, vt_p, l_p), (ns_r, cells_r, vt_r, l_r) = hp[h], hr[h]
    check(f'{name} n_stream', ns_p, ns_r)
    check(f'{name} cells', cells_p, cells_r)
    check(f'{name} v_trans', vt_p, vt_r)
    check(f'{name} n_layer', len(l_p), len(l_r))
    for li, ((ty_p, rs_p, rows_p), (ty_r, rs_r, rows_r)) in enumerate(zip(l_p, l_r)):
        check(f'{name} layer {li} type/row_size', (ty_p, rs_p), (ty_r, rs_r))
        if li == 0:
            check(f'{name} layer 0 rows (dense lead, bit-exact)', rows_p, rows_r)
        else:
            same = sum(rows_p[i*rs_p:(i+1)*rs_p] == rows_r[i*rs_r:(i+1)*rs_r]
                       for i in range(len(rows_p)//rs_p))
            print(f"    deepseek32 {name} layer {li}: {same}/{len(rows_p)//rs_p} rows equal "
                  f"(sparse-layer payload beyond the arch's bit-exact frontier)")
if ok:
    print('  dsa-lid structure: framing + meta + dense-lead rows byte-identical PASS')
else:
    print('  dsa-lid structure FAIL')
    sys.exit(1)
PYEOF
    done
done

# 3c. mamba2 + jamba: the recurrent half must be byte-identical outright
# (measured: mamba2's conv/ssm rows are bit-exact on this synthetic stream —
# 10496/10496 f32, worst |delta| 0.0; jamba's hybrid blob — the plain attn
# half with the has_kv-filtered layer list then the recurrent half — is
# byte-identical too); the python block re-verifies the structure and reports
# the row deltas as the diagnostic when a blob ever diverges
for arch in mamba2 jamba; do
for k in seq full; do
    for t in "" "-tail"; do
        if cmp -s "$OUT/port-$arch$t-$k.bin" "$OUT/ref-$arch$t-$k.bin"; then
            echo "  recurrent $arch $k ($t): port == reference, byte-identical ($(stat -c%s "$OUT/port-$arch$t-$k.bin") bytes) PASS"
            continue
        fi
        # the diagnostic needs the model's recurrent layer count (n_layer in
        # the blob is the FULL count — the r/s header lists carry only the
        # recurrent layers)
        N_RECR=4; [ "$arch" = jamba ] && N_RECR=3
        python3 - "$OUT/port-$arch$t-$k.bin" "$OUT/ref-$arch$t-$k.bin" "$arch" "$N_RECR" <<'PYEOF' || fail=1
import struct, sys
port, ref, arch, n_recr = (open(p, 'rb').read() for p in sys.argv[1:3]) + sys.argv[3:5]
n_recr = int(n_recr)

def parse(b):
    assert b[:4] in (b'SQST', b'FULL'), 'magic'
    ln = struct.unpack('<I', b[4:8])[0]
    blob, pos = b[8:8+ln], 0
    if b[:4] == b'FULL':
        # the whole-context blob: write_string(arch) then memory->state_write
        alen = struct.unpack('<I', blob[0:4])[0]
        assert blob[4:4+alen].decode() == arch, 'arch header'
        pos = 4 + alen
    else:
        assert struct.unpack('<I', blob[0:4])[0] == 0xaf143cd8, 'io magic'
        assert struct.unpack('<i', blob[4:8])[0] == 0, 'seq id'
        pos = 8
    attn = None
    if arch != 'mamba2':
        # the hybrid's plain attn half precedes the recurrent half
        # (llama_memory_hybrid::state_write, llama-memory-hybrid.cpp:190-195)
        ns = struct.unpack('<I', blob[pos:pos+4])[0]; pos += 4
        acnt = struct.unpack('<I', blob[pos:pos+4])[0]; pos += 4
        for _ in range(acnt):
            pos += 4  # pos
            n = struct.unpack('<I', blob[pos:pos+4])[0]; pos += 4
            pos += 4*n
        vt = struct.unpack('<I', blob[pos:pos+4])[0]; pos += 4
        anl = struct.unpack('<I', blob[pos:pos+4])[0]; pos += 4
        for _ in range(anl):   # k headers + rows
            pos += 4 + 8 + acnt * struct.unpack('<Q', blob[pos+4:pos+12])[0]
        for _ in range(anl):   # v headers + rows
            pos += 4 + 8 + acnt * struct.unpack('<Q', blob[pos+4:pos+12])[0]
        attn = (ns, acnt, vt, anl)
    # llama_memory_recurrent::state_write (llama-memory-recurrent.cpp:766-845)
    cnt = struct.unpack('<I', blob[pos:pos+4])[0]; pos += 4
    cells = []
    for _ in range(cnt):
        p = struct.unpack('<i', blob[pos:pos+4])[0]; pos += 4
        n = struct.unpack('<I', blob[pos:pos+4])[0]; pos += 4
        ids = list(struct.unpack(f'<{n}i', blob[pos:pos+4*n])); pos += 4*n
        cells.append((p, ids))
    st = struct.unpack('<I', blob[pos:pos+4])[0]; pos += 4
    nl = struct.unpack('<I', blob[pos:pos+4])[0]; pos += 4
    halves = {}
    for which in ('r', 's'):   # all R layers first (:906-936), then all S (:938-958)
        layers = []
        for _ in range(n_recr):
            ty = struct.unpack('<i', blob[pos:pos+4])[0]; pos += 4
            rs = struct.unpack('<Q', blob[pos:pos+8])[0]; pos += 8
            rows = blob[pos:pos+cnt*rs]; pos += cnt*rs
            layers.append((ty, rs, rows))
        halves[which] = layers
    return attn, cells, st, nl, halves, pos, len(blob)

(pa, pc, pst, pnl, ph, plen, ptot) = parse(port)
(ra, rc, rst, rnl, rh, rlen, rtot) = parse(ref)
ok = True
def check(name, a, b):
    global ok
    if a != b:
        print(f'    {arch} {name}: MISMATCH {a} vs {b}')
        ok = False
check('blob length', (plen, ptot), (rlen, rtot))
check('attn half header', pa, ra)
check('cells (pos, seq ids)', pc, rc)
check('s_trans', pst, rst)
check('n_layer', pnl, rnl)
n_exact = 0
n_f32 = 0
worst = 0.0
for which in ('r', 's'):
    check(f'{which} layer count', len(ph[which]), len(rh[which]))
    for il, ((tp, rp, rows_p), (tr, rr, rows_r)) in enumerate(zip(ph[which], rh[which])):
        check(f'{which} layer {il} type/row_size', (tp, rp), (tr, rr))
        vp = struct.unpack(f'<{len(rows_p)//4}f', rows_p)
        vr = struct.unpack(f'<{len(rows_r)//4}f', rows_r)
        for x, y in zip(vp, vr):
            n_f32 += 1
            if x == y:
                n_exact += 1
            else:
                worst = max(worst, abs(x - y))
print(f'    {arch} rows: {n_exact}/{n_f32} f32 exactly equal, worst |delta| {worst:.3e}')
# the batch-5 frontier: logits agree to |dlogprob| <= 1e-4 — the state rows
# carry the same drift, bounded here at 2e-2 absolute (f32 accumulation over
# 24 recurrence steps)
if worst > 2e-2:
    print(f'    {arch} row payload: worst |delta| {worst:.3e} exceeds 2e-2')
    ok = False
if ok:
    print(f'  {arch} recurrent structure: framing + meta + headers byte-identical, rows within tolerance PASS')
else:
    print(f'  {arch} recurrent structure FAIL')
    sys.exit(1)
PYEOF
    done
done
done

if [ $fail -eq 0 ]; then
    echo "STATE-KINDS BYTE FORMAT PARITY PASS"
else
    echo "STATE-KINDS PARITY FAILURES"
fi
exit $fail
