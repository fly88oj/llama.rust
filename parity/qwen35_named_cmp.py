#!/usr/bin/env python3
"""qwen35_named_cmp.py — NAMED-pairing bisection for the qwen35 trunk decode
dump pair (successor of decode_dump_cmp.py's position matcher for this case).

decode_dump_cmp.py aligns ref/port node streams with a SequenceMatcher over
(opclass, type, shape). For the qwen35 trunk the graphs are too skewed for it
(ref 1560 vs port 1338 nodes: the ref keeps forward_select dead-branch head
nodes + K/V CONT chains the port never builds, the port expands the fused
SWIGLU into SILU+MUL and materializes with RESHAPE where the ref uses CONT), so
the position matcher leaves exactly the interesting blk.1 nodes unpaired
(Qcur_full-1 / Q-norm / Kcur / Vcur / attn_pregate / gate_reshaped /
ffn_swiglu-* / the KV-cache writes).

This comparator pairs by NAME SEMANTICS. Structural facts (verified against
src/models/qwen35.cpp @ aa94f2086 vs graph_arch.rs build_qwen35_attn_layer /
build_qwen35_gdn_layer + the value streams of both dumps):

  * both sides run identical layer topology per step (blk.0 GDN -> blk.1 gated
    attn -> head); the stream splits into steps at the trailing MUL_MAT
    result_output (32000, T) — exactly one per step on each side;
  * inside a step, nodes pair by (canonical class, shape) + occurrence order,
    EXCEPT the documented swaps:
      - MUL_MAT (64, T): ref [..., Vcur, Kcur, ...] vs port [..., Kcur, Vcur,
        ...] (the port's topo emits K before V; values confirm the swap);
      - COPY (128, T): ref CONT attn_pregate-1 / CONT gate_reshaped-1 (plus a
        zeros-reading dead-branch DUP) pair with the port's materializing
        RESHAPE 'kqv_out-1' / the anon (128, T) RESHAPE after it;
      - the f16 KV-cache writes: ref SET_ROWS cache_k_l1 (64,512) +
        cache_v_l1 (1,32768) pair with the port's two SET_ROWS (64,512)
        [k, v] by occurrence;
  * canonical classes fold the deliberate op differences:
      RMS_NORM -> NORM, SWIGLU -> GLU (pairs the port's MUL (n_ff, T) — the
      silu*up half of SILU+MUL), SIGMOID/SOFTPLUS -> SILU (the port renders
      all three as SILU);
  * ref embd-head plumbing (GET_ROWS node_1 / node_107 / node_108 crops, the
    dead-branch DUP node_2, zero-size state SCALE views) has no port
    counterpart by design and is excluded — end-to-end correctness is covered
    by the compared result_output/logits node.

usage: python3 parity/qwen35_named_cmp.py <ref.bin> <port.bin> [--max-report N]
"""
import struct
import sys

sys.path.insert(0, ".")
from decode_dump_cmp import parse  # noqa: E402  (same-dir import)


def canon(nd):
    op = nd["op"]
    if op == "RMS_NORM":
        return "NORM"
    if op == "SWIGLU":
        return "GLU"
    if op in ("SIGMOID", "SOFTPLUS"):
        return "SILU"
    return op


# op families that never compute (aliases of their src)
ALIAS_OPS = {"VIEW", "PERMUTE", "TRANSPOSE", "RESHAPE"}
# ref head plumbing without a port counterpart (documented structural diff)
SKIP_REF_NAMES = {"node_1", "node_2", "node_107", "node_108", "model.input_embed"}

# occurrence swaps: class-key -> {ref_idx: port_idx} (0-based within the step's
# occurrence stream of that class)
SWAPS = {
    "MUL_MAT_64": {3: 4, 4: 3},  # ref [.., V, K, ..] vs port [.., K, V, ..]
}


def step_split(nodes):
    """split at each trailing result_output MUL_MAT (vocab, T) node"""
    steps = []
    cur = []
    for nd in nodes:
        cur.append(nd)
        if nd["op"] == "MUL_MAT" and nd["ne"][0] > 1000:
            steps.append(cur)
            cur = []
    assert not cur, f"{len(cur)} trailing nodes after the last result_output"
    return steps


def is_zero(e):
    return e is not None and len(e) > 0 and all(x == 0.0 for x in e)


def cmp_pair(a, b):
    if a["e"] is None or b["e"] is None:
        return None
    if a["n"] != b["n"]:
        return [(k, x, y) for k, (x, y) in enumerate(zip(a["e"], b["e"]))]
    if a["e"] == b["e"]:
        return []
    return [
        (k, x, y)
        for k, (x, y) in enumerate(zip(a["e"], b["e"]))
        if struct.pack("<f", x) != struct.pack("<f", y)
    ]


def collect_port_buckets(ps):
    """(class, ne, ty) -> [nodes] in step order; special families split out"""
    buckets = {}
    for nd in ps:
        op = nd["op"]
        c = canon(nd)
        ne = nd["ne"]
        if op == "MUL_MAT" and ne[0] == 64:
            key = ("MUL_MAT_64", ne, nd["ty"])  # 7-projection family (K/V swap)
        elif op == "MUL" and ne[0] == 64:
            key = ("GLU", ne, nd["ty"])  # silu*up half of the expanded SWIGLU
        elif op == "VIEW" and ne[0] in (192, 256) and ne[1] == 1:
            # the recurrent state row views pair with the ref's GET_ROWS
            # state fetches (the port reads the row by view, the ref by rows)
            key = ("GET_ROWS", ne, nd["ty"])
        elif op == "RESHAPE" and ne[0] == 128 and ne[1] >= 1:
            continue  # handled as COPY128 below
        elif op in ("CPY", "DUP", "CONT"):
            key = ("COPY", ne, nd["ty"])  # state writeback copies
        else:
            if op in ALIAS_OPS:
                continue
            key = (c, ne, nd["ty"])
        buckets.setdefault(key, []).append(nd)
    # COPY128 family: the port's materializing RESHAPEs of (128, T) —
    # linear_attn_out reshape / kqv_out-1 / the gate reshape
    copies = [nd for nd in ps if nd["op"] == "RESHAPE" and nd["ne"][0] == 128]
    if copies:
        buckets[("COPY128", (128, copies[0]["ne"][1], 1, 1), "f32")] = copies
    return buckets


def main():
    ref_path, port_path = sys.argv[1], sys.argv[2]
    max_report = 8
    if "--max-report" in sys.argv:
        max_report = int(sys.argv[sys.argv.index("--max-report") + 1])
    rt, rn = parse(ref_path)
    pt, pn = parse(port_path)
    assert rt == pt, "token streams differ"
    rsteps = step_split(rn)
    psteps = step_split(pn)
    print(f"ref : {len(rn)} nodes in {len(rsteps)} steps (tokens {rt})")
    print(f"port: {len(pn)} nodes in {len(psteps)} steps (tokens {pt})")
    assert len(rsteps) == len(psteps), "step count mismatch"

    n_cmp = 0
    total_div = 0
    reported = 0
    skipped_ref = 0
    fam_counts = {}
    unpaired = {}
    leftovers = {}

    for si, (rs, ps) in enumerate(zip(rsteps, psteps)):
        pb = collect_port_buckets(ps)
        # port MUL_MAT_64 needs the swap applied to ITS occurrence stream
        for key, nodes in list(pb.items()):
            sw = SWAPS.get(key[0])
            if sw:
                nodes = [nodes[sw.get(i, i)] for i in range(len(nodes))]
                pb[key] = nodes
        # SET_ROWS f16 cache writes: ref (64,512)+(1,32768) pair with port's
        # two (64,512) by occurrence — merge the ref keys
        sr_ref = [nd for nd in rs if nd["op"] == "SET_ROWS" and nd["ty"] == "f16"]
        sr_port = [
            nd
            for nd in ps
            if nd["op"] == "SET_ROWS" and nd["ty"] == "f16" and nd["ne"][0] == 64
        ]
        # ref SET_ROWS cache_k_l1 (64,512) pairs with the port's k write (the
        # port builds k_rows before v_rows, so its first f16 (64,512) SET_ROWS
        # is k). ref cache_v_l1 (1,32768) is the ref's kv-major whole-cache
        # view (the port v cache is embed-major (64,512)) — raw streams are
        # not layout-comparable; v-cache parity is proven by the bit-exact
        # kqv reads over the cache.
        sr_ref_cmp = [nd for nd in sr_ref if nd["ne"][0] == 64]
        for a, b in zip(sr_ref_cmp, sr_port):
            d = cmp_pair(a, b)
            if d is None:
                continue
            n_cmp += 1
            fam_counts.setdefault("SET_ROWS_cache_k", [0, 0])
            fam_counts["SET_ROWS_cache_k"][0] += 1
            if d:
                total_div += 1
                fam_counts["SET_ROWS_cache_k"][1] += 1
                if reported < max_report:
                    reported += 1
                    print(
                        f"\n[DIVERGENT] step {si} SET_ROWS cache_k ne={a['ne']}:"
                        f" {len(d)}/{a['n']} elems differ"
                    )
                    for k, x, y in d[:4]:
                        print(f"    elem {k}: ref {x!r} port {y!r}")
        n_vskip = len(sr_ref) - len(sr_ref_cmp)
        if n_vskip:
            fam_counts.setdefault("SET_ROWS_cache_v[layout-skip]", [0, 0])
            fam_counts["SET_ROWS_cache_v[layout-skip]"][0] += n_vskip

        for ri, rnd in enumerate(rs):
            op = rnd["op"]
            base = rnd["name"].split(" (")[0]
            if op in ("VIEW", "PERMUTE", "TRANSPOSE"):
                continue
            if base in SKIP_REF_NAMES:
                continue
            if op == "SET_ROWS" and rnd["ty"] == "f16":
                continue  # handled above
            if rnd["n"] == 0:
                continue  # zero-size state views (ref's padding artifacts)
            if is_zero(rnd["e"]):
                skipped_ref += 1
                # consume the pairing slot if the class exists (fresh-cache
                # reads are zeros on both sides in the prefill step)
                key0 = (canon(rnd), rnd["ne"], rnd["ty"])
                if key0 in pb and pb[key0]:
                    pb[key0].pop(0)
                continue  # dead-branch slot
            c = canon(rnd)
            ne = rnd["ne"]
            if op == "RESHAPE" and ne[0] != 128:
                # ref-only alias reshapes of already-compared sources (Vcur/
                # Kcur/z/alpha/beta/gate/final_output/state_predelta/conv
                # states/cache whole-state views) — the port views them instead
                skipped_ref += 1
                continue
            if (op in ("CONT", "DUP", "CPY") or op == "RESHAPE") and ne[0] == 128:
                # ref CONT attn_pregate/gate_reshaped + RESHAPE linear_attn_out
                # -> port COPY128 family (materializing RESHAPEs, step order)
                key = ("COPY128", (128, ne[1], 1, 1), "f32")
            elif op in ("CONT", "DUP", "CPY"):
                key = ("COPY", ne, rnd["ty"])  # state writeback copies
            elif c == "SWIGLU":
                key = ("GLU", ne, rnd["ty"])  # port's MUL (n_ff, T)
            elif c == "MUL_MAT" and ne[0] == 64:
                key = ("MUL_MAT_64", ne, rnd["ty"])
            elif op == "GET_ROWS" and ne[0] in (192, 256):
                key = ("GET_ROWS", ne, rnd["ty"])  # pairs port state VIEWs
            else:
                key = (c, ne, rnd["ty"])
            bucket = pb.get(key)
            if not bucket:
                uk = f"{op} {ne[:3]} {base!r}"
                unpaired[uk] = unpaired.get(uk, 0) + 1
                continue
            cand = bucket.pop(0)
            d = cmp_pair(rnd, cand)
            if d is None:
                continue
            n_cmp += 1
            fam = f"{c}{tuple(d2 for d2 in ne[:2] if d2)}"
            fam_counts.setdefault(fam, [0, 0])
            fam_counts[fam][0] += 1
            if d:
                total_div += 1
                fam_counts[fam][1] += 1
                if reported < max_report:
                    reported += 1
                    print(
                        f"\n[DIVERGENT] step {si} ref[{ri}] {op} {base!r}"
                        f" ne={ne} vs port {cand['op']} {cand['name']!r}:"
                        f" {len(d)}/{rnd['n']} elems differ"
                    )
                    for k, x, y in d[:4]:
                        print(f"    elem {k}: ref {x!r} port {y!r}")

        # leftovers
        for key, nodes in pb.items():
            for nd in nodes:
                if nd["op"] == "GET_ROWS" and nd["ne"][0] == 128:
                    continue  # the port's embd/out_rows crops (ref crops
                    # pre-residual: different graph point, not comparable)
                if nd["op"] == "SET_ROWS" and nd["ty"] == "f16":
                    continue  # the port's v-cache write (layout-skip family)
                if nd["op"] == "SILU" and nd["ne"][0] == 64:
                    continue  # the port's ffn silu halves (fused ref SWIGLU)
                if nd["op"] == "DUP" and nd["ne"][0] in (32, 64):
                    continue  # the port's cont of q/k views (ref CONT chain)
                if nd["op"] == "DUP" and nd["ty"] == "f16":
                    continue  # the port's f16 v-cache read dup (alias of the
                    # cache; contents proven by the bit-exact kqv reads)
                lk = f"{nd['op']} {tuple(d for d in nd['ne'][:3] if d)}"
                leftovers[lk] = leftovers.get(lk, 0) + 1

    print(f"\ncompared {n_cmp} computing node pairs; divergent: {total_div}")
    print(f"ref dead-branch zero slots skipped: {skipped_ref}")
    if unpaired:
        print("\nref nodes without a port counterpart (aggregate):")
        for k, n in sorted(unpaired.items()):
            print(f"  {n:3d}x {k}")
    if leftovers:
        print("\nport computing nodes left unpaired (aggregate):")
        for k, n in sorted(leftovers.items()):
            print(f"  {n:3d}x {k}")
    print("\nper-family (compared, divergent):")
    for fam, (cnt, div) in sorted(fam_counts.items()):
        flag = "  <== DIVERGES" if div else ""
        print(f"  {fam:32s} {cnt:3d} {div:3d}{flag}")
    return 0 if total_div == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
