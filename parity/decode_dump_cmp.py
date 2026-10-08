#!/usr/bin/env python3
"""decode_dump_cmp.py — bisect the first divergent graph node between a
reference dump (parity/ref_decode_dump.c) and a port dump
(crates/llama/tests/qwen3_prefill_dump.rs).

Both streams share the DECDMP1 format (see the C probe's header comment). The
two graphs do not have identical node lists (the port leaves tensors unnamed,
builds q/k/v in a different order, expands the fused SWIGLU into SILU+MUL and
has no whole-cache RESHAPE chain), so nodes are aligned with a shape/op-class
SequenceMatcher and only aligned pairs of *computing* ops (same op class, same
type, same shape) are compared bit-for-bit. For each divergent node the
per-dim1-row stats are printed — for the [1024, T] activations a row is one
token's vector, so the output maps 1:1 onto the per-token embedding rows of
parity/embd_rows_probe.sh.

usage: python3 parity/decode_dump_cmp.py <ref.bin> <port.bin> [--max-report N]
"""
import difflib
import struct
import sys

ELEM_CAP = 1 << 19

# ops whose payload is an alias of their src (no compute): never compared
ALIAS_OPS = {"VIEW", "PERMUTE", "RESHAPE", "TRANSPOSE"}


def opclass(op):
    # NORM/RMS_NORM share a class (the port encodes rms_norm as Norm+flag);
    # everything else keeps its name
    if op in ("NORM", "RMS_NORM"):
        return "NORM"
    if op in ("CPY", "CONT", "DUP"):
        return "CPY"
    return op


def parse(path):
    f = open(path, "rb")
    magic = f.read(8)
    assert magic == b"DECDMP1\0", magic
    (n_tok,) = struct.unpack("<I", f.read(4))
    toks = struct.unpack(f"<{n_tok}i", f.read(4 * n_tok))
    (n_nodes,) = struct.unpack("<I", f.read(4))
    nodes = []
    for _ in range(n_nodes):
        (ol,) = struct.unpack("<B", f.read(1))
        op = f.read(ol).decode()
        (nl,) = struct.unpack("<B", f.read(1))
        name = f.read(nl).decode()
        (tl,) = struct.unpack("<B", f.read(1))
        ty = f.read(tl).decode()
        ne = struct.unpack("<4q", f.read(32))
        (n,) = struct.unpack("<Q", f.read(8))
        elems = None
        if n < ELEM_CAP:
            elems = struct.unpack(f"<{n}f", f.read(4 * n))
        nodes.append({"op": op, "name": name, "ty": ty, "ne": ne, "n": n, "e": elems})
    rest = f.read()
    assert rest == b"", f"{len(rest)} trailing bytes"
    return toks, nodes


def key(nd):
    # prefer the graph tensor name (the port's builders mirror the C cb()
    # names); the reference's auto-generated node_NNN names carry no pairing
    # information and fall back to the shape class
    name = nd["name"]
    if name.startswith("node_") and name[5:].isdigit():
        name = ""
    return (name, opclass(nd["op"]), nd["ty"], nd["ne"])


def row_stats(a, b, ne, tok_dim):
    """per token-dim row (bit-identical count, max|d|): a token's block is the
    contiguous product of the dims below tok_dim."""
    row_len = 1
    for d in range(tok_dim):
        row_len *= max(ne[d], 1)
    n_rows = max(ne[tok_dim], 1)
    out = []
    for r in range(n_rows):
        ra = a[r * row_len : (r + 1) * row_len]
        rb = b[r * row_len : (r + 1) * row_len]
        bit = 0
        mx = 0.0
        for x, y in zip(ra, rb):
            if struct.pack("<f", x) == struct.pack("<f", y):
                bit += 1
            else:
                d = abs(x - y)
                if d > mx:
                    mx = d
        out.append((bit, mx, row_len))
    return out


def tok_dim_of(ne, n_tokens):
    """the dim that indexes tokens: the highest dim whose extent == n_tokens"""
    best = None
    for d in range(4):
        if ne[d] == n_tokens:
            best = d
    return best if best is not None else 1


def main():
    ref_path, port_path = sys.argv[1], sys.argv[2]
    max_report = 10
    if "--max-report" in sys.argv:
        max_report = int(sys.argv[sys.argv.index("--max-report") + 1])
    rt, rn = parse(ref_path)
    pt, pn = parse(port_path)
    print(f"ref : {len(rn)} nodes, tokens {rt}")
    print(f"port: {len(pn)} nodes, tokens {pt}")
    assert rt == pt, "token streams differ — dumps are not comparable"
    n_tokens = len(rt)

    sm = difflib.SequenceMatcher(a=[key(x) for x in rn], b=[key(x) for x in pn], autojunk=False)
    n_cmp = 0
    n_div = 0
    reported = 0
    per_name = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            if max(i2 - i1, j2 - j1) <= 4:
                for k in range(max(i2 - i1, j2 - j1)):
                    a = rn[i1 + k] if i1 + k < i2 else None
                    b = pn[j1 + k] if j1 + k < j2 else None
                    print(f"[unmatched] ref[{i1+k}] {fmt(a)}  vs  port[{j1+k}] {fmt(b)}")
            else:
                print(
                    f"[unmatched block] ref[{i1}:{i2}] x{len(rn[i1:i2])} vs port[{j1}:{j2}] x{len(pn[j1:j2])}"
                )
            continue
        for i, j in zip(range(i1, i2), range(j1, j2)):
            a, b = rn[i], pn[j]
            if a["e"] is None or b["e"] is None:
                continue  # over-cap payload (header already matched)
            if opclass(a["op"]) in ALIAS_OPS:
                continue  # views/reshapes alias their src
            n_cmp += 1
            base = a["name"].split(" (")[0] or a["op"]
            st = per_name.setdefault(base, [0, 0])
            st[0] += 1
            if a["e"] != b["e"]:
                n_div += 1
                st[1] += 1
                if reported < max_report:
                    print(
                        f"[DIVERGENT] ref[{i}]/port[{j}]: {a['op']}/{b['op']} "
                        f"{a['name']!r}/{b['name']!r} ne={a['ne']} {a['ty']} n={a['n']}"
                    )
                    td = tok_dim_of(a["ne"], n_tokens)
                    for r, (bit, mx, tot) in enumerate(
                        row_stats(a["e"], b["e"], a["ne"], td)
                    ):
                        print(f"    tok {r}: bit {bit}/{tot}  max|d| {mx:.3e}")
                    for k, (x, y) in enumerate(zip(a["e"], b["e"])):
                        if struct.pack("<f", x) != struct.pack("<f", y):
                            print(f"    first diff elem {k}: ref {x!r} port {y!r}")
                            break
                    reported += 1
    print(f"\ncompared {n_cmp} computing nodes; divergent: {n_div}")
    print("\nper-name summary (occurrences, divergent):")
    for name, (cnt, div) in sorted(per_name.items()):
        flag = "  <== DIVERGES" if div else ""
        print(f"  {name:24s} {cnt:3d} {div:3d}{flag}")


def fmt(nd):
    if nd is None:
        return "-"
    return f"{nd['op']} {nd['name']!r} ne={nd['ne']} {nd['ty']}"


if __name__ == "__main__":
    main()
