#!/usr/bin/env python3
"""clip_node_cmp.py — bit-exact comparison of the reference clip graph node
dump (parity/ref_clip_graph_dump.cpp) against the port's
(MTMD_DEBUG_NODES_BIN, crates/llama/src/clip.rs::debug_dump_nodes_bin).

Record layout (both sides): [u32 name_len][name][u32 ne x4][u64 nbytes][data]

DUP and CONT name the same op in the port (the port maps CONT onto Dup), so
they compare equal. The first differing node is reported with the offending
element indices."""
import struct, sys

def load(path):
    d = open(path, 'rb').read()
    out = []
    off = 0
    while off < len(d):
        (nl,) = struct.unpack_from('<I', d, off); off += 4
        name = d[off:off+nl].decode(); off += nl
        ne = struct.unpack_from('<4I', d, off); off += 16
        (nb,) = struct.unpack_from('<Q', d, off); off += 8
        data = d[off:off+nb]; off += nb
        out.append((name, ne, data))
    return out

def eqname(a, b):
    return a == b or {a, b} <= {'DUP', 'CONT'}

ref = load(sys.argv[1])
port = load(sys.argv[2])
print(f"ref {len(ref)} computed nodes, port {len(port)}")
i = j = 0
matched = 0
while i < len(ref) and j < len(port):
    rn, rne, rd = ref[i]
    pn, pne, pd = port[j]
    if eqname(rn, pn) and rne == pne:
        matched += 1
        if rd != pd:
            print(f"FIRST BIT DIFFERENCE after {matched} matched nodes:")
            print(f"  node: {rn} {rne} ({len(rd)} bytes)")
            # locate the first differing f32
            n = min(len(rd), len(pd)) // 4
            for k in range(n):
                a = struct.unpack_from('<f', rd, k*4)[0]
                b = struct.unpack_from('<f', pd, k*4)[0]
                if a != b:
                    print(f"  first diff at flat index {k} (i0={k % rne[0]}): ref {a!r} ({a.hex() if hasattr(a,'hex') else a}) vs port {b!r}")
                    print(f"    ref  bits {struct.unpack_from('<I', rd, k*4)[0]:08x}")
                    print(f"    port bits {struct.unpack_from('<I', pd, k*4)[0]:08x}")
                    break
            sys.exit(1)
        i += 1; j += 1
    else:
        # try to resync across one stray node
        if i+1 < len(ref) and eqname(ref[i+1][0], pn) and ref[i+1][1] == pne:
            print(f"[resync] ref-only node {rn} {rne}")
            i += 1
        elif j+1 < len(port) and eqname(rn, port[j+1][0]) and pne == port[j+1][1]:
            print(f"[resync] port-only node {pn} {pne}")
            j += 1
        else:
            print(f"ALIGNMENT BREAK after {matched} matched nodes:")
            print(f"  ref : {rn} {rne}")
            print(f"  port: {pn} {pne}")
            sys.exit(2)
if i < len(ref) or j < len(port):
    print(f"LENGTH MISMATCH: ref consumed {i}/{len(ref)}, port {j}/{len(port)}")
    sys.exit(3)
print(f"ALL {matched} computed nodes BIT-EXACT")
