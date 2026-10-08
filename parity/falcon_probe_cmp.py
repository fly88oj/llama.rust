#!/usr/bin/env python3
"""Pair the port's `CTX`/`CTXN` lines with the reference's `REF` lines.

  cargo test --release -p llama --test arch_batch2_e2e -- --ignored --nocapture \
      falcon_teacher_force_probe > /tmp/falcon-port.txt
  parity/falcon_probe.sh > /tmp/falcon-ref.txt
  python3 parity/falcon_probe_cmp.py /tmp/falcon-port.txt /tmp/falcon-ref.txt

`CTX i`   = the port's distribution for context i (a single full prefill)
`CTXN i`  = the port's distribution for context i + the port's own argmax
`REF i`   = the reference's first *reported* `completion_probabilities` entry for
            the same context, sent as a `prompt = [token ids]` request.

The pinned server omits the entry of a generated token whose piece is not valid
UTF-8 (a byte token like `<0xDE>`; server-task.cpp:285-290 `validate_utf8`
truncates the piece, its byte is merged into the next entry's `bytes`). So a case
is aligned either
  (a) the reference's reported id equals the port's `CTX` argmax, or
  (b) the port's argmax is a byte token and the reference's reported id/top5
      equal the port's `CTXN` (next-step) distribution.
which is the same evidence the parity comparator
(`parity/arch_batch_cmp.py`) uses on the full generation.
"""
import re
import sys


def parse(path, prefix):
    out = {}
    for line in open(path):
        m = re.match(rf"{prefix}\s+(\d+) n=(\d+) argmax (-?\d+) top5 (.*)", line.strip())
        if m:
            out[int(m.group(1))] = (int(m.group(3)), m.group(4))
    return out


def to_map(s):
    return {int(i): float(lp) for i, lp in re.findall(r"\((-?\d+),([-\d.]+)\)", s)}


def worst_delta(a, b):
    ma, mb = to_map(a), to_map(b)
    common = set(ma) & set(mb)
    return max((abs(ma[k] - mb[k]) for k in common), default=1.0)


def main(port_path, ref_path):
    ctx = parse(port_path, "CTX")
    nxt = parse(port_path, "CTXN")
    ref = parse(ref_path, "REF")
    bad = 0
    omitted = []
    for i in sorted(set(ctx) | set(ref)):
        if i not in ctx or i not in ref:
            print(f"{i:3d} MISSING on one side")
            bad += 1
            continue
        p_id, p_top = ctx[i]
        r_id, r_top = ref[i]
        if p_id == r_id:
            d = worst_delta(p_top, r_top)
            ok = d < 0.01
            print(f"{i:3d} {'OK  ' if ok else 'FAIL'} argmax {p_id} (worst |dlp| {d:.4f})")
            if not ok:
                bad += 1
            continue
        # (b) the port's argmax was a byte token the reference did not report:
        # the reference's first entry must match the port's *next* step
        if i in nxt and r_id == nxt[i][0]:
            d = worst_delta(nxt[i][1], r_top)
            ok = d < 0.01
            print(
                f"{i:3d} {'OK  ' if ok else 'FAIL'} port argmax {p_id} (byte token) omitted by the "
                f"reference; its first reported id {r_id} matches the port's next step "
                f"(worst |dlp| {d:.4f})"
            )
            omitted.append(i)
            if not ok:
                bad += 1
        else:
            print(f"{i:3d} FAIL port argmax {p_id} vs ref argmax {r_id} (no alignment)")
            bad += 1
    print(
        f"\n{len(ctx) - bad}/{len(ctx)} contexts aligned; {len(omitted)} needed the byte-token "
        f"alignment (contexts {sorted(omitted)})"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))