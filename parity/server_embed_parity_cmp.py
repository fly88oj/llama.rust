#!/usr/bin/env python3
"""Embedding comparison of the reference llama-server responses and the port's
(see parity/run_server_embed_parity.sh) — decoder-model `-fe` embeddings.

Usage: server_embed_parity_cmp.py <tag> <dir> <pooling>

Compares, per pooling mode:
  * the legacy `/embedding` rows: shape (n_rows x n_embd) exactly, then every
    element of every row — bit-exact (both sides run the same f32 kernels; the
    report also carries the max abs/rel deviation in case a band is ever
    needed),
  * the OAI `/v1/embeddings` response: HTTP status, structure and embedding
    values the same way.
Exit code 0 iff every element is bit-identical (and the shapes/statuses match).
"""
import json
import sys


def emb_rows(obj):
    """the embedding vector lists of either response shape"""
    if isinstance(obj, dict) and "embedding" in obj:
        e = obj["embedding"]
        return e if e and isinstance(e[0], list) else [e]
    return None


def cmp_vecs(tag, where, a, b, out):
    """elementwise compare of two row lists; returns True iff bit-exact"""
    ok = True
    if len(a) != len(b):
        print(f"MISMATCH {tag} {where}: n_rows {len(a)} vs {len(b)}")
        return False
    for r, (ra, rb) in enumerate(zip(a, b)):
        if len(ra) != len(rb):
            print(f"MISMATCH {tag} {where} row {r}: n_embd {len(ra)} vs {len(rb)}")
            return False
        diffs = [(x, y) for x, y in zip(ra, rb) if x != y]
        if diffs:
            ok = False
            n = len(diffs)
            maxabs = max(abs(x - y) for x, y in diffs)
            denom = max((abs(x) for x, y in diffs), default=1.0) or 1.0
            maxrel = max(abs(x - y) / denom for x, y in diffs)
            print(
                f"BAND {tag} {where} row {r}: {n}/{len(ra)} elements differ, "
                f"maxabs {maxabs:.3e} maxrel {maxrel:.3e}"
            )
        else:
            print(f"OK {tag} {where} row {r}: {len(ra)} elements bit-exact")
    return ok


def main():
    tag, outdir, pool = sys.argv[1], sys.argv[2], sys.argv[3]
    ok = True

    # legacy /embedding
    ref = json.load(open(f"{outdir}/{tag}-ref-pool{pool}-embedding.json"))
    rust = json.load(open(f"{outdir}/{tag}-rust-pool{pool}-embedding.json"))
    ra, rb = emb_rows(ref), emb_rows(rust)
    if ra is None or rb is None:
        print(f"MISMATCH {tag} pooling {pool}: not an embedding response:\n  ref:  {ref}\n  port: {rust}")
        ok = False
    else:
        ok &= cmp_vecs(tag, f"/embedding pool={pool}", ra, rb, outdir)

    # OAI /v1/embeddings
    ref_status = open(f"{outdir}/{tag}-ref-pool{pool}-oai.status").read().strip()
    rust_status = open(f"{outdir}/{tag}-rust-pool{pool}-oai.status").read().strip()
    if ref_status != rust_status:
        print(f"MISMATCH {tag} /v1/embeddings pool={pool}: status {ref_status} vs {rust_status}")
        ok = False
    else:
        print(f"OK {tag} /v1/embeddings pool={pool}: status {ref_status}")
    if ref_status == "200":
        ref = json.load(open(f"{outdir}/{tag}-ref-pool{pool}-oai.json"))
        rust = json.load(open(f"{outdir}/{tag}-rust-pool{pool}-oai.json"))
        ra, rb = emb_rows(ref.get("data", [{}])[0]), emb_rows(rust.get("data", [{}])[0])
        if ra is None or rb is None:
            print(f"MISMATCH {tag} OAI pool={pool}: no embedding in data[0]")
            ok = False
        else:
            ok &= cmp_vecs(tag, f"/v1/embeddings pool={pool}", ra, rb, outdir)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
