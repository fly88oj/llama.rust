#!/usr/bin/env python3
"""Field-by-field comparison of the reference llama-server responses and the
port's (see parity/run_server_parity.sh).

Usage: server_parity_cmp.py <tag> <dir>

Compares, for each endpoint:
  * the JSON key *sets* (structure) and the values of every field except the
    timing block (`timings.*`, which is wall-clock),
  * the generated `content` / token ids,
  * the HTTP status codes of the malformed-JSON and unknown-path probes.
Exit code 0 iff the token sequences and the non-timing fields all match.
"""
import json
import sys

TIMING_KEYS = {"timings", "prompt_progress"}
# log-probabilities are derived from the raw logits: the port's quantised
# kernels differ from the reference's in the last bits (PARITY.md's band), so
# these fields are compared as a *band* (the ids/ranking are compared exactly)
BAND_KEYS = {"logprob", "prob", "top_logprobs", "top_probs"}
BAND_MAX = 0.20  # the gpt-oss acceptance criterion of PARITY.md


def walk_band(a, b, path="", out=None):
    """collect the logprob/prob leaves as (path, ref, port) pairs"""
    if out is None:
        out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) & set(b)):
            if isinstance(a[k], (int, float)) and isinstance(b[k], (int, float)) and k in BAND_KEYS:
                out.append((f"{path}/{k}", a[k], b[k]))
            else:
                walk_band(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        for i, (x, y) in enumerate(zip(a, b)):
            walk_band(x, y, f"{path}[{i}]", out)
    return out


def walk_diff(a, b, path=""):
    """yield (path, ref_value, port_value) for every differing leaf"""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in TIMING_KEYS:
                continue
            if k not in a:
                yield (f"{path}/{k}", "<missing>", b[k])
            elif k not in b:
                yield (f"{path}/{k}", a[k], "<missing>")
            else:
                yield from walk_diff(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield (f"{path}/len", len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            yield from walk_diff(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a != b:
            yield (path, a, b)
    elif a != b:
        yield (path, a, b)


def load(d, name):
    p = f"{d}/{name}"
    try:
        return json.load(open(p))
    except Exception as e:  # noqa: BLE001
        return {"<read-error>": f"{p}: {e}"}


def cmp_json(d, tag, name, label, ignore=(), report_only=False):
    a = load(d, f"{tag}-ref-{name}")
    b = load(d, f"{tag}-rust-{name}")
    for k in ignore:
        for side in (a, b):
            side.pop(k, None) if isinstance(side, dict) else None
    band = walk_band(a, b)
    diffs = [x for x in walk_diff(a, b) if x[0].split("/")[1] not in ignore]
    # drop the banded leaves from the strict set and report their band instead
    diffs = [x for x in diffs if x[0].rsplit("/", 1)[-1] not in BAND_KEYS]
    status = "OK " if not diffs else ("DIFF" if not report_only else "note")
    extra = ""
    if band:
        worst = max(abs(x - y) for _, x, y in band)
        okb = "OK " if worst <= BAND_MAX else "DIFF"
        extra = f" [{okb}] logprob band: {len(band)} values, max |Δ| {worst:.4f} (limit {BAND_MAX})"
        if worst > BAND_MAX:
            for p, x, y in band:
                print(f"        {p}: ref={x!r} port={y!r}")
    print(f"[{status}] {label}: {len(diffs)} differing field(s){extra}")
    for p, x, y in diffs[:40]:
        print(f"        {p}: ref={x!r} port={y!r}")
    return diffs


def main():
    tag, d = sys.argv[1], sys.argv[2]
    ok = True

    # the generated tokens are the acceptance criterion
    a = load(d, f"{tag}-ref-completion.json")
    b = load(d, f"{tag}-rust-completion.json")
    print(f"reference content: {a.get('content')!r}")
    print(f"port      content: {b.get('content')!r}")
    if a.get("content") != b.get("content"):
        print("FAIL: generated text differs")
        ok = False
    else:
        print("token/text match: OK")

    for name, label in [
        ("health", "GET /health"),
        ("props", "GET /props (informational)"),
        ("completion", "POST /completion"),
        ("probs", "POST /completion n_probs=3"),
        ("tokenize", "POST /tokenize"),
        ("notfound", "POST /nope"),
        ("badjson", "POST /completion (malformed JSON)"),
    ]:
        report_only = name in ("props", "health")
        diffs = cmp_json(d, tag, f"{name}.json", label, report_only=report_only)
        if diffs and not report_only:
            ok = False

    # statuses
    for name in ["badjson", "notfound"]:
        ra = open(f"{d}/{tag}-ref-{name}.status").read().strip()
        rb = open(f"{d}/{tag}-rust-{name}.status").read().strip()
        same = "OK " if ra == rb else "DIFF"
        print(f"[{same}] HTTP status {name}: ref={ra} port={rb}")
        if ra != rb:
            ok = False

    # SSE framing
    def sse_frames(path):
        out = []
        for line in open(path):
            line = line.rstrip("\n")
            if line.startswith("data: "):
                try:
                    out.append(json.loads(line[len("data: "):]))
                except Exception:  # noqa: BLE001
                    out.append({"<parse-error>": line})
        return out

    fa = sse_frames(f"{d}/{tag}-ref-stream.txt")
    fb = sse_frames(f"{d}/{tag}-rust-stream.txt")
    print(f"SSE frames: ref={len(fa)} port={len(fb)}")
    ref_text = "".join(f.get("content", "") for f in fa)
    port_text = "".join(f.get("content", "") for f in fb)
    if ref_text != port_text:
        print(f"FAIL: streamed text differs\n  ref: {ref_text!r}\n port: {port_text!r}")
        ok = False
    else:
        print(f"SSE text match: OK ({ref_text!r})")
    for i, (x, y) in enumerate(zip(fa, fb)):
        for p, u, v in walk_diff(x, y, f"[{i}]"):
            if p.rsplit("/", 1)[-1] in BAND_KEYS:
                continue
            print(f"        frame{p}: ref={u!r} port={v!r}")
            ok = False
    band = [z for i, (x, y) in enumerate(zip(fa, fb)) for z in walk_band(x, y, f"[{i}]")]
    if band:
        worst = max(abs(x - y) for _, x, y in band)
        print(f"        SSE logprob band: {len(band)} values, max |Δ| {worst:.4f}")
        if worst > BAND_MAX:
            ok = False

    print("RESULT:", "MATCH" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()