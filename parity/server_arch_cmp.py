#!/usr/bin/env python3
"""Server-side arch sweep comparator: the FULL /completion response of the
reference llama-server vs the port's on the same model file and request body
(parity/run_server_arch_parity.sh's protocol — fresh server, first request,
temperature 0).

Every field must match except:
  * `timings.*` — wall clock
  * the logprob/prob leaves, compared as the PARITY.md band (the port's
    quantised kernels differ in the last bits; ids/ranking compared exactly)
  * `id` / `created` / `system_fingerprint`-style fields when told so

Usage: server_arch_cmp.py <tag> <dir>
Reads <dir>/<tag>-{ref,rust}.json; exit 0 iff 0 differing fields.
"""
import json
import sys

TIMING_KEYS = {"timings", "prompt_progress"}
BAND_KEYS = {"logprob", "prob", "top_logprobs", "top_probs"}
BAND_MAX = 0.20  # the gpt-oss acceptance criterion of PARITY.md


def walk_band(a, b, path="", out=None):
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


def main():
    tag, d = sys.argv[1], sys.argv[2]
    try:
        a = json.load(open(f"{d}/{tag}-ref.json"))
    except Exception as e:  # noqa: BLE001
        a = {"<read-error>": f"{e}"}
    try:
        b = json.load(open(f"{d}/{tag}-rust.json"))
    except Exception as e:  # noqa: BLE001
        b = {"<read-error>": f"{e}"}

    band = walk_band(a, b)
    diffs = [x for x in walk_diff(a, b) if x[0].rsplit("/", 1)[-1] not in BAND_KEYS]

    tokens_a = a.get("tokens") or []
    tokens_b = b.get("tokens") or []
    tok_same = tokens_a == tokens_b
    content_same = a.get("content") == b.get("content")
    n = min(len(tokens_a), len(tokens_b))
    first_div = next((i for i in range(n) if tokens_a[i] != tokens_b[i]), -1)

    print(f"  content: ref={a.get('content')!r}")
    print(f"           port={b.get('content')!r}")
    extra = ""
    if band:
        worst = max(abs(x - y) for _, x, y in band)
        extra = f" [band max |Δ| {worst:.4f} over {len(band)} values]"
        if worst > BAND_MAX:
            diffs.append(("<logprob-band>", worst, BAND_MAX))
    if diffs:
        print(f"  DIFF: {len(diffs)} differing field(s){extra}")
        for p, x, y in diffs[:25]:
            print(f"        {p}: ref={x!r} port={y!r}")
    else:
        print(f"  OK: 0 differing fields{extra}")
    if tok_same and content_same and not diffs:
        print("  RESULT: MATCH")
        sys.exit(0)
    print(f"  RESULT: MISMATCH (tokens {'same' if tok_same else f'differ at {first_div}'}, "
          f"content {'same' if content_same else 'differ'})")
    sys.exit(1)


if __name__ == "__main__":
    main()
