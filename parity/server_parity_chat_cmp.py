#!/usr/bin/env python3
"""Compare the chat/OAI/embeddings endpoint captures of run_server_parity_chat.sh.

Same comparison protocol as server_parity_cmp.py: field-by-field on the parsed
JSON, ignoring the values that differ by design (created timestamps, random
ids, timings, build string, slot assignment). SSE captures are compared frame
by frame. Embedding floats are compared with a numeric tolerance (the port's
encoder has the documented per-thread numeric tail) and the max deviation is
reported.
"""
import difflib
import json
import sys

# values that differ per run / by design
SKIP_EXACT = {"created", "id", "timings", "system_fingerprint", "id_slot"}
# /models: created differs per second; aliases echo the path (identical here)
# /slots: the released-slot history (`task ? task : task_prev`) is per-request
# noise — which slot a request landed on and its params differ between runs,
# so the per-task keys of IDLE slots are skipped (the live fields
# id/n_ctx/speculative/is_processing and the shape — the per-task keys exist —
# are still compared)
SKIP_SLOTS = SKIP_EXACT | {
    "id_task", "n_prompt_tokens", "n_prompt_tokens_processed",
    "n_prompt_tokens_cache", "params", "next_token",
}


def norm(v, skip):
    if isinstance(v, dict):
        return {k: norm(x, skip) for k, x in v.items() if k not in skip}
    if isinstance(v, list):
        return [norm(x, skip) for x in v]
    return v


def float_stats(a, b, path, stats):
    """collect max|Δ| between two float arrays (both must be numbers)"""
    if isinstance(a, list) and isinstance(b, list):
        for x, y in zip(a, b):
            float_stats(x, y, path, stats)
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        stats[0] = max(stats[0], abs(a - b))


def walk(a, b, path, exact_ok, problems, stats):
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        problems.append(f"{path}: type {type(a).__name__} vs {type(b).__name__}")
        return
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                problems.append(f"{path}.{k}: only in rust")
            elif k not in b:
                problems.append(f"{path}.{k}: only in ref")
            else:
                walk(a[k], b[k], f"{path}.{k}", exact_ok, problems, stats)
    elif isinstance(a, list):
        if len(a) != len(b):
            problems.append(f"{path}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", exact_ok, problems, stats)
    elif isinstance(a, bool) or isinstance(a, str) or a is None:
        if a != b:
            problems.append(f"{path}: {a!r} vs {b!r}")
    else:  # numbers
        if a == b:
            return
        float_stats(a, b, path, stats)
        # logprobs carry the port's documented numeric tail (ids and order
        # must still match) — same bound as PARITY.md's logprob comparisons
        if "logprob" in path or path.endswith(".prob"):
            if abs(a - b) > 0.30:
                problems.append(f"{path}: {a!r} vs {b!r} (beyond logprob tail)")
        elif not exact_ok:
            problems.append(f"{path}: {a!r} vs {b!r}")


def cmp_json(tag, out, name, skip=SKIP_EXACT, float_tol=None, expect_diff=False):
    ref_p, rust_p = f"{out}/{tag}-ref-{name}.json", f"{out}/{tag}-rust-{name}.json"
    try:
        a = json.load(open(ref_p))
        b = json.load(open(rust_p))
    except FileNotFoundError as e:
        print(f"[{name}] MISSING: {e}")
        return False
    problems, stats = [], [0.0]
    walk(norm(a, skip), norm(b, skip), "", float_tol is not None, problems, stats)
    ok = True
    if problems:
        if float_tol is not None and stats[0] <= float_tol:
            print(f"[{name}] OK (floats within tol; max|Δ| = {stats[0]:.3e})")
        elif expect_diff:
            print(f"[{name}] EXPECTED-DIFF (documented as unported): "
                  f"{len(problems)} field(s); first: {problems[0]}")
        else:
            print(f"[{name}] {len(problems)} DIFFERING FIELD(S); max float |Δ| = {stats[0]:.3e}")
            for p in problems[:12]:
                print(f"    {p}")
            ok = False
    else:
        extra = f"; max float |Δ| = {stats[0]:.3e}" if stats[0] > 0 else ""
        print(f"[{name}] 0 differing field(s){extra}")
    return ok


def cmp_status(tag, out, name, expect_diff=False):
    a = open(f"{out}/{tag}-ref-{name}.status").read().strip()
    b = open(f"{out}/{tag}-rust-{name}.status").read().strip()
    ok = a == b
    if ok:
        print(f"[{name}.status] {a} == {b}")
    elif expect_diff:
        print(f"[{name}.status] {a} vs {b} -> EXPECTED-DIFF (documented as unported)")
    else:
        print(f"[{name}.status] {a} vs {b} -> DIFF")
    return ok or expect_diff


def cmp_stream(tag, out, name):
    """SSE frames: parse each `data:` line, compare with the same skips; the
    frame count and order must match. [DONE] terminal included."""
    def frames(p):
        try:
            txt = open(p).read()
        except FileNotFoundError:
            return None
        out = []
        for line in txt.splitlines():
            if line.startswith("data: "):
                payload = line[6:]
                out.append(payload if payload == "[DONE]" else json.loads(payload))
        return out

    a = frames(f"{out}/{tag}-ref-{name}.stream")
    b = frames(f"{out}/{tag}-rust-{name}.stream")
    if a is None or b is None:
        print(f"[{name}.stream] MISSING")
        return False
    if len(a) != len(b):
        print(f"[{name}.stream] frame count {len(a)} (ref) vs {len(b)} (rust)")
        return False
    problems = []
    for i, (x, y) in enumerate(zip(a, b)):
        nx, ny = norm(x, SKIP_EXACT), norm(y, SKIP_EXACT)
        if json.dumps(nx, sort_keys=True) != json.dumps(ny, sort_keys=True):
            problems.append(i)
    if problems:
        print(f"[{name}.stream] differing frames: {problems[:10]}")
        i = problems[0]
        print("  ref :", json.dumps(norm(a[i], SKIP_EXACT))[:400])
        print("  rust:", json.dumps(norm(b[i], SKIP_EXACT))[:400])
        return False
    print(f"[{name}.stream] {len(a)} frames, all identical")
    return True


def main():
    tag, out = sys.argv[1], sys.argv[2]
    results = []
    checks = [
        # the tools surface: lazy-grammar tool call (get_weather, auto),
        # eager grammar (two tools + required), and a plain-text refusal —
        # both finish_reasons ("tool_calls" and "stop") are covered. The
        # tool-call ids are random per run (SKIP_EXACT drops every "id").
        ("chat-tools", "status+json"),
        ("chat-tools2", "status+json"),
        ("chat-tools-refusal", "status+json"),
        ("chat-tools", "stream"),
        ("chat", "json"), ("chat-alias", "json"), ("chat-schema", "json"),
        ("chat-jsonobj", "json"), ("chat-stop", "json"), ("chat-logprobs", "json"),
        ("oaicmpl", "json"),
        ("chat", "stream"), ("chat-usage", "stream"), ("chat-schema", "stream"),
        ("oaicmpl", "stream"),
        ("models", "json"), ("slots", "slots"), ("rerank", "status+json"),
        ("embd501", "status+json"),
    ]
    for check in checks:
        name, kind = check[0], check[1]
        expect_diff = len(check) > 2 and check[2]
        if "stream" in kind:
            results.append(cmp_stream(tag, out, name))
        elif kind == "status+json":
            results.append(cmp_status(tag, out, name, expect_diff=expect_diff))
            results.append(cmp_json(tag, out, name, expect_diff=expect_diff))
        elif kind == "slots":
            results.append(cmp_json(tag, out, name, skip=SKIP_EXACT | SKIP_SLOTS))
        else:
            results.append(cmp_json(tag, out, name))

    # embeddings on bge-m3: floats compared with tolerance (captures are
    # prefixed "$TAG-bge-")
    for name in ["embd-oai", "embd-oai1", "embd-legacy", "embd-legacy2"]:
        results.append(cmp_json(f"{tag}-bge", out, name, float_tol=2e-3))
    results.append(cmp_status(f"{tag}-bge", out, "embd-bad"))

    n_ok = sum(1 for r in results if r)
    print(f"\n{len(results)} checks: {n_ok} OK, {len(results) - n_ok} DIFFERING")
    print("RESULT:", "MATCH" if n_ok == len(results) else "DIFF")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
