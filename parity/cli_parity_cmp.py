#!/usr/bin/env python3
"""Comparison helper for parity/run_cli_arch_parity.sh.

  body   <mode> <prompt> <schema> <n>   -> the JSON request body on stdout
  report <mode> <n> <tag>               -> MATCH report from the artifacts
"""
import json
import re
import sys


def body(mode, prompt, schema, n):
    b = {
        "prompt": prompt,
        "n_predict": int(n),
        "temperature": 0,
        "cache_prompt": False,
        "logprobs": 20,
    }
    if mode == "schema":
        b["json_schema"] = json.loads(schema)
    print(json.dumps(b))


def report(mode, n, tag):
    n = int(n)
    ref = json.load(open(f"/tmp/parity-{tag}-ref.json"))
    probs = ref.get("completion_probabilities", [])
    ref_ids = [p["id"] for p in probs]
    ref_lps = [[(e["id"], e["logprob"]) for e in p.get("top_logprobs", [])] for p in probs]
    out = open(f"/tmp/parity-{tag}-cli.out").read()
    err = open(f"/tmp/parity-{tag}-cli.err").read()

    print(f"ref  text      : {ref['content']!r}")
    print(f"ref  evaluated : {ref['tokens_evaluated']} predicted={ref['tokens_predicted']}")
    for line in err.splitlines():
        if "grammar: source=" in line or line.startswith("arch ="):
            print(f"cli  {line.strip()}")
    m = re.search(r"gen tokens: \[(.*)\]", out)
    rust = [int(x) for x in m.group(1).replace(",", " ").split()] if m else []

    if mode == "schema":
        # the reference's own output must be valid JSON; compare token ids too
        try:
            json.loads(ref["content"])
            print("ref  json valid: True")
        except Exception as e:  # noqa: BLE001
            print(f"ref  json valid: False ({e})")
        m2 = re.findall(r"gen tokens: \[(.*?)\]", out)
        if m2:
            ids = [int(x) for x in m2[-1].replace(",", " ").split()]
            k = min(len(ref_ids), len(ids))
            matched = sum(1 for a, b in zip(ref_ids, ids) if a == b)
            first = next((i for i, (a, b) in enumerate(zip(ref_ids, ids)) if a != b), None)
            print(f"cli  tokens ({len(ids)}): {ids}")
            print(f"MATCH: {matched}/{k}  first_diff={first}")
        return

    print(f"cli  tokens ({len(rust)}): {rust}")
    k = min(len(ref_ids), len(rust))
    matched = sum(1 for a, b in zip(ref_ids, rust) if a == b)
    first = next((i for i, (a, b) in enumerate(zip(ref_ids, rust)) if a != b), None)
    print(f"MATCH: {matched}/{k}  first_diff={first}")
    # per-step top-5 of the port (LLAMA_RUST_DEBUG lines) for the tie analysis
    mine = {}
    for line in err.splitlines():
        mm = re.match(r"step\s+(\d+): top5 \[(.*?)\] greedy=", line)
        if mm:
            mine[int(mm.group(1))] = mm.group(2)
    if first is not None and first < len(ref_lps):
        print(
            f"ref  step {first} top5 (id:logprob): "
            + ", ".join(f"{i}:{lp:+.3f}" for i, lp in ref_lps[first][:5])
        )
        if first in mine:
            print(f"mine step {first} top5 (id:logit): {mine[first]}")
        ref_margin = (
            ref_lps[first][0][1] - ref_lps[first][1][1] if len(ref_lps[first]) > 1 else float("nan")
        )
        print(
            f"ref  token {ref_ids[first]} margin {ref_margin:.3f}; mine picked {rust[first]} "
            f"(ref {ref_ids[first]} logprob {ref_lps[first][0][1]:+.3f})"
        )
    print("OK" if matched == k else "DIVERGED")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "body":
        body(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        report(sys.argv[2], sys.argv[3], sys.argv[4])