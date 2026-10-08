#!/usr/bin/env python3
"""Token + logprob-delta comparator for parity/arch_batch_parity.sh.

  report <tag>   -> reads /tmp/parity-<tag>-{ref.json,cli.out,cli.err}

The reference side is the pinned server's /completion answer (temperature 0,
logprobs 20, cache_prompt false, FIRST request on a fresh server); the port side
is llama-cli's `gen tokens` line plus its LLAMA_RUST_DEBUG `step N: top5`
lines (id, logit, logprob). The delta band is the worst |dlogprob| over the
entries both sides report (ref top-20 ∩ port top-5), which is the same measure
the earlier arch reports in PARITY.md use.

**Reference quirk (byte tokens)**: the pinned server omits the
`completion_probabilities` entry of a token whose piece is not valid UTF-8 (a
byte token such as `<0xDE>`): `tokens_predicted` counts it, the entry list does
not, and its byte is merged into the *next* entry's `bytes` field
(server-task.cpp:285-290 `validate_utf8` truncates the piece). The port's own
list does contain it, so the ids are aligned as a *subsequence* first: the
skipped port tokens must be exactly the number the reference generated but did
not report, and all such tokens are printed for the record. The logprob delta
is then computed on the aligned pairs (the port's step index shifted by the
number of skipped entries before it).
"""
import json
import re
import sys


def align(ref_ids, cli_ids):
    """Map ref index -> cli index if ref_ids is a subsequence of cli_ids.

    Returns (mapping, skipped_cli_ids) or (None, None).
    """
    mapping = []
    skipped = []
    j = 0
    for t in ref_ids:
        while j < len(cli_ids) and cli_ids[j] != t:
            skipped.append(cli_ids[j])
            j += 1
        if j == len(cli_ids):
            return None, None
        mapping.append(j)
        j += 1
    skipped.extend(cli_ids[j:])
    return mapping, skipped


def main(tag):
    ref = json.load(open(f"/tmp/parity-{tag}-ref.json"))
    probs = ref.get("completion_probabilities", [])
    ref_ids = [p["id"] for p in probs]
    ref_steps = [{e["id"]: e["logprob"] for e in p.get("top_logprobs", [])} for p in probs]
    predicted = ref.get("tokens_predicted", len(ref_ids))

    out = open(f"/tmp/parity-{tag}-cli.out").read()
    err = open(f"/tmp/parity-{tag}-cli.err").read()
    m = re.search(r"gen tokens: \[(.*)\]", out)
    cli_ids = [int(x) for x in m.group(1).replace(",", " ").split()] if m else []

    # the port's debug dump rides on stdout (llama-cli main.rs `dump_top3`)
    mine = {}
    for line in (out + "\n" + err).splitlines():
        mm = re.match(r"step\s+(\d+): top5 \[(.*?)\] greedy=", line)
        if not mm:
            continue
        d = {}
        for it in re.findall(r"\((-?\d+), ([-\d.eE+]+), ([-\d.eE+]+)\)", mm.group(2)):
            d[int(it[0])] = float(it[2])
        mine[int(mm.group(1))] = d

    mapping, skipped = align(ref_ids, cli_ids)
    if mapping is None:
        print(
            f"{tag:>14}: tokens 0/{len(ref_ids)}  ref ids are not a subsequence of the port's "
            f"({len(cli_ids)} tokens)  DIVERGED"
        )
        print(f"{'':>14}  ref  ids: {ref_ids}")
        print(f"{'':>14}  cli  ids: {cli_ids}")
        print(f"{'':>14}  ref  text: {ref.get('content', '')[:70]!r}")
        return 1

    # first *aligned* divergence (by construction there is none when the mapping
    # covers every ref id — the check exists for the report's first_diff field)
    first = None
    for i, j in enumerate(mapping):
        if ref_ids[i] != cli_ids[j]:
            first = i
            break

    worst = 0.0
    worst_at = None
    n = 0
    for i, rmap in enumerate(ref_steps):
        j = mapping[i]
        if j not in mine:
            continue
        for tid, lp in rmap.items():
            if tid in mine[j]:
                d = abs(lp - mine[j][tid])
                n += 1
                if d > worst:
                    worst, worst_at = d, (i, tid)

    # the reference reports `tokens_predicted` tokens but omits the byte-token
    # entries; the port must have generated exactly tokens_predicted tokens while
    # the reference reported len(mapping) of them
    n_skipped_expected = predicted - len(ref_ids)
    consistent = len(skipped) == n_skipped_expected and len(cli_ids) == predicted
    matched = len(mapping)
    ok = consistent and matched == len(ref_ids) and matched > 0

    print(
        f"{tag:>14}: tokens {matched}/{len(ref_ids)} aligned first_diff={first}  "
        f"worst |dlogprob| {worst:.4f} over {n} common(top20 x top5) entries"
        + (f" at step {worst_at[0]} id {worst_at[1]}" if worst_at else "")
        + ("  " + ("OK" if ok else "DIVERGED"))
    )
    if skipped:
        print(
            f"{'':>14}  ref omitted {len(skipped)} byte-token entries "
            f"(predicted {predicted}, reported {len(ref_ids)}): port ids {skipped}"
        )
    if n_skipped_expected != len(skipped) or len(cli_ids) != predicted:
        print(
            f"{'':>14}  count mismatch: port generated {len(cli_ids)}, ref predicted {predicted}, "
            f"ref reported {len(ref_ids)}"
        )
    print(f"{'':>14}  ref  text: {ref.get('content', '')[:70]!r}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))