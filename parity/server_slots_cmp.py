#!/usr/bin/env python3
"""The /slots parity comparator (see parity/run_server_slots_parity.sh).

Checks:
  1. the port's save→restore continuation equals the uninterrupted run's
     tokens 8..16 (the KV state round-tripped);
  2. the port's save/restore/erase response *shapes* match the reference's
     key-for-key (timings values and the byte counts n_written/n_read are
     wall-clock / format-dependent and only their presence is compared);
  3. the recurrent arch (mamba2): the save succeeds and the restored
     continuation matches both the port's uninterrupted run and the
     reference server's own restored continuation (the recurrence
     round-tripped cross-implementation).

Usage: server_slots_cmp.py <dir>
"""
import json
import sys

d = sys.argv[1]
ok = True


def load(name):
    try:
        return json.load(open(f"{d}/{name}"))
    except Exception as e:  # noqa: BLE001
        return {"<read-error>": f"{e}"}


def keys(o):
    return sorted(o.keys()) if isinstance(o, dict) else []


# 1) the continuation vs the uninterrupted run
unint = load("uninterrupted.json")
first8 = load("first8.json")
cont = load("port-continue.json")
t_full = unint.get("tokens") or []
t_head = first8.get("tokens") or []
t_cont = cont.get("tokens") or []
n_prompt = len(t_head) + 0
# the continuation prompt = the original prompt tokens + the first 8 generated;
# its length is recoverable as (tokens_evaluated of first8) + 8
prompt_len = (first8.get("tokens_evaluated") or 0) + len(t_head)
cache_n = ((cont.get("timings") or {}).get("prompt_n") or 0), cont.get("tokens_cached")
if t_full[:8] == t_head and t_full[8:16] == t_cont:
    print(f"[OK ] round-trip: uninterrupted {t_full[:4]}... == first8 + restored continuation "
          f"({len(t_cont)} tokens identical)")
else:
    ok = False
    print(f"[DIFF] round-trip tokens: uninterrupted={t_full}")
    print(f"       first8            ={t_head}")
    print(f"       restored continue ={t_cont}")
# the continuation must have come from the RESTORED cache (a full prefix hit —
# no re-decode), which only happens if the restore actually placed the cells.
# `[TAG_PROMPT_LOGITS]` (server-context.cpp:3401-3406): with the whole prompt a
# cache hit, the LAST token is re-evaluated for its logits, so cache_n is the
# prompt length minus one (`timings.cache_n` = n_prompt_cached)
cache_n = (cont.get("timings") or {}).get("cache_n")
if cache_n == prompt_len - 1 and prompt_len > 1:
    print(f"[OK ] cache hit: the continuation reused the {prompt_len - 1} restored cells "
          f"(cache_n == prompt length - 1; the last token re-evaluated for logits)")
else:
    ok = False
    print(f"[DIFF] cache hit: timings.cache_n={cache_n} != prompt length - 1 "
          f"({prompt_len - 1}) — the restored cells were not reused")

# 2) the response shapes vs the reference
for act in ("save", "restore", "erase"):
    a = load(f"ref-{act}.json")
    b = load(f"port-{act}.json")
    if keys(a) != keys(b):
        ok = False
        print(f"[DIFF] {act}: key sets differ: ref={keys(a)} port={keys(b)}")
        continue
    # semantic fields compared exactly
    for k in ("id_slot", "filename", "n_saved", "n_restored", "n_erased"):
        if k in a and a.get(k) != b.get(k):
            ok = False
            print(f"[DIFF] {act}.{k}: ref={a.get(k)!r} port={b.get(k)!r}")
    # timings: only save/restore carry one (the erase result has none in the
    # reference either — server-task.cpp:1649-1656); the value is wall clock
    if act in ("save", "restore"):
        for side, o in (("ref", a), ("port", b)):
            t = o.get("timings")
            if not isinstance(t, dict) or not t:
                ok = False
                print(f"[DIFF] {act}: {side} timings missing: {t!r}")
        if keys(a.get("timings")) != keys(b.get("timings")):
            ok = False
            print(f"[DIFF] {act}.timings keys: ref={keys(a.get('timings'))} port={keys(b.get('timings'))}")
    print(f"[OK ] {act}: shape {keys(b)} matches the reference"
          + (f" (n_saved={b.get('n_saved')})" if "n_saved" in b else "")
          + (f" (n_restored={b.get('n_restored')})" if "n_restored" in b else "")
          + (f" (n_erased={b.get('n_erased')})" if "n_erased" in b else ""))

# the restore really restored: the pollute prompt ran in between and its
# generation differs from the original prompt's
pollute = load("pollute.json")
if pollute.get("content") == unint.get("content"):
    ok = False
    print("[DIFF] the pollute prompt produced the same content — the pollution did not happen?")

# 3) the recurrent arch's round-trip: the save must SUCCEED (the conv/ssm
#    cells are serialized — llama-memory-recurrent.cpp:766-1224 on the port)
#    and the restored continuation must match both the port's uninterrupted
#    run AND the reference server's own restored continuation
recur_save = load("recur-port-save.json")
if isinstance(recur_save, dict) and recur_save.get("error"):
    ok = False
    print(f"[DIFF] recurrent arch: save still errors — {recur_save['error'].get('message')!r}")
else:
    n_saved = recur_save.get("n_saved")
    print(f"[OK ] recurrent arch: /slots save succeeded (n_saved={n_saved})")

    def tokens(name):
        return (load(name).get("tokens") or [])

    ru, rh, rc = tokens("recur-port-uninterrupted.json"), tokens("recur-port-first8.json"), \
        tokens("recur-port-continue.json")
    if ru[:8] == rh and ru[8:16] == rc:
        print(f"[OK ] recurrent round-trip: uninterrupted {ru[:4]}... == first8 + restored "
              f"continuation ({len(rc)} tokens identical)")
    else:
        ok = False
        print(f"[DIFF] recurrent round-trip tokens: uninterrupted={ru}")
        print(f"       first8            ={rh}")
        print(f"       restored continue ={rc}")

    # the cross-check: the reference's own restored continuation on the same
    # model (its file format differs, so only the trajectory is comparable)
    rrc = tokens("recur-ref-continue.json")
    if rrc == rc:
        print(f"[OK ] recurrent cross-check: the port's restored continuation == the "
              f"reference's ({len(rrc)} tokens)")
    else:
        ok = False
        print(f"[DIFF] recurrent cross-check: port continue={rc} vs reference continue={rrc}")

# 3b) the save/restore response shapes on the recurrent arch match the
#     reference's (the same key sets as the qwen2.5 legs)
for act in ("save", "restore"):
    a = load(f"recur-ref-{act}.json")
    b = load(f"recur-port-{act}.json")
    if keys(a) != keys(b):
        ok = False
        print(f"[DIFF] recurrent {act}: key sets differ: ref={keys(a)} port={keys(b)}")
    else:
        print(f"[OK ] recurrent {act}: shape {keys(b)} matches the reference")

print("RESULT:", "MATCH" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
