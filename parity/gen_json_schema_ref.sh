#!/usr/bin/env bash
# gen_json_schema_ref.sh — regenerate the JSON-schema → GBNF parity artifacts (agent A).
#
#   parity/schemas/*.json        one JSON schema per file (extracted from the reference
#                                test suite + hand written fixtures)
#   parity/schemas/integration_cases.json
#                                the test_schema() cases of test-grammar-integration.cpp
#                                (schema + expected accept/reject strings)
#   parity/json_schema_ref.txt   the ground truth dump: for every schema the GBNF the
#                                reference converter generates (hex encoded), plus the
#                                reference matcher's verdict for the expected strings
#
# Requires the pinned reference build:
#   /home/jeffrey/llm/llama.cpp-pinned              (source, bd4f514db1)
#   /home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin  (libllama-common.so, exports
#                                                    json_schema_to_grammar)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN=/home/jeffrey/llm/llama.cpp-pinned
REF=/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin
SCHEMAS="$ROOT/parity/schemas"
DUMP="$ROOT/parity/ref_json_schema_dump"
OUT="$ROOT/parity/json_schema_ref.txt"

mkdir -p "$SCHEMAS"

# ---------------------------------------------------------------------------
# 1. extract every schema of the reference test suite into parity/schemas/
#    (test-json-schema-to-grammar.cpp: 82 conversion cases; test-grammar-integration.cpp:
#     33 test_schema() cases with their expected accept/reject strings)
# ---------------------------------------------------------------------------
python3 - "$PIN" "$SCHEMAS" <<'PY'
import json, os, re, sys

pin, schemas_dir = sys.argv[1], sys.argv[2]

def skip_ws_comments(s, i):
    while i < len(s):
        if s[i] in " \t\r\n":
            i += 1
        elif s.startswith("//", i):
            j = s.find("\n", i)
            i = len(s) if j < 0 else j + 1
        elif s.startswith("/*", i):
            j = s.find("*/", i)
            i = len(s) if j < 0 else j + 2
        else:
            break
    return i

ESCAPES = {'n': '\n', 't': '\t', 'r': '\r', '0': '\0', 'a': '\a', 'b': '\b',
           'f': '\f', 'v': '\v', '\\': '\\', '"': '"', "'": "'", '?': '?'}

def read_literal(s, i):
    """s[i] starts a C++ string literal; returns (decoded value, index past it)."""
    if s[i] == 'R':
        assert s[i + 1] == '"', s[i:i + 4]
        j = s.index('(', i + 2)
        delim = s[i + 2:j]
        end = s.index(')' + delim + '"', j)
        return s[j + 1:end], end + len(delim) + 2
    assert s[i] == '"', s[i:i + 4]
    out, j = [], i + 1
    while s[j] != '"':
        if s[j] == '\\':
            c = s[j + 1]
            if c in ESCAPES:
                out.append(ESCAPES[c]); j += 2
            elif c == 'x':
                out.append(chr(int(s[j + 2:j + 4], 16))); j += 4
            elif c in '01234567':
                k = j + 1
                while k < len(s) and s[k] in '01234567' and k - j < 4:
                    k += 1
                out.append(chr(int(s[j + 1:k], 8))); j = k
            elif c == 'u':
                out.append(chr(int(s[j + 2:j + 6], 16))); j += 6
            else:
                raise SystemExit("unknown escape \\%s" % c)
        else:
            out.append(s[j]); j += 1
    return ''.join(out), j + 1

def matching_brace(s, i):
    """s[i] == '{'; returns the index of the matching '}', skipping literals/comments."""
    depth, j = 0, i
    while j < len(s):
        if s[j] == '"':
            _, j = read_literal(s, j)
            continue
        if s[j] == 'R' and s[j + 1] == '"':
            _, j = read_literal(s, j)
            continue
        if s.startswith("//", j):
            j = s.find("\n", j) + 1
            continue
        if s.startswith("/*", j):
            j = s.find("*/", j) + 2
            continue
        if s[j] == '{':
            depth += 1
        elif s[j] == '}':
            depth -= 1
            if depth == 0:
                return j
        j += 1
    raise SystemExit("unbalanced braces")

def literals_in(s):
    """every string literal of a region, in order, with comments skipped"""
    out, i = [], 0
    while i < len(s):
        if s[i] == '"' or (s[i] == 'R' and i + 1 < len(s) and s[i + 1] == '"'):
            value, i = read_literal(s, i)
            out.append(value)
        elif s.startswith("//", i):
            i = s.find("\n", i) + 1
        elif s.startswith("/*", i):
            i = s.find("*/", i) + 2
        else:
            i += 1
    return out

def slug(name):
    out = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')
    return out or 'case'

# --- test-json-schema-to-grammar.cpp: test({STATUS, "name", R"""(schema)""", ...}) ---
src = open(os.path.join(pin, "tests", "test-json-schema-to-grammar.cpp")).read()
cases = []
for m in re.finditer(r'test\(\{', src):
    i = skip_ws_comments(src, m.end())
    status = re.match(r'(SUCCESS|FAILURE)', src[i:])
    assert status, src[i:i + 20]
    i = skip_ws_comments(src, i + status.end())
    assert src[i] == ',', src[i:i + 20]
    i = skip_ws_comments(src, i + 1)
    name, i = read_literal(src, i)
    i = skip_ws_comments(src, i)
    assert src[i] == ',', src[i:i + 20]
    i = skip_ws_comments(src, i + 1)
    schema, _ = read_literal(src, i)
    try:
        json.loads(schema)
    except Exception as e:
        raise SystemExit("case %r: schema is not JSON: %s" % (name, e))
    cases.append((name, schema))
assert len(cases) == 81, len(cases)

# the two extra schemas of main(): the recursive "$defs/node" document and the
# "sub-schema $ref" parameters (test-json-schema-to-grammar.cpp:1554-1600)
tail = src[src.index("int main()"):]
extras = []
for m in re.finditer(r'common_json::parse\(\s*R"""\(', tail):
    schema, _ = read_literal(tail, tail.index('R"""', m.start()))
    json.loads(schema)
    extras.append(schema)
assert len(extras) == 2, len(extras)
cases.append(("parsed document (recursion)", extras[0]))
cases.append(("sub-schema $ref", extras[1]))

written = set()
for n, (name, schema) in enumerate(cases, 1):
    fname = "tc-%03d-%s.json" % (n, slug(name))
    with open(os.path.join(schemas_dir, fname), "w") as f:
        f.write(schema.strip() + "\n")
    written.add(fname)
print("extracted %d conversion cases" % len(cases))

# --- parity/schemas/_handwritten.json: the fixtures that the reference tests do not
# --- cover (formats, float bounds, ignored keywords, $defs/recursion, patterns, dotall)
handwritten = json.load(open(os.path.join(schemas_dir, "_handwritten.json")))
for n, entry in enumerate(handwritten, 1):
    prefix = "dotall" if entry.get("dotall") else "x"
    fname = "%s-%03d-%s.json" % (prefix, n, slug(entry["name"]))
    with open(os.path.join(schemas_dir, fname), "w") as f:
        f.write(json.dumps(entry["schema"], indent=1, ensure_ascii=False) + "\n")
print("materialized %d handwritten fixtures" % len(handwritten))

# --- test-grammar-integration.cpp: test_schema(desc, R"""(schema)""", {..}, {..}) --
src = open(os.path.join(pin, "tests", "test-grammar-integration.cpp")).read()
integration = []
for m in re.finditer(r'test_schema\(\s*(?="|R")', src):
    i = skip_ws_comments(src, m.end())
    desc, i = read_literal(src, i)
    i = skip_ws_comments(src, i)
    assert src[i] == ',', src[i:i + 10]
    i = skip_ws_comments(src, i + 1)
    schema, i = read_literal(src, i)
    i = skip_ws_comments(src, i)
    assert src[i] == ',', src[i:i + 10]
    i = skip_ws_comments(src, i + 1)
    blocks = []
    for _ in range(2):
        assert src[i] == '{', src[i:i + 10]
        j = matching_brace(src, i)
        blocks.append(src[i + 1:j])
        i = skip_ws_comments(src, j + 1)
        if i < len(src) and src[i] == ',':
            i = skip_ws_comments(src, i + 1)
    passing, failing = (literals_in(b) for b in blocks)
    try:
        schema_obj = json.loads(schema)
    except Exception as e:
        raise SystemExit("integration case %r: schema is not JSON: %s" % (desc, e))
    integration.append({"name": desc, "schema": schema_obj,
                        "passing": passing, "failing": failing})
assert len(integration) == 33, len(integration)
with open(os.path.join(schemas_dir, "integration_cases.json"), "w") as f:
    f.write(json.dumps(integration, indent=1, ensure_ascii=False) + "\n")
print("extracted %d integration cases" % len(integration))
PY

# ---------------------------------------------------------------------------
# 2. build the dumper
# ---------------------------------------------------------------------------
g++ -O2 -std=c++17 -o "$DUMP" "$ROOT/parity/ref_json_schema_dump.cpp" \
    -I"$PIN/common" -I"$PIN/src" -I"$PIN/include" -I"$PIN/ggml/include" \
    -L"$REF" -lllama-common -lllama -lggml-base -Wl,-rpath,"$REF"

# ---------------------------------------------------------------------------
# 3. dump every schema (sorted glob order, so the file is reproducible)
# ---------------------------------------------------------------------------
export LLAMA_LOG_LEVEL=error

schemas=()
dotalls=()
shopt -s nullglob
for f in "$SCHEMAS"/*.json; do
    b="$(basename "$f")"
    case "$b" in
        integration_cases.json|_*) continue ;;
    esac
    schemas+=("$f")
    case "$b" in
        dotall-*) dotalls+=("$f") ;;
    esac
done

{
    echo "# parity/json_schema_ref.txt — JSON schema → GBNF ground truth (reference bd4f514db1)"
    echo "# generated by parity/gen_json_schema_ref.sh — do not edit"
    echo "# CASE <name> / SCHEMA <hex> / GBNF <hex>|GBNF_FAIL <hex> / ACCEPT|REJECT <0|1> <hex> / END"

    echo "# --- test-json-schema-to-grammar.cpp cases + hand written fixtures ---"
    "$DUMP" "${schemas[@]}"

    echo "# --- common_json::parse() + dump() round trip of every fixture ---"
    "$DUMP" --roundtrip "${schemas[@]}"

    echo "# --- the same fixtures through build_grammar(dotall=true) ---"
    if [ "${#dotalls[@]}" -gt 0 ]; then
        "$DUMP" --dotall "${dotalls[@]}"
    fi

    echo "# --- test-grammar-integration.cpp test_schema() cases + matcher verdicts ---"
    "$DUMP" --cases "$SCHEMAS/integration_cases.json"
} > "$OUT"

echo "== $OUT =="
grep -c '^CASE ' "$OUT"
if grep -q '^REF_MISMATCH' "$OUT"; then
    echo "!! the reference matcher disagrees with the extracted expectations:"
    grep -n '^REF_MISMATCH' "$OUT"
    exit 1
fi
if grep -q '^PARSE_FAIL' "$OUT"; then
    echo "!! a fixture is not valid JSON:"
    grep -n -B2 '^PARSE_FAIL' "$OUT"
    exit 1
fi
wc -l "$OUT"