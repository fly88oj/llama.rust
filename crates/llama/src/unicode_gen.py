#!/usr/bin/env python3
"""unicode_gen.py — one-shot converter: src/unicode-data.cpp -> src/unicode_data.rs.

Parses the four tables (unicode_ranges_flags, unicode_set_whitespace,
unicode_map_lowercase, unicode_map_uppercase, unicode_ranges_nfd) from the
pinned llama.cpp unicode-data.cpp and emits Rust static arrays.

Usage:
    python3 unicode_gen.py <path-to-unicode-data.cpp> <output-unicode_data.rs>
"""

import re
import sys


def parse_table(src: str, name: str, npairs: int):
    """Parse `const ... NAME = { {...}, ... };` returning list of int tuples."""
    m = re.search(r"\b" + name + r"\s*=\s*\{", src)
    if not m:
        raise RuntimeError(f"table {name} not found")
    start = m.end()
    end = src.index("};", start)
    body = src[start:end]
    if npairs == 1:
        items = re.findall(r"0x([0-9A-Fa-f]+)", body)
        return [(int(x, 16),) for x in items]
    rows = re.findall(r"\{(0x[0-9A-Fa-f]+)(?:,\s*(0x[0-9A-Fa-f]+)(?:,\s*(0x[0-9A-Fa-f]+))?)?\}", body)
    out = []
    for row in rows:
        vals = tuple(int(x, 16) for x in row if x)
        assert len(vals) == npairs, f"{name}: bad row {row}"
        out.append(vals)
    return out


def fmt_pairs(pairs, per_line=6, fmt="({:#08x}, {:#06x})"):
    lines = []
    for i in range(0, len(pairs), per_line):
        lines.append("    " + ", ".join(fmt.format(*p) for p in pairs[i : i + per_line]) + ",")
    return "\n".join(lines)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    src = open(sys.argv[1], "r", encoding="utf-8").read()

    ranges_flags = parse_table(src, "unicode_ranges_flags", 2)
    set_whitespace = [r[0] for r in parse_table(src, "unicode_set_whitespace", 1)]
    map_lowercase = parse_table(src, "unicode_map_lowercase", 2)
    map_uppercase = parse_table(src, "unicode_map_uppercase", 2)
    ranges_nfd = parse_table(src, "unicode_ranges_nfd", 3)

    # sanity checks mirroring unicode.cpp's asserts
    assert ranges_flags[0][0] == 0
    assert ranges_flags[-1][0] == 0x110000

    out = []
    out.append("//! unicode_data.rs — generated from llama.cpp src/unicode-data.cpp")
    out.append("//! by unicode_gen.py (one-shot conversion). Do not edit by hand.")
    out.append("")
    out.append("/// (start, flags) pairs, sorted by start; last range ends at MAX_CODEPOINTS.")
    out.append(f"pub const UNICODE_RANGES_FLAGS: &[(u32, u16)] = &[\n{fmt_pairs(ranges_flags)}\n];")
    out.append("")
    out.append("/// Codepoints with the `is_whitespace` helper flag (regex: \\s superset source).")
    ws_lines = []
    for i in range(0, len(set_whitespace), 10):
        ws_lines.append("    " + ", ".join(f"{v:#08x}" for v in set_whitespace[i : i + 10]) + ",")
    out.append(f"pub const UNICODE_SET_WHITESPACE: &[u32] = &[\n" + "\n".join(ws_lines) + "\n];")
    out.append("")
    out.append("/// (from, to) simple lowercase mapping, sorted by `from` (binary-searched).")
    out.append(f"pub const UNICODE_MAP_LOWERCASE: &[(u32, u32)] = &[\n{fmt_pairs(map_lowercase)}\n];")
    out.append("")
    out.append("/// (from, to) simple uppercase mapping, sorted by `from`.")
    out.append(f"pub const UNICODE_MAP_UPPERCASE: &[(u32, u32)] = &[\n{fmt_pairs(map_uppercase)}\n];")
    out.append("")
    out.append("/// (first, last, nfd) canonical decomposition base-codepoint ranges.")
    nfd_lines = []
    for i in range(0, len(ranges_nfd), 4):
        nfd_lines.append("    " + ", ".join(f"({a:#08x}, {b:#08x}, {c:#08x})" for a, b, c in ranges_nfd[i : i + 4]) + ",")
    out.append(f"pub const UNICODE_RANGES_NFD: &[(u32, u32, u32)] = &[\n" + "\n".join(nfd_lines) + "\n];")
    out.append("")

    with open(sys.argv[2], "w", encoding="utf-8") as f:
        f.write("\n".join(out))

    print(
        f"ranges_flags={len(ranges_flags)} whitespace={len(set_whitespace)} "
        f"lower={len(map_lowercase)} upper={len(map_uppercase)} nfd={len(ranges_nfd)}"
    )


if __name__ == "__main__":
    main()
