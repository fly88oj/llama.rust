#!/usr/bin/env bash
# limited.sh — run a command inside a systemd user scope with hard memory/CPU
# caps (cgroup-level enforcement), so a runaway test or benchmark can never
# take the whole machine down. Born from the 2026-09-28/29 post-mortems:
# uncapped parallel test suites ballooned to 63-68 GiB and froze the box
# twice; see the progress log. The scope kills the offender internally at
# MemoryMax and throttles at MemoryHigh before that; CPUQuota bounds total
# CPU (800% = 8 cores) so builds leave the desktop responsive.
#
# usage: parity/limited.sh [-m MEMMAX] [-h MEMHIGH] [-s SWAPMAX] [-c CPUPCT] -- <cmd...>
#   defaults: 48G / 40G / 8G / 800%
#   gpt-oss-20b dual-side benchmark (ref server + our CLI, ~30G anon): fits.
set -euo pipefail

MEM=48G HIGH=40G SWAP=8G CPU=800%
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m) MEM="$2";  shift 2;;
    -h) HIGH="$2"; shift 2;;
    -s) SWAP="$2"; shift 2;;
    -c) CPU="$2";  shift 2;;
    --) shift; break;;
    *)  break;;
  esac
done
[[ $# -gt 0 ]] || { echo "limited.sh: no command given" >&2; exit 2; }

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

if command -v systemd-run >/dev/null 2>&1 \
   && systemd-run --user --scope --quiet \
        --unit="llamarust-${$//x/x}-$RANDOM" \
        -p MemoryMax="$MEM" -p MemoryHigh="$HIGH" \
        -p MemorySwapMax="$SWAP" -p CPUQuota="$CPU" \
        true 2>/dev/null; then
  exec systemd-run --user --scope --quiet \
    --unit="llamarust-$$-$RANDOM" \
    -p MemoryMax="$MEM" -p MemoryHigh="$HIGH" \
    -p MemorySwapMax="$SWAP" -p CPUQuota="$CPU" \
    "$@"
fi

# fallback when the user manager is unavailable: plain RLIMIT_AS (address
# space only — mmap-heavy tests need the headroom, so add the swap allowance)
echo "limited.sh: systemd-run unavailable, falling back to ulimit" >&2
MEM_KB=$(numfmt --from=iec "$MEM" 2>/dev/null || echo 50331648)
SWP_KB=$(numfmt --from=iec "$SWAP" 2>/dev/null || echo 8388608)
(
  ulimit -v $((MEM_KB + SWP_KB))
  exec "$@"
)
