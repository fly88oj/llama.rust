#!/usr/bin/env bash
# cap_session.sh — hard-cap THIS session's systemd user scope (everything any
# agent/test/benchmark in the session spawns, wrapper or not) so a runaway
# process dies inside the scope instead of freezing the machine. Re-apply
# after every session restart (the transient scope gets a new id):
#
#   parity/cap_session.sh            # 60G hard / 52G throttle (defaults)
#
# Born from the 2026-10-01 00:01-00:05 OOM burst: a python3 parity script
# ballooned to 43.4 GiB while LM Studio held a 26B/256K-ctx mlock server —
# global OOM killed opera/traefik/coredns as collateral. The per-command
# parity/limited.sh caps one invocation; this caps the whole session as the
# backstop. Neither touches the user's own apps (LM Studio etc.).
set -euo pipefail

MAX="${1:-60G}"
HIGH="${2:-52G}"

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
SCOPE="$(sed 's|.*/||' /proc/self/cgroup)"
case "$SCOPE" in
  app-zcode-*.scope|run-*.scope) ;;
  *) echo "cap_session.sh: unexpected cgroup '$SCOPE' — refusing to cap" >&2; exit 2 ;;
esac

systemctl --user set-property "$SCOPE" "MemoryMax=$MAX" "MemoryHigh=$HIGH"
systemctl --user show "$SCOPE" -p MemoryMax -p MemoryHigh
