#!/usr/bin/env bash
# Kill the *stale* reference llama-server instances left behind by parity runs
# (the build-rust-ref binary on port 8793), but never touch the user's own
# LM Studio server (the /models/... one on port 8080) or anything else.
# Run from anywhere: bash parity/kill_stale_servers.sh
pids=$(pgrep -f 'build-rust-ref/bin/llama-serve[r]')
if [ -z "$pids" ]; then
  echo "no stale reference servers running"
  exit 0
fi
echo "$pids" | xargs -r kill
sleep 2
left=$(pgrep -f 'build-rust-ref/bin/llama-serve[r]' || true)
if [ -n "$left" ]; then
  echo "$left" | xargs -r kill -9
fi
echo "killed: $(echo "$pids" | tr '\n' ' ')"
pgrep -af 'llama-serve[r]' || echo "(none running now)"