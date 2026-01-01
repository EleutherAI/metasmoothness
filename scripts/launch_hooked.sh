#!/bin/bash
# launch_hooked.sh <config> <gpus> <port> <name> [bergson_path] -- launch_one.sh plus a post-exit chmod of the run dir,
# for nodes where anon is uid 1001 (iris, secret-ord) so outputs stay readable from uid-1000 nodes.
set -u
cd /data/anon/metasmoothness
out=$(bash scripts/launch_one.sh "$@" 2>&1 | tail -1); echo "$out"
pid=$(echo "$out" | grep -oE "pid [0-9]+" | awk '{print $2}')
rp=$(grep -m1 -E "^\s*run_path:" "$1" | awk '{print $2}')
[ -n "$pid" ] && [ -n "$rp" ] && nohup sh -c "while kill -0 $pid 2>/dev/null; do sleep 60; done; chmod -R a+rX '$rp' 2>/dev/null; chmod -R a+rX '$(dirname $rp)' 2>/dev/null" >/dev/null 2>&1 &
echo "  chmod hook armed for $rp (pid $pid)"
