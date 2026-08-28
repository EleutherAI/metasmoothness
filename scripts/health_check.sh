#!/bin/bash
# health_check.sh — find jobs holding GPUs without doing work.
#
# A wedged bergson job is indistinguishable from a healthy one at a glance: the
# process is alive, its GPUs hold memory, and one rank spins at 100% utilisation.
# Two MAGIC jobs sat dead for an hour that way while nvidia-smi showed both their
# GPUs busy.
#
# What does NOT work, and was tried first: flagging jobs whose log has not grown.
# It flagged 25 of 45 jobs, nearly all healthy. Two reasons. Progress bars here are
# coarse -- a filter shard's outer bar ticks once per query (~60 min) and a bank
# shard's once per subset (~54 min) -- so an hour of silence is normal. And stderr
# is block-buffered at 8KB, so even a fast inner bar flushes only every ~12 min.
#
# What DOES work: CPU. A working bergson rank consumes very close to one core, so
# a node should burn (busy GPUs x interval) CPU-seconds per interval. A wedged job
# holds its GPU memory and its 100% utilisation but stops consuming CPU entirely --
# GPU "utilisation" counts a busy-wait spin in a stuck collective as work, and CPU
# does not.
#
# The denominator is our own worker RANKS, not GPUs on the box. Counting GPUs made
# a healthy iris-0 read as 57%, because three of its GPUs belong to another user.
set -u
NODES="lotus-0 lucia-ord-0 secret-ord-0 allium-0 shared-ord-0 bellflower-0 iris-0 wisteria-0 jasmine-0 violet-0 clover-0 yarrow-0 poppy-0 heather-0 orchid-0"
INTERVAL=${1:-20}
for node in $NODES; do
  timeout 90 kubectl exec "$node" -- su - lucia -c '
    h=$(hostname)
    P=$(pgrep -u lucia -f "bin/python -s -P" | tr "\n" ",")
    P=${P%,}
    [ -z "$P" ] && { printf "  %-13s idle\n" "$h"; exit 0; }
    # One worker rank per GPU we hold, so the worker count is the denominator --
    # and unlike GPU indices it needs no environ parsing and cannot accidentally
    # count another user's GPUs on a shared node.
    ranks=$(pgrep -u lucia -cf "multiprocessing.spawn")
    b=$(ps -o times= -p "$P" 2>/dev/null | awk "{s+=\$1} END {print s+0}")
    sleep '"$INTERVAL"'
    a=$(ps -o times= -p "$P" 2>/dev/null | awk "{s+=\$1} END {print s+0}")
    used=$((a-b)); exp=$((ranks * '"$INTERVAL"'))
    [ "$exp" -le 0 ] && exp=1
    pct=$(( used * 100 / exp ))
    printf "  %-13s %2d ranks, CPU %3ds/%ss (expect ~%d) = %d%%%s\n" \
      "$h" "$ranks" "$used" "'"$INTERVAL"'" "$exp" "$pct" "$([ $pct -lt 60 ] && echo "   <== UNDER, look for a wedged rank")"
  ' 2>/dev/null
done
