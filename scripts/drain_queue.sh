#!/bin/bash
# drain_queue.sh <queue-file>
#
# Launch each job in the queue as GPU pairs free, one at a time.
#
# Queue lines are:  <config-path>|<master-port>|<name>
#
# Two mistakes this exists to not repeat:
#   * A launched job takes ~a minute to show memory in nvidia-smi, so a tight loop
#     reads the SAME pair as free and stacks a second job on it. Two 2000-step
#     MAGIC jobs landed on bellflower 6,7 sixteen seconds apart that way and the
#     second died of OOM. Hence the wait after every successful launch.
#   * Node allowlist per D21: marisa-0 and shivam2-0 are permanently off, and
#     iris-0 GPUs 0-2 belong to another user, so iris is excluded rather than
#     risk pairing into someone else's work.
set -u
QUEUE="$1"
NODES="lucia-ord-0 secret-ord-0 allium-0 shared-ord-0 bellflower-0 lotus-0 wisteria-0 jasmine-0 violet-0 clover-0 yarrow-0 poppy-0 heather-0 orchid-0"
# macOS ships bash 3.2, which has neither mapfile nor associative arrays.
JOBS=()
while IFS= read -r _line || [ -n "$_line" ]; do
  [ -n "$_line" ] && JOBS+=("$_line")
done < "$QUEUE"
done_count=0
for round in $(seq 1 400); do
  [ "$done_count" -ge "${#JOBS[@]}" ] && break
  for idx in "${!JOBS[@]}"; do
    line="${JOBS[$idx]}"
    [ -z "$line" ] && continue
    IFS='|' read -r cfg port name <<< "$line"
    launched=0
    for node in $NODES; do
      pair=$(timeout 40 kubectl exec "$node" -- su - lucia -c \
        'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
         awk -F, "{gsub(/ /,\"\",\$2); if (\$2+0<=100) f=f\" \"\$1} END {n=split(f,a,\" \"); if (n>=2) printf \"%s,%s\", a[1], a[2]}"' 2>/dev/null)
      [ -z "$pair" ] && continue
      out=$(timeout 120 kubectl exec "$node" -- su - lucia -c \
        "bash /tmp/launch_one.sh $cfg $pair $port $name" 2>&1 | tr -d '\r')
      echo "[$(date -u +%H:%M:%S)] $node $pair :: $(echo "$out" | grep -aE 'launched|LAUNCH-FAILED' | head -1)"
      if echo "$out" | grep -qa launched; then
        JOBS[$idx]=""; done_count=$((done_count + 1)); launched=1
        sleep 120
        break
      fi
      # A refusal means this node is not usable for this job right now -- most
      # often the claim lock rejecting a pair that a previous launch already took
      # but whose memory has not appeared in nvidia-smi yet. Move to the NEXT
      # node. Breaking here instead made the drainer retry one node forever and
      # never reach three entirely idle ones.
    done
    [ "$launched" = 1 ] && break
  done
  sleep 45
done
echo "[$(date -u +%H:%M:%S)] drained $done_count/${#JOBS[@]}"
