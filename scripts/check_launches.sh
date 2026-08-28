#!/bin/bash
# Verify every registered launch actually started, ten minutes after the fact.
#
# The failure this exists to catch: a launch that reports success and does nothing.
# It has happened three times here -- an abort message filtered out by grepping for
# the success line, a kill loop that parsed an empty pid, and config mutations
# written after the file. Each time the command looked fine and nothing re-checked.
#
# A launch is GOOD if its process is alive and its GPUs hold memory. It is DONE if
# the process is gone but its log has real content (it ran and finished). It is
# DEAD if the process is gone and the log is empty or missing -- that is the silent
# failure, and it is the only state that needs a human.
#
# Usage: check_launches.sh [min_age_seconds]   (default 600)
set -u
REG=/mnt/ssd-2/lucia/paper_runs/_logs/launch_registry.tsv
LOGS=/mnt/ssd-2/lucia/paper_runs/_logs
MIN_AGE=${1:-600}
NOW=$(date +%s)
ANY_DEAD=0

reg=$(timeout 90 kubectl exec bellflower-0 -- su - lucia -c "cat $REG 2>/dev/null" 2>/dev/null)
[ -z "$reg" ] && { echo "  registry empty or unreadable"; exit 0; }

printf "  %-34s %-14s %-7s %-6s %s\n" NAME NODE GPUS AGE STATE
while IFS=$'\t' read -r iso epoch node gpus name pid cfg; do
  [ -n "${epoch:-}" ] || continue
  age=$(( NOW - epoch ))
  [ "$age" -lt "$MIN_AGE" ] && continue
  out=$(timeout 90 kubectl exec "$node" -- su - lucia -c "
      alive=no; kill -0 $pid 2>/dev/null && alive=yes
      mem=0
      for g in \$(echo $gpus | tr ',' ' '); do
        m=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i \$g 2>/dev/null)
        mem=\$(( mem + \${m:-0} ))
      done
      sz=0; [ -f $LOGS/$name.log ] && sz=\$(wc -c < $LOGS/$name.log)
      echo \"\$alive \$mem \$sz\"" 2>/dev/null)
  set -- ${out:-no 0 0}
  alive=$1; mem=$2; sz=$3
  if [ "$alive" = yes ] && [ "${mem:-0}" -gt 0 ]; then state=GOOD
  elif [ "$alive" = yes ]; then state="STARTING(no gpu yet)"
  elif [ "${sz:-0}" -gt 2000 ]; then state=DONE
  else state=DEAD; ANY_DEAD=1
  fi
  printf "  %-34s %-14s %-7s %-6s %s\n" "$name" "$node" "$gpus" "$((age/60))m" "$state"
done <<< "$reg"

[ "$ANY_DEAD" = 1 ] && echo "  *** DEAD entries above never produced output -- investigate before assuming they ran"
exit 0
