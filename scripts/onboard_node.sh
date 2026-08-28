#!/bin/bash
# Onboard a new pod: verify it can actually run our jobs, then stage the launcher.
#
# Every check here exists because its absence has cost time on this fleet:
#   GPU type      D17 makes GPU type part of run identity; an A100 pod cannot host
#                 retrains for an A40-trained row
#   uid           lucia is 1001 on iris/secret-ord and 1000 elsewhere. A model
#                 written 0600 by one reads as FileNotFoundError from the other
#   ssd-2 write   D23 forbids ssd-1 writes; everything we produce goes to ssd-2
#   ssd-1 read    both bergson checkouts live there, and filters need the
#                 bergson-filter one specifically
#   env           the pinned interpreter, not whatever python is on PATH
#   datasets      a run that trains on the wrong corpus is invisible in its own logs
#
# Usage: onboard_node.sh <node> [<node> ...]
set -u
SRC=/private/tmp/claude-501/-Users-luciaquirke/d0f8ae2e-b4e5-4f45-858d-75ab36032ee4/scratchpad
PY=/home/lucia/envs/paper/bin/python

for node in "$@"; do
  echo "=== $node ==="
  out=$(timeout 90 kubectl exec "$node" -- su - lucia -c '
    echo "host=$(hostname)"
    echo "uid=$(id -u)"
    echo "gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
    echo "gpumodel=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "free=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "{if (\$1+0==0) c++} END {print c+0}")"
    echo "ssd2_write=$( (touch /mnt/ssd-2/lucia/paper_runs/_logs/.probe_$$ 2>/dev/null && rm -f /mnt/ssd-2/lucia/paper_runs/_logs/.probe_$$ && echo yes) || echo NO)"
    echo "ssd1_read=$([ -d /mnt/ssd-1/lucia/bergson-main-paper-429 ] && echo yes || echo NO)"
    echo "bergson_filter=$([ -d /mnt/ssd-2/lucia/bergson-filter ] && echo yes || echo NO)"
    echo "python=$([ -x /home/lucia/envs/paper/bin/python ] && echo yes || echo NO)"
    echo "datasets=$([ -d /mnt/ssd-2/lucia/datasets_local/train_128k.hf ] && echo yes || echo NO)"
    echo "repo=$([ -d /mnt/ssd-2/lucia/metasmoothness ] && echo yes || echo NO)"
    echo "torch=$(/home/lucia/envs/paper/bin/python -c "import torch;print(torch.__version__)" 2>/dev/null || echo NO)"
  ' 2>&1)
  if [ -z "$out" ] || echo "$out" | grep -qi "error\|not found"; then
    echo "  UNREACHABLE or broken: ${out:0:120}"
    continue
  fi
  echo "$out" | sed 's/^/  /'
  bad=$(echo "$out" | grep -c "=NO")
  model=$(echo "$out" | grep "^gpumodel=" | cut -d= -f2)
  case "$model" in *A40*) ;; *) echo "  WARNING: $model is not A40 -- D17 means it cannot host retrains for A40 rows";; esac
  if [ "$bad" -gt 0 ]; then
    echo "  NOT READY: $bad required capability missing"
    continue
  fi
  for f in launch_one.sh check_launches.sh; do
    timeout 100 kubectl cp "$SRC/$f" "$node:/tmp/$f" >/dev/null 2>&1 && echo "  staged $f"
  done
  echo "  READY"
done
