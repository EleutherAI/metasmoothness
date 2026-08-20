#!/usr/bin/env bash
# Run a queue of tuning rows sequentially on a fixed GPU set.
#
# Usage: run_tuning_slot.sh <cuda_devs> <run_id> [<run_id>...]
#   e.g. run_tuning_slot.sh 4,5 tune_adamw_4k_lr0.0001 tune_adamw_4k_lr0.0002
#
# Per row: generate the config (gen_tuning_run.py, nproc = |cuda_devs|), train
# under a hard 2h deadline, evaluate heldout CE, delete checkpoints, append
# "<run_id> <heldout>" to the slot log. A failed run logs FAILED and the queue
# moves on. Results still go through the builder (NODES.md) — the log is the
# hand-off, not the record.
set -uo pipefail

DEVS=$1; shift
REPO=$(cd "$(dirname "$0")/.." && pwd)
BERGSON=/mnt/ssd-1/lucia/bergson-damping
RUNS=/mnt/ssd-2/lucia/paper_runs/tuning
NPROC=$(awk -F, '{print NF}' <<<"$DEVS")
LOG=$RUNS/slot_${DEVS//,/-}.log

for RID in "$@"; do
  echo "=== $RID (GPUs $DEVS) $(date -u +%H:%M:%S) ==="
  python "$REPO/scripts/gen_tuning_run.py" "$RID" --nproc "$NPROC" || { echo "$RID GENFAIL" | tee -a "$LOG"; continue; }
  CFG=$RUNS/${RID}_s42/tune.yaml
  echo "CMD: PYTHONPATH=$BERGSON CUDA_VISIBLE_DEVICES=$DEVS bergson $CFG"
  timeout 7200 env PYTHONPATH="$BERGSON" CUDA_VISIBLE_DEVICES="$DEVS" bergson "$CFG"
  RC=$?
  if [ $RC -ne 0 ]; then echo "$RID FAILED rc=$RC" | tee -a "$LOG"; continue; fi
  H=$(CUDA_VISIBLE_DEVICES=${DEVS%%,*} timeout 1800 python "$REPO/scripts/heldout_eval.py" "$RUNS/${RID}_s42/model" | tail -1)
  echo "$RID $H" | tee -a "$LOG"
  rm -rf "$RUNS/${RID}_s42/checkpoints"
done
echo "SLOT_DONE $DEVS $(date -u +%H:%M:%S)" | tee -a "$LOG"
