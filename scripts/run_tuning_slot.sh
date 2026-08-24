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
# Match the generators (gen_tuning_run.py, gen_experiment_run.py): the pinned
# paper checkout, not the shared bergson-damping working copy whose branch
# state is not controlled.
BERGSON=/mnt/ssd-1/lucia/bergson-main-paper-429
RUNS=/mnt/ssd-2/lucia/paper_runs/tuning
NPROC=$(awk -F, '{print NF}' <<<"$DEVS")
# D15: the pinned venv is the only valid environment, and bergson must not
# run with a checkout as cwd -- Python puts cwd first on sys.path, which
# silently shadows PYTHONPATH with whatever branch that checkout is on.
# The interpreter every other run in the grid uses, and the one the
# generators print (gen_tuning_run.PYTHON). /mnt/ssd-2/lucia/envs/paper is a
# separate venv owned by another user -- same torch, but not the pinned one.
ENVPY=/home/lucia/envs/paper/bin/python
# Distinct rendezvous port per slot -- concurrent slots sharing the default
# 29500 hang in distributed init before CUDA is ever touched.
PORT=$((29500 + ${DEVS%%,*}))
LOG=$RUNS/slot_${DEVS//,/-}.log

for RID in "$@"; do
  echo "=== $RID (GPUs $DEVS) $(date -u +%H:%M:%S) ==="
  $ENVPY -s -P "$REPO/scripts/gen_tuning_run.py" "$RID" --nproc "$NPROC" || { echo "$RID GENFAIL" | tee -a "$LOG"; continue; }
  # Launch from the git-tracked config, not the copy inside the run
  # directory -- that copy is disposable and gets swept.
  CFG=$REPO/configs/tuning/${RID}_s42.yaml
  [ -f "$CFG" ] || { echo "$RID NOCONFIG $CFG" | tee -a "$LOG"; continue; }
  # Deadline from the run's own step count: a 256k row is 16000 steps and a
  # flat 2h killed it at 4600 with no checkpoint to resume from. TIMEOUT=<secs>
  # overrides.
  STEPS=$($ENVPY -s -P -c "
import csv, sys
rid = sys.argv[1]
with open(sys.argv[2]) as f:
    for r in csv.DictReader(f):
        if r['run_id'] == rid:
            print(r['steps'] or 0); break
    else:
        print(0)
" "$RID" "$REPO/tuning.csv" 2>/dev/null || echo 0)
  DEADLINE=${TIMEOUT:-$(( STEPS > 0 ? STEPS * 2 + 1800 : 7200 ))}
  [ "$DEADLINE" -lt 7200 ] && DEADLINE=7200
  echo "CMD: (cd /tmp) PYTHONPATH=$BERGSON CUDA_VISIBLE_DEVICES=$DEVS MASTER_PORT=$PORT $ENVPY -s -P -m bergson $CFG  (deadline ${DEADLINE}s, ${STEPS} steps)"
  (cd /tmp && timeout "$DEADLINE" env PYTHONNOUSERSITE=1 PYTHONPATH="$BERGSON" MASTER_PORT="$PORT" \
     HF_HUB_OFFLINE=1 \
     CUDA_VISIBLE_DEVICES="$DEVS" "$ENVPY" -s -P -m bergson "$CFG")
  RC=$?
  if [ $RC -ne 0 ]; then echo "$RID FAILED rc=$RC" | tee -a "$LOG"; continue; fi
  H=$(CUDA_VISIBLE_DEVICES=${DEVS%%,*} timeout 1800 $ENVPY -s -P "$REPO/scripts/heldout_eval.py" "$RUNS/${RID}_s42/model" | tail -1)
  echo "$RID $H" | tee -a "$LOG"
  rm -rf "$RUNS/${RID}_s42/checkpoints"
done
echo "SLOT_DONE $DEVS $(date -u +%H:%M:%S)" | tee -a "$LOG"
