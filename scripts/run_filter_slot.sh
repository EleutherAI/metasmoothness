#!/usr/bin/env bash
# Run a queue of tail-filter measurements sequentially on a fixed GPU set.
#
# Usage: run_filter_slot.sh <cuda_devs> <run_id>:<source> [<run_id>:<source>...]
#   e.g. run_filter_slot.sh 6,7 plan_adam_eps1e17_16k_scale0.25:magic
#
# Per job: generate the config (gen_filter.py, nproc = |cuda_devs|), run it, and
# append "<run_id> <source> OK|FAILED" to the slot log. A failure logs and the
# queue moves on, so one bad row does not idle the GPUs for the rest of a night.
#
# The filter estimator retrains once per query, so D17 applies exactly as it does
# to a bank: run each row on the same GPU TYPE its bank was built on, or the
# retrained models are not comparable to the bank's random-removal control.
set -uo pipefail

DEVS=$1; shift
REPO=$(cd "$(dirname "$0")/.." && pwd)
# The filter estimator is PR #430; the pinned -429 worktree predates it and
# rejects the config with "Couldnt instantiate class ... Validate". Use the
# same checkout gen_filter.py targets.
BERGSON=/mnt/ssd-1/lucia/bergson-filter
ENVPY=/home/lucia/envs/paper/bin/python
NPROC=$(awk -F, '{print NF}' <<<"$DEVS")
PORT=$((30100 + ${DEVS%%,*}))
# Keyed by host as well as devices: the log lives on the shared volume, so two
# nodes running the same GPU indices otherwise append to the same file, which
# made three live queues look like one finished one.
LOG=/mnt/ssd-2/lucia/paper_runs/experiments/filter_slot_$(hostname)_${DEVS//,/-}.log

for JOB in "$@"; do
  RID=${JOB%%:*}
  SRC=${JOB##*:}
  echo "=== $RID ($SRC) GPUs $DEVS $(date -u +%H:%M:%S) ==="
  $ENVPY -s -P "$REPO/scripts/gen_filter.py" "$RID" --source "$SRC" --nproc "$NPROC" \
    || { echo "$RID $SRC GENFAIL" | tee -a "$LOG"; continue; }

  R=$(ls -d /mnt/ssd-*/lucia/paper_runs/experiments/"$RID" 2>/dev/null | head -1)

  # bergson refuses to start into an existing run path, so a queue that was
  # killed mid-job leaves an output directory that fails every later attempt
  # with FileExistsError and "queries=0". Clear it, but ONLY when it is
  # incomplete: filter_summary.csv is written at the end, so its presence means
  # finished data that must never be deleted.
  OUT=$R/filter_proponents_$SRC
  if [ -d "$OUT" ] && [ ! -f "$OUT/filter_summary.csv" ]; then
    echo "$RID $SRC clearing incomplete prior output ($(du -sh "$OUT" 2>/dev/null | cut -f1))" | tee -a "$LOG"
    rm -rf "$OUT"
  fi

  CFG=$R/filter_proponents_$SRC.yaml
  [ -f "$CFG" ] || { echo "$RID $SRC NOCONFIG $CFG" | tee -a "$LOG"; continue; }

  (cd /tmp && env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 PYTHONPATH="$BERGSON" \
     MASTER_PORT="$PORT" CUDA_VISIBLE_DEVICES="$DEVS" \
     "$ENVPY" -s -P -m bergson "$CFG") > "$R/filter_proponents_$SRC.log" 2>&1
  RC=$?

  SUM=$R/filter_proponents_$SRC/filter_summary.csv
  N=0; [ -f "$SUM" ] && N=$(( $(wc -l < "$SUM") - 1 ))
  if [ $RC -ne 0 ]; then echo "$RID $SRC FAILED rc=$RC queries=$N" | tee -a "$LOG"
  else echo "$RID $SRC OK queries=$N" | tee -a "$LOG"; fi
done
echo "FILTER_SLOT_DONE $DEVS $(date -u +%H:%M:%S)" | tee -a "$LOG"
