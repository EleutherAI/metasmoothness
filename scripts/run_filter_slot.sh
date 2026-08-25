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

# `lucia` is uid 1001 on iris-0/secret-ord-0/maria-1 and uid 1000 on the other
# seven nodes, and CephFS stores the numeric uid. With the default umask 077,
# anything one group writes is unreadable to the other -- it fails instantly as
# FileNotFoundError while `ls` shows the file, and it had already made 42 of 100
# models in one retrain bank invisible to most of the fleet. Born-readable is the
# only fix that does not need chasing afterwards. See notes/uid_split.md.
umask 022

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

  # Weight decay, gradient clipping and logit scale are cut (2026-08-25): no
  # further results are wanted for them, so they never take a GPU again. The
  # classification lives in scripts/axes.py so the rule is stated once.
  CUTWHY=$($ENVPY -s -P -c "
import sys; sys.path.insert(0, '$REPO/scripts')
from axes import is_cut
print(is_cut(sys.argv[1]) or '')
" "$RID" 2>/dev/null)
  if [ -n "$CUTWHY" ]; then
    echo "$RID $SRC CUT: $CUTWHY" | tee -a "$LOG"
    continue
  fi
  # The step-ladder rows were registered ms-only, so they have a trained base but
  # no leave-k-out bank to borrow a random control from. Those are precisely the
  # rows the filter curve needs past the point where LDS can be computed, so fall
  # back to a fresh control rather than refusing the job.
  RDIR=$(ls -d /mnt/ssd-*/lucia/paper_runs/experiments/"$RID" 2>/dev/null | head -1)
  NOBANK=""
  if [ -n "$RDIR" ] && [ ! -d "$RDIR/retrained/base" ]; then
    NOBANK="--no-bank"
    echo "$RID $SRC no bank -- drawing a fresh random control" | tee -a "$LOG"
  fi
  $ENVPY -s -P "$REPO/scripts/gen_filter.py" "$RID" --source "$SRC" --nproc "$NPROC" $NOBANK \
    || { echo "$RID $SRC GENFAIL" | tee -a "$LOG"; continue; }

  R=$(ls -d /mnt/ssd-*/lucia/paper_runs/experiments/"$RID" 2>/dev/null | head -1)

  # bergson refuses to start into an existing run path, so a queue that was
  # killed mid-job leaves an output directory that fails every later attempt
  # with FileExistsError and "queries=0". Clear it, but ONLY when it is
  # incomplete: filter_summary.csv is written at the end, so its presence means
  # finished data that must never be deleted.
  CFG=$R/filter_proponents_$SRC.yaml
  OUT=$R/filter_proponents_$SRC

  # Claim across the FLEET, not just this node. The output lives on a shared
  # filesystem but `ps` only sees local processes, so the liveness check below
  # cannot see a run on another host -- three processes were once found writing
  # one run_path at once. mkdir is atomic on CephFS, so it settles the race.
  CLAIMS=/mnt/ssd-2/lucia/paper_runs/_claims
  mkdir -p "$CLAIMS"
  CLAIM=$CLAIMS/${RID}__${SRC}
  if ! mkdir "$CLAIM" 2>/dev/null; then
    OWNER=$(cat "$CLAIM/host" 2>/dev/null || echo unknown)
    AGE=$(( ($(date +%s) - $(stat -c %Y "$CLAIM" 2>/dev/null || date +%s)) / 60 ))
    # A claim older than 6h with no summary is a crashed run, not a live one.
    if [ "$AGE" -gt 360 ] && [ ! -f "$OUT/filter_summary.csv" ]; then
      echo "$RID $SRC breaking stale claim from $OWNER (${AGE}m)" | tee -a "$LOG"
      rm -rf "$CLAIM" && mkdir "$CLAIM"
    else
      echo "$RID $SRC SKIP: claimed by $OWNER (${AGE}m ago)" | tee -a "$LOG"
      continue
    fi
  fi
  hostname > "$CLAIM/host"
  trap 'rm -rf "$CLAIM"' EXIT
  if [ -d "$OUT" ] && [ ! -f "$OUT/filter_summary.csv" ]; then
    # Incomplete is not the same as abandoned: these directories are also what a
    # LIVE job on another node is writing into, and every row here takes hours.
    # Only clear one nothing has touched for 30 minutes.
    # Recent mtime alone is not proof of life: a run killed a minute ago leaves
    # one behind and would block its own retry. Require that no process still
    # holds the config before treating it as live.
    HOLDER=$(ps -eo args | grep -F "$CFG" | grep -v grep | head -1)
    if [ -z "$(find "$OUT" -newermt '-30 minutes' -print -quit 2>/dev/null)" ] || [ -z "$HOLDER" ]; then
      echo "$RID $SRC clearing stale output, untouched 30m ($(du -sh "$OUT" 2>/dev/null | cut -f1))" | tee -a "$LOG"
      rm -rf "$OUT"
    else
      echo "$RID $SRC SKIP: incomplete output is being written right now (another node?)" | tee -a "$LOG"
      continue
    fi
  fi

  CFG=$R/filter_proponents_$SRC.yaml
  [ -f "$CFG" ] || { echo "$RID $SRC NOCONFIG $CFG" | tee -a "$LOG"; continue; }

  (cd /tmp && env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 PYTHONPATH="$BERGSON" \
     MASTER_PORT="$PORT" CUDA_VISIBLE_DEVICES="$DEVS" \
     "$ENVPY" -s -P -m bergson "$CFG") > "$R/filter_proponents_$SRC.log" 2>&1
  RC=$?

  SUM=$R/filter_proponents_$SRC/filter_summary.csv
  N=0; [ -f "$SUM" ] && N=$(( $(wc -l < "$SUM") - 1 ))
  rm -rf "$CLAIM"
  if [ $RC -ne 0 ]; then echo "$RID $SRC FAILED rc=$RC queries=$N" | tee -a "$LOG"
  else
    echo "$RID $SRC OK queries=$N" | tee -a "$LOG"
    # Drop the per-query retrained models now the summary exists. Each finished
    # run leaves 14-27 GB here and 13 of them had filled /mnt/ssd-2 to 0 bytes,
    # which stops every job and even blocks git from writing objects. The
    # numbers live in filter_summary.csv / filter_proponents.csv / random_filter.csv,
    # which are siblings of this directory and are kept.
    # NOTE: this is the filter run's OWN checkpoints, never the retrain bank --
    # the bank is $R/retrained and $R/validation*.csv and is not touched.
    if [ -f "$OUT/filter_summary.csv" ] && [ -d "$OUT/checkpoints" ]; then
      echo "$RID $SRC reclaiming $(du -sh "$OUT/checkpoints" 2>/dev/null | cut -f1) of filter checkpoints" | tee -a "$LOG"
      rm -rf "$OUT/checkpoints"
    fi
  fi
done
echo "FILTER_SLOT_DONE $DEVS $(date -u +%H:%M:%S)" | tee -a "$LOG"
