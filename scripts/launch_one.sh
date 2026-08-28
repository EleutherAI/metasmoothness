#!/bin/bash
# launch_one.sh <config> <gpus> <master-port> <name> [bergson-path]
#
# Detached launcher with a launch REGISTRY, so a run that never started cannot
# pass unnoticed. Three separate launches have silently no-op'd in this project:
# an abort filtered out by grepping for the success line, a kill loop that parsed
# an empty pid, and a config write that landed before its own mutations. In each
# case the command reported success and nothing checked afterwards.
#
# Every launch appends a row to the registry. scripts/check_launches.sh reads it
# ten minutes later and verifies the run is really on a GPU. See CLAUDE.md.
set -u
CFG="$1"; GPUS="$2"; PORT="$3"; NAME="$4"
LOGS=/mnt/ssd-2/lucia/paper_runs/_logs
REG=$LOGS/launch_registry.tsv
# The newer pods (wisteria/jasmine/violet/clover) run as root with no lucia home,
# so /home/lucia/envs is absent -- but the same pinned env is on the shared volume.
# Pick whichever exists rather than hardcoding one.
if [ -x /home/lucia/envs/paper/bin/python ]; then
  PY=/home/lucia/envs/paper/bin/python
else
  PY=/mnt/ssd-2/lucia/envs/paper/bin/python
fi
# Those pods write as root. Without this, files land 644/755 and the uid 1000/1001
# pods cannot write into the run dirs for merges and recovery.
umask 0000
mkdir -p "$LOGS"

fail() { echo "  LAUNCH-FAILED $NAME on $(hostname) gpu $GPUS :: $*"; exit 1; }

[ -f "$CFG" ] || fail "no config $CFG"

# Which bergson checkout: filter/validate steps carry `method`, which only exists
# on Validate in bergson-filter. The training checkout dies on it.
if [ $# -ge 5 ]; then
  BERG="$5"
elif grep -qa '^ *- *validate:' "$CFG" 2>/dev/null; then
  BERG=/mnt/ssd-2/lucia/bergson-filter
else
  BERG=/mnt/ssd-1/lucia/bergson-main-paper-429
fi

# Refuse a duplicate. Two processes sharing a run_path is not a race we survive:
# bergson clears run_path at startup, so the second wipes the first's state and then
# dies with FileExistsError -- and both write to the same log, so the failure looks
# like it belongs to the healthy run. This has happened twice: a whole 256k sweep
# ran doubled for eight minutes, and a muon shard was launched onto a second pair
# while already training. Check the CONFIG path, not the name, since the same
# config can be launched under different labels.
if pgrep -af "m bergson" 2>/dev/null | grep -qF -- "$CFG"; then
  fail "already running with this config -- refusing to double-launch"
fi

# CLAIM the GPUs before checking them. nvidia-smi is not an arbiter: a job takes
# up to a minute to show memory, so two launchers polling in that window both read
# the pair as free and both launch onto it. That has now happened twice -- two
# 2000-step MAGIC jobs onto bellflower 6,7 sixteen seconds apart, and a MAGIC job
# and a filter shard onto yarrow 6,7 thirteen seconds apart. Both times the second
# job died and the first looked fine, so nothing surfaced it.
#
# mkdir is atomic on the shared volume, so exactly one launcher can create a given
# claim. Claims older than CLAIM_TTL are stale (the launcher died before
# releasing) and are reclaimed. The claim is released by the run itself finishing;
# until then the GPU shows memory and the normal check covers it.
CLAIMS=$LOGS/gpu_claims
CLAIM_TTL=900
mkdir -p "$CLAIMS"
CLAIM="$CLAIMS/$(hostname)_${GPUS//,/-}"
if ! mkdir "$CLAIM" 2>/dev/null; then
  age=$(( $(date +%s) - $(stat -c %Y "$CLAIM" 2>/dev/null || echo 0) ))
  holder=$(cat "$CLAIM/pid" 2>/dev/null || echo "")
  # A claim whose holder is gone is stale no matter how young. Without this, a job
  # that dies seconds after launching -- FileExistsError is the common one -- locks
  # its pair for the full TTL, and two idle GPUs sit unusable while the queue backs
  # up behind them. Age alone is the fallback for a claim with no readable pid.
  if [ -n "$holder" ] && ! kill -0 "$holder" 2>/dev/null; then
    echo "  reclaiming $GPUS: holder pid $holder is gone (claim ${age}s old)"
  elif [ "$age" -ge "$CLAIM_TTL" ]; then
    echo "  reclaiming a stale claim on $GPUS (${age}s old)"
  else
    fail "gpu $GPUS claimed ${age}s ago by live pid ${holder:-unknown} -- refusing to stack"
  fi
  touch "$CLAIM"
fi
release_claim() { rmdir "$CLAIM" 2>/dev/null; }

for g in ${GPUS//,/ }; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null)
  # 100MiB threshold, not 0: a free GPU often reports a few MiB of driver
  # baseline, and refusing on that blocks legitimate launches.
  if [ "${used:-0}" -gt 100 ]; then
    release_claim
    fail "gpu $g already has ${used}MiB in use"
  fi
done

cd /tmp || { release_claim; fail "cannot cd /tmp"; }
CUDA_VISIBLE_DEVICES="$GPUS" MASTER_PORT="$PORT" PYTHONNOUSERSITE=1 PYTHONPATH="$BERG" \
  setsid nohup "$PY" -s -P -m bergson "$CFG" >> "$LOGS/$NAME.log" 2>&1 < /dev/null &
PID=$!
# Record the holder so a later launcher can tell a live claim from a dead one.
echo "$PID" > "$CLAIM/pid"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(date +%s)" "$(hostname)" "$GPUS" "$NAME" "$PID" "$CFG" >> "$REG"
echo "  launched $NAME on $(hostname) gpu $GPUS pid $PID  (registered)"
