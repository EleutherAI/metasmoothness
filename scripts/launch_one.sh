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
PY=/home/lucia/envs/paper/bin/python
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

for g in ${GPUS//,/ }; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null)
  [ "${used:-0}" -gt 0 ] && fail "gpu $g already has ${used}MiB in use"
done

cd /tmp || fail "cannot cd /tmp"
CUDA_VISIBLE_DEVICES="$GPUS" MASTER_PORT="$PORT" PYTHONNOUSERSITE=1 PYTHONPATH="$BERG" \
  setsid nohup "$PY" -s -P -m bergson "$CFG" >> "$LOGS/$NAME.log" 2>&1 < /dev/null &
PID=$!
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(date +%s)" "$(hostname)" "$GPUS" "$NAME" "$PID" "$CFG" >> "$REG"
echo "  launched $NAME on $(hostname) gpu $GPUS pid $PID  (registered)"
