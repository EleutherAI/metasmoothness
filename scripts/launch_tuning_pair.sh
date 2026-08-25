#!/bin/sh
# launch_pair.sh <devs> <port> <tuning_run_id>
# Launches one tuning config on a GPU pair.
#
# Refuses three ways, in order of how much damage they prevent:
#   1. the run already FINISHED  -- bergson runs overwrite:true, so a second
#      launch rmtree's a completed model. A cross-uid PermissionError is the only
#      thing that caught this the first time, and that is luck, not a guard.
#   2. the run is already CLAIMED by another node
#   3. either GPU is in use
DEVS=$1; PORT=$2; ID=$3
C=/mnt/ssd-2/lucia/metasmoothness/configs/tuning/${ID}_s42.yaml
RUN=/mnt/ssd-2/lucia/paper_runs/tuning/${ID}_s42
CLAIM=/mnt/ssd-2/lucia/paper_runs/_claims/${ID}__tune

test -f "$C" || { echo "MISSING CONFIG $C"; exit 1; }

if [ -f "$RUN/model/model.safetensors" ]; then
  echo "SKIP $ID: already finished (model saved) -- relaunching would delete it"
  exit 1
fi

busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | awk -F", " -v d="$DEVS" 'BEGIN{split(d,a,",")} {for(k in a) if($1==a[k] && $2+0>500) print $1}')
if [ -n "$busy" ]; then echo "SKIP $ID: gpu $busy busy"; exit 1; fi

mkdir -p /mnt/ssd-2/lucia/paper_runs/_claims 2>/dev/null
if ! mkdir "$CLAIM" 2>/dev/null; then
  echo "SKIP $ID: claimed by $(cat "$CLAIM/host" 2>/dev/null)"; exit 1
fi
hostname > "$CLAIM/host"

umask 022
cd /tmp || exit 1
setsid nohup env CUDA_VISIBLE_DEVICES="$DEVS" MASTER_PORT="$PORT" PYTHONNOUSERSITE=1 \
  HF_HUB_OFFLINE=1 PYTHONPATH=/mnt/ssd-1/lucia/bergson-main-paper-429 \
  /home/lucia/envs/paper/bin/python -s -P -m bergson "$C" \
  > /tmp/${ID}.log 2>&1 < /dev/null &
echo "LAUNCHED $ID on $DEVS"
