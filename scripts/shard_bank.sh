#!/bin/sh
# shard_bank.sh <run_id> <cuda_devs> <subset_start> <subset_stop> [port]
#
# Adds a shard to an in-progress bank build. A single-pair build runs ~21 min per
# subset, so 100 subsets is ~35 h; six shards bring that under six.
#
# Safe to run against a live build because:
#   * subsets.json is written ONLY by the shard with start == 0, and every other
#     shard reads it, so all shards retrain the SAME subsets. Never launch a
#     shard with start 0 alongside an existing build.
#   * each shard writes validation_<start>_<stop>.csv, so they do not collide.
#     magic_lds.py / ekfac_lds.py merge the slices.
#
# The base build must have reached Validating (subsets.json present) first.
set -e
RID=$1; DEVS=$2; START=$3; STOP=$4; PORT=${5:-$((49000 + START))}

R=""
for b in 1 2; do
  d=/mnt/ssd-$b/lucia/paper_runs/experiments/$RID
  [ -d "$d" ] && R=$d && break
done
[ -n "$R" ] || { echo "no run dir for $RID"; exit 1; }

SRC=$R/bank_build.yaml
OUT=$R/bank_from_filter
[ -f "$SRC" ] || { echo "no bank_build.yaml at $SRC"; exit 1; }
[ -f "$OUT/subsets.json" ] || { echo "REFUSING: $OUT/subsets.json absent -- let the base build reach Validating first"; exit 1; }
[ "$START" -gt 0 ] || { echo "REFUSING start=0: that shard owns subsets.json and is already running"; exit 1; }

busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | awk -F", " -v d="$DEVS" 'BEGIN{split(d,a,",")} {for(k in a) if($1==a[k] && $2+0>500) print $1}')
[ -z "$busy" ] || { echo "SKIP: gpu $busy busy"; exit 1; }

# D17: GPU type is part of run identity. A40 and A100 differ by 6.9e-04 against a
# 1.1e-3 signal, and a mixed-hardware bank measured 0.055 LDS low. Sharding onto
# whatever pair is free silently builds exactly that: it already happened here,
# four shards on A100 for two A40 rows, and 11 subsets had to be discarded.
#
# The first version of this guard inferred the row's hardware from experiments.csv
# and FAILED OPEN when it could not tell -- which it could not, since there is no
# gpu_type column -- so it launched onto A100 again on its first test. Now the
# base build writes bank_from_filter/HARDWARE and this refuses when the file is
# missing. Fail closed: a bank silently built on mixed hardware is far more
# expensive than a shard that declines to start.
# World size is part of run identity too, the same way GPU type is: a shard run
# on one GPU produces subsets at nproc 1 against a bank built at nproc 2, and the
# retrains are then not bit-comparable with the rest. Caught this the hard way --
# passing DEVS=3 launched a single-GPU shard that the hardware guard happily
# allowed, because it was the right GPU type.
NPROC=$(awk -F: '/nproc_per_node/ {gsub(/ /,"",$2); print $2; exit}' "$SRC")
GIVEN=$(awk -F, '{print NF}' <<EOF
$DEVS
EOF
)
if [ -n "$NPROC" ] && [ "$NPROC" != "$GIVEN" ]; then
  echo "REFUSING: base build uses nproc $NPROC and you gave $GIVEN device(s) -- world size is part of run identity"
  exit 1
fi

HWFILE=$OUT/HARDWARE
[ -f "$HWFILE" ] || { echo "REFUSING: $HWFILE absent -- write the base build's GPU type there first (A40 or A100)"; exit 1; }
WANT=$(tr -d ' \n' < "$HWFILE")
HAVE=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | grep -oE "A100|A40")
[ -n "$HAVE" ] || { echo "REFUSING: cannot determine this node's GPU type"; exit 1; }
if [ "$WANT" != "$HAVE" ]; then
  echo "REFUSING: $RID is a $WANT bank and this node has $HAVE -- mixed hardware measures 0.055 LDS low (D17)"
  exit 1
fi

CLAIM=/mnt/ssd-2/lucia/paper_runs/_claims/${RID}__bank${START}_${STOP}
mkdir "$CLAIM" 2>/dev/null || { echo "SKIP: claimed by $(cat "$CLAIM/host" 2>/dev/null)"; exit 1; }
hostname > "$CLAIM/host"

CFG=$R/bank_shard_${START}_${STOP}.yaml
/home/lucia/envs/paper/bin/python -s -P - "$SRC" "$CFG" "$START" "$STOP" "$OUT" <<'PY'
import sys, yaml
src, dst, start, stop, out = sys.argv[1:6]
c = yaml.safe_load(open(src))
v = c["steps"][0]["validate"]
v["subset_start"] = int(start)
v["subset_stop"] = int(stop)
v["subsets"] = out + "/subsets.json"   # reuse, never regenerate
# bergson refuses to start into an existing run_path, and every shard shares the
# base build directory by design. resume is what makes that legal; overwrite
# would destroy the subsets the base shard has already retrained.
v["resume"] = True
v["overwrite"] = False
yaml.safe_dump(c, open(dst, "w"), sort_keys=False)
print(f"  shard {start}-{stop} -> {dst}")
PY

umask 022
cd /tmp
setsid nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 TORCH_NCCL_ENABLE_MONITORING=0 \
  CUDA_VISIBLE_DEVICES="$DEVS" MASTER_PORT="$PORT" PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 \
  PYTHONPATH=/mnt/ssd-1/lucia/bergson-filter \
  /home/lucia/envs/paper/bin/python -s -P -m bergson "$CFG" \
  > "$R/bank_shard_${START}_${STOP}.log" 2>&1 < /dev/null &
echo "LAUNCHED $RID shard $START-$STOP on $DEVS"
