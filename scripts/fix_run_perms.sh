#!/bin/bash
# fix_run_perms.sh <run-dir> [<run-dir> ...]
#
# Make everything a run wrote readable from every pod.
#
# lucia is uid 1001 on iris-0 and secret-ord-0 and uid 1000 everywhere else. The
# shared volume does not care, but file MODE does: safetensors writes
# model.safetensors 0600 no matter what umask the launcher sets, so a model
# trained on one pod is invisible from the other half of the fleet. It does not
# fail as a permission error either -- transformers reports
#
#     FileNotFoundError: No such file or directory: .../model.safetensors
#
# which reads as a missing file and sends you looking for a failed training run.
# This has now cost time three times: the 64k bank (101 files), the 128k bank,
# and the 16k/bs512 base.
#
# Run it after any training that another pod will read. It is idempotent.
set -u
for run in "$@"; do
  [ -d "$run" ] || { echo "  no such dir: $run"; continue; }
  n=$(find "$run" -type f ! -perm -0044 2>/dev/null | wc -l)
  d=$(find "$run" -type d ! -perm -0011 2>/dev/null | wc -l)
  find "$run" -type f ! -perm -0044 -exec chmod a+r  {} \; 2>/dev/null
  find "$run" -type d ! -perm -0011 -exec chmod a+rx {} \; 2>/dev/null
  echo "  $(basename "$run"): fixed $n file(s), $d dir(s)"
done
