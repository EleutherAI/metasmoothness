#!/bin/sh
# Print the GPU indices on this node that are genuinely free.
#
# Memory alone is NOT enough. A job in dataset preprocessing holds its GPUs but
# sits at ~3 MiB with no CUDA compute app registered, so a "<500 MiB" check calls
# it free and the next launch lands on top of it. That is how several jobs got
# double-launched onto busy cards.
#
# A GPU counts as busy if EITHER it has memory/a compute process, OR some live
# bergson process names it in CUDA_VISIBLE_DEVICES.
BUSY=/tmp/.gpu_busy.$$
: > "$BUSY"

nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | awk -F", " '$2+0 >= 500 {print $1}' >> "$BUSY"

# Only a process that still HOLDS a CUDA context reserves a GPU. A finished job
# whose process lingers -- zombie, or a parent that never reaped -- still carries
# CUDA_VISIBLE_DEVICES in its environ, and counting that marks the GPU busy
# forever. Six A100s on marisa-0 read as busy for twelve hours that way, while
# nvidia-smi showed 3 MiB and no compute apps on any of them.
#
# So skip processes that are defunct or no longer running, and cross-check
# against the compute-app list: if nvidia-smi reports no process on a GPU and it
# holds under 500 MiB, it is free regardless of who claims it in environ.
LIVE=/tmp/.gpu_live.$$
nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader | sort -u > "$LIVE"

for pid in $(pgrep -f "python.*bergson" 2>/dev/null); do
  state=$(awk '{print $3}' /proc/$pid/stat 2>/dev/null)
  [ "$state" = "Z" ] && continue
  devs=$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1)
  [ -n "$devs" ] || continue
  # only honour the claim if that GPU actually has a compute app on it
  for d in $(echo "$devs" | tr ',' ' '); do
    bus=$(nvidia-smi --query-gpu=index,gpu_bus_id --format=csv,noheader \
          | awk -F", " -v g="$d" '$1==g {print $2}')
    [ -n "$bus" ] && grep -qF "$bus" "$LIVE" && echo "$d" >> "$BUSY"
  done
done
rm -f "$LIVE"

sort -u "$BUSY" -o "$BUSY"
nvidia-smi --query-gpu=index --format=csv,noheader | while read i; do
  grep -qx "$i" "$BUSY" || printf "%s," "$i"
done
rm -f "$BUSY"
