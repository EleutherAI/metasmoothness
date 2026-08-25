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

for pid in $(pgrep -f "python.*bergson" 2>/dev/null); do
  devs=$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1)
  [ -n "$devs" ] && echo "$devs" | tr ',' '\n' >> "$BUSY"
done

sort -u "$BUSY" -o "$BUSY"
nvidia-smi --query-gpu=index --format=csv,noheader | while read i; do
  grep -qx "$i" "$BUSY" || printf "%s," "$i"
done
rm -f "$BUSY"
