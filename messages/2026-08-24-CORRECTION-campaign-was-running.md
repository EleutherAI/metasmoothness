# 2026-08-24 — CORRECTION: the campaign was running; commit 51b6cb4 says otherwise

The claim commit `51b6cb4` (tune_adamw_64k_bs32 / tune_muon_64k_bs32 for lotus-0)
states "nothing was running on this node" and "GPU utilization for this campaign
was zero". **Both are false.** The claim itself stands; only its justification was
wrong. Retracting here so nobody reads that message and concludes the grid died.

## What was actually true

At the time of that commit, GPUs 3-7 were running our own work — the 32k/64k
experiment banks and the gpt2-medium row. `plan_muon_eps1e17_64k_bs256` was at
step 419/500, 4h13m in.

## The two bad inferences

1. **`/proc/<pid>` missing does not mean "another container".** `nvidia-smi`
   reports **host** PIDs; this shell runs in a PID namespace (`/proc/1` is
   `sleep infinity`). Proof: the two tuning jobs launched from here are local PIDs
   3435803/3435986 and appear in `nvidia-smi` as 50715/52675. Never infer process
   ownership from a `nvidia-smi` PID not existing in `/proc`.

2. **A 0% `utilization.gpu` sample means very little here.** The retrain/MAGIC
   workloads are I/O-bound in phases — muon 64k averages 36 s/it — so point samples
   land on 0% constantly while the run is healthy.

3. Minor: the 36 zombie `bergson` processes are defunct children of a live parent,
   not evidence that anything died.

## How to actually check whether the grid is running

    find /mnt/ssd-2/lucia/paper_runs -maxdepth 3 -name '*.log' -mmin -30 -printf '%TR %p\n' | sort -r

Recent log mtimes plus a tail of the progress bar. That is authoritative; process
tables and instantaneous GPU utilization are not.

## Contention check (the reason this mattered)

Verified the two new tuning slots on GPUs 0 and 2 did not slow the in-flight run:
per-step deltas on `plan_muon_eps1e17_64k_bs256` stayed 20-50s across the launch
and its cumulative average kept falling (36.50 -> 36.20 s/it).
