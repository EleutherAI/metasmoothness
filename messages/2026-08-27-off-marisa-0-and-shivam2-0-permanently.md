# Off marisa-0 and shivam2-0 permanently (Lucia's instruction, 2026-08-27)

Do not launch anything on marisa-0 or shivam2-0 again. Both are retired for our
work. Everything below is resumable by whoever picks it up.

## shivam2-0: four jobs stopped, all resumable

### 1. london16k_bs256_adamw/experiment.yaml  -- 10 HOURS OF PROGRESS, RESUME THIS FIRST

    Validating: 74/100 subsets  [9:47:42 elapsed, ~3:26 remaining]
    retrained/ holds 75 entries (base + 74 subsets)
    config has resume: true, overwrite: false

Relaunching the SAME experiment.yaml continues from subset 74. Do not wipe
bank_from_filter / retrained; do not pass overwrite. Losing this costs ~10h.

    /mnt/ssd-2/lucia/paper_runs/experiments/london16k_bs256_adamw/experiment.yaml

### 2. plan_muon_eps1e17_64k_bs256 bank -- MOVE TO lotus-0

    bank_from_filter/subsets.json  EXISTS
    retrained/ holds 2 (base + subset 0). Barely started, nothing to preserve
    beyond subsets.json, which MUST be kept -- every shard removes the documents
    it names.

    bank_build.yaml        was on shivam2-0 [6,7]
    bank_shard_20_30.yaml  was on shivam2-0 [4,5], had reached "Validating 0/10"

Continue on **lotus-0**, which is also A100. The bank has only ever run on A100
(shivam2-0), so lotus-0 preserves GPU-type identity under D17. Do NOT continue it
on an A40 node: a mixed-hardware bank scored 0.055 low.

Shard configs are already generated and correct (subsets/resume/overwrite):
10_20, 20_30, 30_40, 40_50, 50_60, 60_70, 70_80, 80_90, 90_100.

### 3. plan_adam_eps1e17_64k_bs256/experiment.yaml

Running 1d23h with no experiment.log on either volume. Its bank is separately
at 99/100 and finishing on lotus-0, so this process was redundant. Nothing to resume.

## marisa-0: cannot be cleaned from inside the pod

    bank_build for plan_adam_eps1e17_64k_bs256, 21h+, state **DN**
    (uninterruptible sleep) -- immune to kill -9.

All 8 GPUs sit at a uniform 54432 MiB. I cannot clear this; it needs a node
restart or host access. That bank is NOT at risk - it is at 99/100 and its last
subset is completing on lotus-0.

Every claim held by either node has been released.

## Remaining A100 capacity is now lotus-0 only

marisa-0 and shivam2-0 were two of our four A100 nodes; maria-1 is already
off-limits. Bank shards require A100 under D17, so **all bank work is now
single-node on lotus-0**. This is the binding constraint on growing n for the
LDS <-> filter-delta correlation, and it is worth saying plainly: the muon 64k
bank at one node is on the order of days, not hours.

Filter-delta coverage remains complete (0 of 40 rows have an LDS without a
delta), so there is no cheaper substitute - only new banks and new MAGIC scores
add points.
