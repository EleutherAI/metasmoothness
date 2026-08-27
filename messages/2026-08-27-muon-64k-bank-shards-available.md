# plan_muon_eps1e17_64k_bs256 bank: 8 shards unclaimed, configs already written

This is the second new (LDS, filter-delta) pair for the scorer-agreement
question, and it is the only work right now that grows n. Filter-delta coverage
is complete - I re-audited straight off experiments.csv columns (magic_lds vs
filter_magic_delta, ekfac_lds vs filter_ekfac_delta) across all 40 rows and
found 0 rows with an LDS and no delta. So no amount of idle GPU produces a new
point by re-using an existing LDS; only new banks and new MAGIC scores do.

## State

    bank_from_filter/subsets.json   EXISTS  <- shards are safe to start
    retrained/                      1 (base only)
    bank_build                      running on shivam2-0 [6,7], serial over 0..99
    bank_shard_20_30                running on shivam2-0 [4,5]
    bank_shard_30_40                CHAINED on lotus-0 [4,5], starts when the
                                    adam bank shard there exits

## Unclaimed, config already generated

    bank_shard_10_20.yaml   bank_shard_40_50.yaml   bank_shard_50_60.yaml
    bank_shard_60_70.yaml   bank_shard_70_80.yaml   bank_shard_80_90.yaml
    bank_shard_90_100.yaml

All were emitted by `scripts/gen_bank.py <run> --shard A B`, which I fixed today.
They carry subsets/resume/overwrite. Do NOT hand-roll a shard config: without
`subsets:` pointing at the shared subsets.json a shard invents its own subsets,
removes different documents than its peers, and produces a corrupt bank that
still reports a plausible LDS.

## Launch recipe

Claim first: mkdir _claims/plan_muon_eps1e17_64k_bs256__bank<A>_<B>, hostname inside.

    cd /tmp && setsid nohup env CUDA_VISIBLE_DEVICES=<2 gpus> MASTER_PORT=<unique> \
      PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      BERGSON_DIST_TIMEOUT_MIN=1440 \
      PYTHONPATH=/mnt/ssd-1/lucia/bergson-filter \
      /home/lucia/envs/paper/bin/python -s -P -m bergson <run>/bank_shard_A_B.yaml \
      > <run>/bank_shard_A_B.log 2>&1 < /dev/null &

Two constraints that are not negotiable:

  - **A100 only.** bank_build ran on shivam2-0. D17 makes GPU type part of run
    identity and a mixed-hardware bank scored 0.055 low. Available A100: shivam2-0,
    lotus-0. marisa-0 is wedged (21h D-state process, immune to kill -9) - do not
    use it. maria-1 is off-limits.
  - **PYTHONPATH must be bergson-filter.** The pinned -429 Validate has no
    `method` field, so `method: lds` is silently dropped there.

## Reading progress

Do not trust the log tail for liveness. Two independent reasons today:
stderr is block-buffered at 8KB, so a healthy run can go 5+ minutes without
writing; and after a relaunch that truncates an existing log, the new process
writes from offset 0 while `tail` still shows the DEAD run output. Check
instead: live children of the launcher, GPU utilisation, and `ls retrained | wc -l`.

Never read an LDS from a merge that prints INCOMPLETE. `scripts/merge_bank.py`
says so explicitly; I twice reported an EK-FAC "collapse" (0.1085, 0.1659) that
was really 0.4146 / 0.4281 once all 100 subsets were present.
