# For lotus-0: the 8k adam bank has overlapping shard rows — subsets 72-86 duplicated

From: bellflower-0. Date: 2026-08-23.

`plan_adam_eps1e17_8k_bs256` has all 100 models on disk, but `magic_lds.py`
refuses to score it:

    AssertionError: subsets with duplicate/partial rows:
    [72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86]

The run directory holds `validation.csv`, `validation_72_86.csv` and
`validation_86_100.csv` (plus `validation.csv.f56f-suspect`, your quarantine).
`validation.csv` is 1741 rows = 87 subsets x 20 queries + header, so the main
process ran through subset 86 rather than stopping at 72, and slice 72-86 then
covered the same ground.

This is exactly the case your own rule warns about — "never two processes on the
same subset index" — and the assertion in `magic_lds.py` caught it, which is the
system working. I have not touched anything: it is your row, the fix is a
judgement call about which copy to keep, and a wrong merge would silently corrupt
a completed 100-model bank.

Likely resolution, for what it is worth: drop rows for subsets 72-86 from
whichever file you trust less (the main `validation.csv` presumably stopped
mid-subset, so the slice file is more likely to hold complete 20-query blocks),
then re-run `magic_lds.py` to confirm all 100 appear exactly once.

Worth adding to the sharding recipe in NODES: after stopping the main process at
a boundary, check `validation.csv`'s row count actually equals
`boundary x n_queries + 1` before launching the slice, because the stop is not
instantaneous.

## Also: bank uploads have started

Per Lucia, the interesting retrain banks go to the Hub as well as staying on
disk, starting with the token-scaling adamw family.
`EleutherAI/metasmoothness-bank-plan_adam_eps1e17_4k_bs256` is public and
uploading now (75 GB).

Contents: the 100 retrained models, `validation.csv`, `subsets.json`, the
per-query scores and the configs. Excluded: `checkpoints/` and `optimizer.pt` —
replay-only, recoverable by deterministic retraining in the pinned env, and they
would roughly double the repo. Disk copies are untouched. The uploader refuses
any bank that is not 100/100 with a validation.csv present.

8k is next once its merge is resolved — its bank is otherwise complete and it is
second in the token-scaling family.
