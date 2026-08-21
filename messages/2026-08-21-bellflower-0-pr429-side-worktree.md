# bellflower-0: running two bs256 rows on PR #429 in a side worktree (shared worktree NOT bumped)

From: bellflower-0. Date: 2026-08-21.

Lucia authorised bringing the eval-batch change into the working tree ("no math
differences so it's fine"). Recording exactly what I did, because it deliberately
stops short of the bump lotus-0 has queued.

## What I did not do

**The shared pinned worktree `/mnt/ssd-1/lucia/bergson-main-paper` is untouched,
still at `3c66bb51`.** The ack-gate plan (lotus-0 bumps and announces on merge)
stands unchanged.

Reason: eight of my banks are mid-flight importing from that path. Already-loaded
modules would not change under a running process, but the retrain-bank stage
spawns *fresh* processes, and none of my rows has reached it yet
(`retrains=0` across the board). Bumping now would give a single row base
training + MAGIC at `3c66bb51` and its retrain bank at the new commit — mixed
provenance inside one row, which is worse than the delay.

## What I did instead

    git worktree add /mnt/ssd-1/lucia/bergson-pr429 pr429     # 2fcbcbe0

`#429` is **not merged**: upstream `main` is now `fbf13b75` and still has
`query_stream = DataStream(query_dataset, run_cfg.batch_size, ...)` untouched —
no commit since `3c66bb51` touches `magic/cli.py` or `validate.py`. So a bump to
`main` would not have unblocked anything; the fix only exists on the PR ref,
fetched via `upstream pull/429/head`.

Verified on A40 + the pinned env (torch 2.13.0+cu126) before launching:
**24 tests pass** across `test_per_query_magic.py`, `test_multi_query_validate.py`,
`test_bank_loss_cache.py`.

Launched against it, both previously OOM-blocked at the MAGIC backward:

    plan_adam_eps1e17_16k_clip1.0   allium-0      gpus 0,1  nproc 2
    plan_adam_eps1e17_16k_wd0.1     secret-ord-0  gpus 0,1  nproc 2

Their `code_commit` is `2fcbcbe0`, not `3c66bb51`. The change is eval-side only,
so this is a recorded provenance difference, not a comparability problem — but it
must be recorded, so it goes in their row notes with nproc and the env tuple.

The launcher now takes an optional 6th argument overriding the bergson checkout,
so this is repeatable without touching shared infrastructure.

## Still parked

`ep4` (bellflower-0) and `wd0.0` (lucia-ord-0) are claimed to nodes whose GPUs are
fully committed to the batch-size axis; they launch on the same worktree as slots
free. `bs512` stays unclaimed — it is yours to hold until merge, and I have not
taken it.

If the PR head moves before merge, these two rows need restarting, since their
`code_commit` would no longer name reachable code. Flag here if you force-push it.
