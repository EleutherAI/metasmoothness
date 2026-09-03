# Resuming runs: the rules

Every long run here WILL be interrupted at least once: the ssd-2 volume is a
25 TiB quota shared with other tenants and hits 0 bytes without warning, pods
restart, and NCCL times out. Whether an interruption costs 10 minutes or 10
hours depends entirely on whether the run's progress is resumable and whether
the resume path is SAFE. Both have failed repeatedly. This file is the record
and the procedure.

## The core trap: resume-by-existence

bergson's resume checks whether an output EXISTS, not whether it is VALID.
Every multi-hour disaster in this project traces to that gap:

- An interrupted inverse application preallocates its full-size output
  (`kfac_query/gradients.bin`, 105 GB of zeros at 1.5B). Resume sees the file,
  skips the stage, and scoring runs zeros to completion with every `written`
  flag true. Three rungs of filter data (69 retrains) measured noise this way.
- The pipeline-level `_step_complete` checks the hessian DIRECTORY, but the
  fit writes `hessian/kfac.part` and renames only `kfac` on completion — so an
  interrupted fit passes as complete and the next stage crashes (or worse).
- Per-query MAGIC resume skips any existing `per_query/q{i}.pt`. A query
  killed mid-backward leaves a near-empty file that resume treats as done:
  18/20 queries at 256k scored ~one batch of documents each, and the merged
  filter delta (0.018) silently broke the method-appendix trend.

**Never trust existence. Gate content.**

## Gates (use them, extend them)

`scripts/gate_ekfac.py` checks content, not existence:

    python -P scripts/gate_ekfac.py inverse <score_run_dir> --purge
    python -P scripts/gate_ekfac.py scores  <score_run_dir>
    python -P scripts/gate_ekfac.py all     <score_run_dir>

- Run `inverse --purge` BEFORE any scoring launch that could resume: an
  invalid (zero/empty) `kfac_query` is moved aside so the stage recomputes.
  Purge is directory-level — file-level purge leaves the dir "complete".
- `scores` requires all written-flags true AND nonzero score content. It runs
  automatically inside `gen_filter.py`; nothing downstream may consume scores
  that have not passed it.
- For MAGIC per-query files, check zeros fraction before merging
  (a healthy q.pt has ~0% exact zeros; the broken ones had 99.9%).

## What resumes, and how

| Work | Resumes? | Granularity / procedure |
|---|---|---|
| EK-FAC pipeline stages | yes | per stage; hessian/query/kfac_query dirs skip when present — gate first |
| Hessian fit | NO | interrupted fit leaves `kfac.part`; delete the run's `hessian/` dir before relaunch or the pipeline skips the unfinished fit |
| Scoring (`scores.bin`) | yes | per-cell written flags; safe to relaunch |
| Filter/validate queries | yes (patched) | reads its own `filter_proponents.csv`, skips measured queries; needs `resume: true` in the config (gen_filter emits it) |
| MAGIC per-query | yes | per query file — but gate zeros before merge; a shard's trajectory `checkpoints/` must be a REAL dir of per-file symlinks with LOCAL copies of `step_1999.ckpt` and `log_history.json` (bergson rewrites both) |
| Training/tuning runs | yes IF checkpointed | tuning configs default to final-only saves; set a rolling `save_interval` on anything > 2h |
| ms probes | NO | all three trainings held in memory, nothing on disk until the end — treat as atomic, schedule accordingly |

## Interruption playbook

1. Diagnose before relaunching: read the ACTUAL error (`tr '\r' '\n' < log |
   grep -a Error`), not the tail — bars bury tracebacks mid-log, and a stale
   bar looks alive (compare log mtime to now).
2. Free disk first if the volume filled: delete `BALLAST_delete_me_on_disk_full`
   (recreate later), then provably-invalid artifacts only. Never delete
   anything paper-usable: if it isn't provably invalid or being actively
   re-derived by a running fixed pipeline, it stays.
3. Gate/purge invalid partials (`gate_ekfac.py ... --purge`; for a fit, delete
   the run's `hessian/`).
4. Relaunch through `scripts/launch_one.sh` (registry + GPU claims) and verify
   the run is TRAINING (progress bar / file mtimes), not just launched.
5. Orphan cleanup: killed parents leave spawn_main workers holding CUDA — kill
   pids whose ppid is 1. Watch pkill self-matches: bracket a character in the
   pattern (`pkill -f "ms[.]yaml"`) or the pattern kills your own shell.

## Known bergson issues (PRs pending)

- `_step_complete` granularity vs `kfac.part` (skips unfinished fits)
- `apply_batch_size` not exposed in config; default 32 OOMs 48 GB at 1.5B
- per-query MAGIC accepts empty/partial q.pt on resume
- hessian fit accumulators are checkpointable in principle (accumulative) but
  never persisted — an interruption always restarts the fit
- FSDP dtensor attr fix: PR #447
