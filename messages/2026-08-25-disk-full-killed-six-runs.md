# 2026-08-25 — ssd-2 filled and silently killed six runs; what to check

## What happened

ssd-2 hit 0 bytes free. Six runs died within the same window, none loudly:

| run | state when it died | lost |
|---|---|---|
| `gpt2-medium` | 15/20 queries | nothing (per_query is resume state) |
| `adam_32k` | 18/20 queries | nothing |
| `adam_64k` | 6/20 queries | nothing |
| `muon_64k` | 8/20 queries | nothing |
| `ep4` | 19/20 queries | nothing |
| `muon_32k` | 20/20 queries, 47/100 models | nothing |

All six resumed cleanly because `per_query/*.pt` and `retrained/subset_*` are
resume state. **No measured work was lost** -- but ~1h45m of fleet time was,
because nothing announced the failure.

## What filled it

The tail-filter runs. Each one retrains the base model and saves a full
`sqrt`-spaced trajectory (~15-28 GB depending on the row), and six ran at once:
roughly 100 GB on top of an already-tight volume. Two things worth knowing:

- That trajectory **duplicates the bank's own** -- the filter run retrains a base
  the bank already has checkpoints for.
- It is written **once, at startup**, not per query. I initially reported it as
  "overwritten every query"; that was wrong. Checkpoint mtimes span a single
  ~3.5-minute window with sqrt-spaced steps, and `retrain_and_eval` never saves.
- The per-query *filtered* models get no checkpoints at all, so a later MAGIC run
  on a filtered model is not possible today regardless of save mode.

`save_mode: "final"` is implemented on bergson branch `feat/save-mode-final` for
when a run genuinely does not need the trajectory. Note `save_mode: interval`
is NOT a substitute -- the config docstring says it is unsupported by MAGIC.

## How to notice faster

`_orchestration/watchdog_unfinished.py` reports the newest log age per unfinished
row and is what found all six. It is not automatic. Until it is, run it after
anything that could have disturbed the fleet -- a full disk, a node reboot, a
kill:

    /home/lucia/envs/paper/bin/python -s -P \
      /mnt/ssd-2/lucia/paper_runs/_orchestration/watchdog_unfinished.py

Two caveats it has: lotus-0 writes its logs outside the run dir, so its rows
always look stale; and a row whose newest log is `ms.log` may have a finished ms
probe and a dead main run, so check `main_alive` separately rather than trusting
the age alone.

**Check `df -h /mnt/ssd-2` before launching anything that retrains in bulk.** A
filter sweep, a bank, or a set of ms probes can each add tens of GB per row.
