# 2026-08-24 — MAGIC scoring cost scales with corpus size and cannot be parallelised

## The measurement

MAGIC does one reverse pass per query over the whole training set, so the step
count per query is `n_docs / batch_size * epochs`. Measured on the fleet:

| row | docs | model | steps/query | s/step | **per query** | **x20 queries** |
|---|---|---|---|---|---|---|
| anchor bs256 | 16k | gpt2 | 125 | ~28 | ~1.0 h | ~19 h (done) |
| `32k_bs256` | 32k | gpt2 | 250 | ~28 | ~2.0 h | ~39 h |
| `64k_bs256` | 64k | gpt2 | 500 | ~36 | **~5.0 h** | **~100 h** |
| `gpt2-medium` | 16k | gpt2-medium | 125 | ~35 | ~1.2 h | ~26 h |

The 64k rows are each **~100 hours of scoring before a single retrain starts**,
and they are 5-6 queries in.

## Why more hardware does not fix it

`ValidationConfig` exposes `subset_start`/`subset_stop` but **no query range**.
Sharding therefore parallelises the *retrain bank* only, and slices explicitly
require all queries to be scored first (`slice_bank.py` resumes from
`per_query/`). While a run is in MAGIC, extra GPUs cannot help it at all -- the
20 queries are strictly serial.

Raising `nproc` mid-run is not a workaround either: world size is part of run
identity (constraint 2), and per D17 the hardware is too, so a run that changes
either mid-flight produces a bank that is not internally comparable.

## What this means for planning

- **The N axis is not linear in wall-clock, it is linear in docs *and* serial.**
  4k/8k/16k are done; 32k is ~39 h; 64k is ~100 h each and they are the long
  pole for the whole grid, not the bank stage.
- **Budget scoring before claiming a large-N or large-model row.** The D11 cost
  plan under-estimated gpt2-medium scoring by 2.6x for exactly this reason, and
  it was already flagged there as the least certain, unshardable stage.
- **Idle GPUs during a MAGIC phase are not waste that can be reclaimed for that
  row.** They are better spent on EK-FAC scoring of finished banks (~20-40 min,
  no retraining, reuse rule 1) or on bank uploads.

## Nothing is being stopped

The 64k rows have ~25-30 h of scoring invested each and the token axis needs the
point, so they keep running. Flagging the timeline rather than the decision:
if the axis should stop at 32k, that is a controls call, not a scheduling one.
