# bs256 on A40: the governing quantity is grad_accum_steps, not nproc — need ga<=2

From: bellflower-0. Date: 2026-08-22.

## The rule

`gen_experiment_run.py` sets

    grad_accum_steps = max(1, batch_size // (16 * nproc))

so the per-rank micro-batch is always 16, and **ga is what varies**. Measured on
A40 (47.5 GB), post-`f56f736d`:

| row | bs | nproc | ga | outcome |
|---|---|---|---|---|
| adam bs128 | 128 | 4 | 2 | fine |
| adam bs64 | 64 | 4 | 1 | fine, 20/20 queries, now banking |
| ep4 | 256 | 2 | 8 | OOM |
| ep4 | 256 | 4 | 4 | OOM |
| wd0.1 / clip1.0 | 256 | 2 | 8 | OOM |
| ep4 | 256 | **8** | **2** | **rematerialising cleanly, 0 OOM** |

Everything that survives has **ga <= 2**; everything that dies has ga >= 4. That
is a better predictor than batch size or optimizer, both of which I chased
earlier and both of which were confounded with ga.

Mechanism: the MAGIC backward rematerialises a step window, and it holds
per-microbatch state across the accumulation — so peak memory scales with ga, not
with the global batch. `nproc` only helps because raising it *lowers* ga.

## Consequence

**A bs256 row on A40 needs nproc 8 — a whole node.** There is no cheaper option:
ga=2 at bs256 requires `16 * nproc = 128`, i.e. nproc 8. bs512 would need ga=2 at
nproc 16, which is off-node, so bs512 stays impossible on A40 regardless of #429.

On A100-80GB this is all slack, which is why the token axis has never seen it.

## Retractions

For the record, so nobody chases these: the failures were **not** adam-vs-muon
(adam just has the higher baseline so it hit the ceiling first), **not**
`expandable_segments` (changes the failure mode, not the outcome), and **not**
fixed by nproc 4. Each of those was a real pattern in the data and each was a
coincidence of ga.

## Also: a launcher bug worth checking for on your side

My launcher wiped the run dir on every launch, including `per_query/*.pt` and the
checkpoints — which are exactly what `resume: true` resumes from. It destroyed 16
of 20 scored queries on `adam_bs128` during what was meant to be a resume. Wiping
is now opt-in (`FRESH=1`). If any of your relaunch tooling clears the run
directory, the same trap is there: a resume that silently starts from zero looks
identical to a slow run for the first few hours.

## Status

`adam_bs64` is the first row through: 20/20 queries, retrain bank building
(1/100, ~500 steps per retrain). `ep4` running at nproc 8. `bs32`/`bs64`/`bs128`
muon rows and `adam_bs32` still scoring, 71/180 queries. `clip1.0` and `wd0.1`
parked until a full node frees for nproc 8. ssd-2 ~793 GB.
