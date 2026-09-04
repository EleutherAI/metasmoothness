# Compute

What the paper cost, per stage, per figure and per run. Regenerate with

    python scripts/compute_ledger.py            # print from the cached scan
    python scripts/compute_ledger.py --scan     # re-walk both run trees and ~1 GB of logs
    python scripts/compute_ledger.py --write    # refresh data/compute_ledger.csv

The script derives figure membership from `scripts/scaling_plot_mpl.py`'s own axis
constants, the way `which_figure.py` does, so the ledger cannot drift from what is
drawn. The numbers below are its output; the per-run detail lives in
`data/compute_ledger.csv`.

## Hardware

Internal cluster, 8-GPU nodes. NVIDIA **A40 (48 GB)** on lucia-ord-0, secret-ord-0,
allium-0, shared-ord-0, bellflower-0 and iris-0; **A100-80GB** on lotus-0. Earlier
phases also used three borrowed A100-80GB pods (marisa-0, maria-1, shivam2-0), both
now off the allowlist. fp32 throughout, GPT-2-124M except where a row says otherwise.

Essentially every run is `nproc 2` — 802 of the 948 entries in
`paper_runs/_logs/launch_registry.tsv`. EK-FAC fits on the Qwen rows and some
large-N banks ran at 4; nothing ran at 8 that produced a paper number. Peak storage
~1 TB: ~28 GB of checkpoints per run plus ~0.5 GB per retrained model.

The fleet is deliberately mixed (D17), so per-retrain cost is not uniform across
rows at one corpus size: an A100 row runs roughly 2× an A40 row. The `src` column
in the ledger says whether a row's cost was measured from its own logs or supplied
by the law below.

## The cost law

Retraining dominates everything, and it is linear in corpus size. One 2-epoch
GPT-2-124M retrain costs **0.0347 wall-seconds per training document** at `nproc 2`,
and that held independently at every rung from 4k to 256k. So each stage has one
price at 4k and a power-of-two multiplier after it:

| documents | ×    | 1 retrain | LDS (100 + base) | QLD (23) | MAGIC (20 queries) |
|-----------|------|-----------|------------------|----------|--------------------|
| 4k        | 1×   | 0.08      | 8                | 1.8      | 11                 |
| 8k        | 2×   | 0.15      | 16               | 3.5      | 22                 |
| 16k       | 4×   | 0.31      | 31               | 7.1      | 44                 |
| 32k       | 8×   | 0.62      | 62               | 14.2     | 87                 |
| 64k       | 16×  | 1.23      | 125              | 28.4     | 174                |
| 128k      | 32×  | 2.47      | 249              | 56.8     | 348                |
| 256k      | 64×  | 4.94      | 498              | 113.5    | 697                |
| 512k      | 128× | 9.87      | 997              | 227.0    | 1394               |

GPU-hours. An **LDS** is 100 leave-1%-out retrains plus the base. A **QLD** is 23
retrains: 20 proponent-filtered, one per query, plus 3 random-removal controls —
and the controls are 3 models *for the whole row*, shared across scoring methods
via `retrained_dir`, so a second method on a row already measured costs 20, not 23.
See CLAUDE.md; sharding a filter S ways without `--controls shared` turns 3 controls
into 3S trainings.

**MAGIC** is the exception in kind, not in scaling: one reverse pass per query over
the whole corpus, strictly serial, unshardable because `ValidationConfig` exposes
`subset_start`/`subset_stop` but no query range. 0.245 GPU-s per document per query,
~1.4× that on bs≤32 rows.

**EK-FAC** (gradient collection, K-FAC fit, eigenvalue correction, scoring) has no
clean per-document law and is taken as measured: roughly flat at 3–7 GPU-h per row
below 64k, then linear — 24 at 128k, 40 at 256k, 51 at 512k.

**Qwen-1.5B** costs 2.9× GPT-2 per document to retrain and far more to fit. Its
Hessian work alone is 1,609 GPU-h, more than every GPT-2 EK-FAC fit in the paper
combined.

## Per figure

| figure | rows | LDS | MAGIC | EK-FAC | QLD | total | exclusive |
|---|---|---|---|---|---|---|---|
| `filter_scaling.png` | 8 | — | — | 134 | 893 | **1,047** | 397 |
| `filter_muon_appendix.png` | 25 | — | — | 224 | 666 | **923** | 211 |
| `filter_method_appendix.png` | 5 | — | 328 | 20 | 295 | **646** | 148 |
| `filter_variants_appendix.png` | 8 | — | — | 29 | 61 | **93** | 30 |
| `filter_vs_lds.png` | 24 | 727 | 1,106 | 59 | 353 | **2,254** | 1,644 |
| `qwen15b_heldout_trend.png` | 4 | — | — | 416 | 154 | **573** | 573 |
| `heldout_in_vs_out_partial.png` | 8 | — | — | 134 | 329 | **483** | 329 |
| **union, deduplicated** | **39** | **727** | **1,106** | **662** | **1,986** | **4,521** | |

GPU-hours. Rows are shared between figures, so the totals double-count on purpose:
*total* is everything a figure draws on, *exclusive* is what cutting only that
figure would save.

The shape of this table is the paper's own argument. `filter_vs_lds.png` is the only
figure that needs a 100-retrain bank or a MAGIC scoring pass, and it costs more than
the other six combined. Every other figure is built from 23-retrain QLDs.

## Per run

Full table in `data/compute_ledger.csv` (53 rows). The expensive end:

| run | N | retrain | LDS | MAGIC | EK-FAC | QLD | total |
|---|---|---|---|---|---|---|---|
| `plan_adam_eps1e17_512k_bs256` | 512k | 9.87 | — | — | 51 | 948 | **1,008** |
| `plan_adam_eps1e17_256k_bs256` | 256k | 4.94 | — | — | 40 | 568 | **612** |
| `qwen15b_256k_bs256` | 256k | 14.32 | — | — | 411 | — | **425** |
| `qwen15b_128k_bs256` | 128k | 7.16 | — | — | 394 | — | **402** |
| `qwen15b_64k_bs256` | 64k | 3.58 | — | — | 391 | — | **395** |
| `plan_adam_eps1e17_128k_bs256` | 128k | 2.47 | — | — | 24 | 363 | **389** |
| `plan_adam_eps1e17_64k_bs256` | 64k | 1.05 | 5 | 164 | 7 | 193 | **370** |
| `plan_adam_eps1e17_32k_bs256` | 32k | 0.62 | 63 | 87 | 4 | 99 | **254** |

The 16k rows cluster at 45–140 GPU-h each, and there the bank and the MAGIC pass
dominate — the QLD on top is 6–25. `plan_adam_eps1e17_4k_bs256` carries seven filter
campaigns, a bank and a MAGIC pass for 30 GPU-h, less than a seventh of one 512k QLD.

## Totals

| | GPU-h |
|---|---|
| Behind the figures (39 rows) | 4,521 |
| All 53 rows carrying a bank, a filter or a fit | 7,374 |
| + lr tuning (178 measured runs) and metasmoothness probes | **8,393 retained** |
| + identified work spent and discarded | **~8,900 for the project** |

Retrains on disk: 2,700 bank + 2,356 QLD query + 144 QLD control + 53 base = **5,253**.

Of the 7,374, **2,853 GPU-h is not drawn by any figure**. Two different things make
that up, and they should not be conflated:

- **14 rows feed no figure at all — 1,628 GPU-h.** Chiefly the Qwen 64k/128k/256k
  EK-FAC rungs (1,221, Tier 3 backlog), the bs32 token-axis rows, the London
  distribution-shift arm, `gpt2medium_16k_bs32` and the seed-43 replication.
- **The remaining 1,225 GPU-h sits on rows that do appear in a figure, in components
  no figure draws** — the jina baseline, MAGIC passes on rows the method appendix
  truncates away at 64k, and filter variants measured but not plotted.

## Spent and discarded

Disclose these; the ledger counts *retained* work only, so none of it is in the
numbers above except where noted.

- **29 tuning launches returned no held-out loss** — 247 GPU-h. Largely the hung
  runs below plus the 512k/1M rows that never finished.
- **13 London tuning runs hung on 14 GPUs for 10 h** — ~140 GPU-h. D19/D20: they
  reported healthy to every liveness check, which is why `hung_check.py` exists.
- **Four EK-FAC scorings lost** — two wedged writing to ssd-1 (~9.5 h each), two in
  D-state (~9 h each), ~75 GPU-h.
- **The Muon 64k bank, 57/100, discarded** — ~70 GPU-h. A partial bank is not a
  noisy finished one; `magic_lds.py` returns a clean-looking 0.8957 for it.
- **Every bank measured outside the pinned venv, struck and re-run** (D15). The
  artifacts were deleted, so these hours are not recoverable from disk and are not
  in the ~8,900.
- **13 held-out QLDs re-running every filter at N ≥ 64k** after the query set turned
  out to be contaminated — ~330 GPU-h. This one *is* inside the 7,374, since the
  re-runs are the measurement the paper reports.

Before the paper, a separate exploration phase on the retired volume
(`/mnt/ssd-1/lucia/bergson-damping/runs`) holds 168 run directories and ~23,000
retrains — GPT-2 on WikiText at 1 epoch, so cheaper per retrain, roughly
2,000–4,000 GPU-h. None of it appears in the paper.

## How these were derived, and what they are worth

Retrain counts are the union of subset indices in every `validation*.csv` and query
indices in every `filter_proponents.csv` across both roots, deduplicated against
shards, merge copies and relaunch races. Costs come from completed tqdm bars in the
run logs, with rank-duplicate lines collapsed; where a row's logs did not survive,
the cost law supplies the number.

Two traps the script exists to document, because both produced badly wrong first
answers:

- **tqdm prints `100%` before it is done.** A 128,000-step bar reads 100% from step
  127,360, so keying on the printed percentage counts the last few hundred frames of
  every bar as separate passes and turns an 11-hour K-FAC fit into 9,000 hours. Key
  on `done == total`.
- **Every rank writes its own bar to the same log.** Detecting a new pass by "elapsed
  went down" fires on every rank switch and inflates the total ~17×.

Treat the totals as **±20%**. They are reconstructed from artifacts, not read from a
job accounting system. Work redone after a crash or a resume is deliberately absent —
the logs show it is real (the `clip1.0` bank logged 149 `Validating` iterations for a
100-subset bank), but it belongs in the discarded section, not in a row's cost.

## For the checklist

> All experiments were performed on NVIDIA A40 or A100 GPUs. Each LDS required 100
> GPT-2 retrains, while each QLD used 23 retrains. Runs used two GPUs in fp32 on an
> internal cluster of 8-GPU nodes. The dominant cost is retraining, which is linear
> in corpus size at 1.9 × 10⁻⁵ GPU-hours per training document for 2-epoch GPT-2-124M;
> one LDS therefore costs 31 GPU-hours at 16k documents and 997 at 512k, while one
> QLD costs 7.1 and 227. MAGIC scoring adds a serial reverse pass per query that
> cannot be sharded. The figures in the paper account for approximately 4,500
> GPU-hours; the full research project used approximately 8,900, the difference being
> the learning-rate sweep, metasmoothness probes, backlog rungs that did not make the
> paper, and preliminary or failed runs. A separate earlier exploration phase, not
> reported here, used a further 2,000–4,000 GPU-hours.
