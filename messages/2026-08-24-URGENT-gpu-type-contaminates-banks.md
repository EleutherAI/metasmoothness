# 2026-08-24 — URGENT: never shard one bank across A40 and A100 nodes

## The finding

A bank's retrains are deterministic on identical hardware and **not** across GPU
types. sm_muon was accidentally sharded across both, which gave a direct
measurement:

| subsets | hardware compared | mean disagreement in `diff` |
|---|---|---|
| 18-28 | A40 vs A40 | **2.5e-07** |
| 29-39 | A40 vs A40 | **6.7e-07** |
| 40-69 | A100 vs A40 | **6.9e-04** |
| 70-99 | A100 vs A40 | **6.9e-04** |

A40-vs-A40 agreement (2.5e-7) matches the 8k shard-boundary check (2.1e-6), so
retraining itself is reproducible. Across GPU types it is three orders of
magnitude worse.

**Why that matters more than it looks.** Split the A40-vs-A100 difference into
the offset shared by every subset of a query and the residual that differs
between them (3 pairs, 240 measurements, `data/gpu_noise_floor.csv`):

| part of the A40-vs-A100 difference | median | max |
|---|---|---|
| offset shared by all subsets of a query | **7.0e-04** | 3.2e-03 |
| residual that differs between subsets | **5.7e-05** | 2.6e-04 |

LDS is a **within-query rank** correlation, so an offset that moves all of a
query's subsets together cannot change it; only the residual can. Against the
subset signal it competes with -- each subset's deviation from its query mean,
median **4.7e-04** -- the residual is 12%. **A bank computed entirely on one GPU
type is therefore fine, and either type gives the same score.**

The damage comes from *mixing inside one query*. There the offset stops
cancelling and becomes a between-subset difference of 7.0e-04: **1.5x the
signal**, a perturbation that reorders subsets freely:

    sm_muon scored from the mixed A40/A100 set : 0.7828
    sm_muon scored from the homogeneous A40 set: 0.8379

**0.055 apart — larger than most optimizer effects in the grid** (the bs32
optimizer gap is +0.046). A bank whose ground truth came from two GPU types is
not internally comparable, and neither number is trustworthy.

## What to do

- **Every subset of a given query must come from the same GPU type.** In
  practice that means sharding a bank only across nodes of one type; splitting
  *queries* across hardware would be safe, but nothing in the pipeline does
  that. A40 fleet:
  bellflower-0, lucia-ord-0, secret-ord-0, allium-0, iris-0. A100: marisa-0,
  maria-1, shivam2-0. Do not mix, including the main process.
- **Record the GPU type in the row notes**, next to nproc. Constraint 2 already
  makes world size part of run identity; hardware is too, for the same reason.
- **Check any bank you sharded while the A100 pods were borrowed.** The audited
  ones: muon bs32 (lucia-ord-0 / bellflower-0 / secret-ord-0, all A40, clean),
  wd0.0, wd0.1, clip1.0, scale0.5, scale0.25 (all A40, clean). Both anchors were
  contaminated and are fixed.
- **Before sharding, list `validation_*.csv` in the run dir, not just
  `validation.csv`.** sm_muon had already been re-sharded by another node at
  04:30 after its main died; I did not check, and added a second, redundant set
  of slices at 09:12. That duplicated ~4 hours of compute and is what created
  the mixed-hardware bank in the first place.

## Both anchors are now done, on clean A40 data

    sm_adamw  0.9411 [0.9326, 0.9477]   half-width 0.0076 (tightest row in the grid)
    sm_muon   0.8379 [0.8205, 0.8519]   half-width 0.0157

**The anchor pair, the reference contrast for the whole grid:**
paired adamw - muon = **+0.1032 [+0.0851, +0.1213]**, and adamw wins **20/20
queries** -- the only unanimous contrast measured anywhere so far.

No recomputation was needed: sm_adamw's A40 main process had completed all 100
subsets on its own, and sm_muon's A40 slice set covered 18-99. The A100 slices
are renamed `validation_*.csv.a100` rather than deleted, so the comparison stays
reproducible.
