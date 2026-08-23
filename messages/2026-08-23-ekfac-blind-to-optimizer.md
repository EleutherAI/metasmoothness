# 2026-08-23 — EK-FAC cannot separate the optimizers; ekfac_lds.py was truncating

## Read this first if you have run scripts/ekfac_lds.py on a SHARDED bank

`ekfac_lds.py` did `next(bank.rglob("validation.csv"))`. On a sharded bank that
returns the **pre-shard prefix**, not the merged ground truth: `validation.csv`
holds only subsets 0..first_boundary and the rest live in
`validation_<a>_<b>.csv`, merged into `validation_merged.csv`.

muon bs32 scored **0.4512 on 22 of 100 subsets** and printed a perfectly
confident-looking number. The only tell was a CI half-width of 0.108, and only
if you checked it against the D6 threshold of 0.06. Corrected value on the full
bank: **0.4567 [0.4161, 0.4918]**.

`paired_diff.py` already had this fix; `ekfac_lds.py` did not. It now prefers
`validation_merged.csv` and **prints which file and how many subsets it used**,
so the next truncation is visible in the output rather than hidden in a CI.

The 4k pair — the only previously recorded EK-FAC cells — is unaffected, because
neither 4k bank was sharded. **If you recorded an EK-FAC number for any other
sharded bank, re-check it.**

## The result: EK-FAC does not see the optimizer contrast

All scored on existing banks (reuse rule 1, no rebuilds), D7 config inherited
from the accepted 4k template:

| bank | MAGIC | EK-FAC |
|---|---|---|
| adam bs32 | 0.9201 | 0.4586 [0.4194, 0.4934] |
| muon bs32 | 0.8737 | 0.4567 [0.4161, 0.4918] |
| adam bs64 | 0.7811 | 0.4239 [0.3854, 0.4593] |
| muon bs64 | 0.8690 | running |

At bs32 MAGIC separates the optimizers (paired +0.0464, 17/20 query wins) while
EK-FAC puts them 0.002 apart against half-widths near 0.04. muon bs64 is the
decisive test, because MAGIC **reverses** at bs64 (muon 0.8690 above adam
0.7811). If EK-FAC returns ~0.42 there too, it is blind to the contrast in both
directions, not merely insensitive.

Do not claim this from the bs32 pair alone.

## gpt2-medium will not finish on the borrowed pods

Measured on shivam2-0 (4x A100-80GB, nproc 4): MAGIC scoring is **~78 min per
query x 20 = ~26 h**, against the ~10 h in the signed-off D11 cost plan. Plus
~7 h for a sharded bank, so ~33 h against ~19 h of pod time left.

**Scoring cannot be sharded** — `ValidationConfig` exposes
`subset_start`/`subset_stop` but no query range, so the 20 queries are strictly
serial no matter how many pods are free. More hardware does not help while
scoring is the binding stage.

It stays on the A100s for the window (fastest hardware, and `per_query/*.pt` is
the resume unit so finished queries survive). **Before the pods go back, the run
directory must be moved from ssd-4 to ssd-2** — ssd-4 is mounted only on the
A100 pods, so leaving it there strands the work. Recorded in DECISIONS D11.

## Housekeeping others will hit

- Bank weights are written mode **0600**, and the fleet is split across uid 1000
  and uid 1001 (iris-0 and secret-ord-0 are 1001). This broke a bank upload and a
  held-out eval. Read permissions are now widened at all depths across the
  experiments trees on ssd-1 and ssd-2, from the uid-1000 side.
- `upload_bank.py` no longer publishes `ekfac_scores/`, `scores/` or
  `per_query/` — derived, scorer-specific, many GB, and recomputable from the
  bank in ~20 minutes. Six banks are now on the Hub, including both
  token-scaling adam points (4k and 8k).
