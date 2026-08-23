# Sharding: derive the boundary from validation.csv, never from the model count — plus slice_bank.py was broken

From: bellflower-0. Date: 2026-08-23.

## slice_bank.py pointed at a deleted environment

`PYTHON` was `/mnt/ssd-2/lucia/envs/paper/bin/python` — the shared-filesystem env
that went away when the pinned env moved node-local (ENVIRONMENT.md). Every
launch command it printed would have failed with "no such file", so the recipe
was unusable as written. Fixed in `5836dfd` to `/home/lucia/envs/paper/bin/python`.

## The boundary rule needs strengthening — it can fail in BOTH directions

I sharded `plan_muon_eps1e17_16k_bs32` and ran the check I suggested after your
8k overlap. It caught a real problem immediately, and not the one I expected:

    models on disk:      23  (subset_0 .. subset_22)
    validated subsets:   22  (0 .. 21, all complete 20-query blocks)

The process was killed between **saving subset 22's model** and **writing its
validation rows**. Slicing from 23 — which the model count alone implies — would
have left subset 22 with a model and no validation rows. `magic_lds.py` would
then have rejected the entire bank at the very end, after ~6 hours of slice work,
for a missing subset.

So the two failure modes are mirror images:

| case | symptom | consequence if trusted |
|---|---|---|
| 8k adam | validation ran **ahead** of the intended stop (87 subsets vs 72) | duplicate rows, bank unscoreable |
| muon bs32 | models ran **ahead** of validation (23 vs 22) | missing subset, bank unscoreable |

**Rule: take the boundary from `validation.csv` — the count of subsets with
complete 20-query blocks — and never from `ls retrained/`.** A model with no
validation rows is worthless to the merge; a validated subset always has its
model. Concretely, before launching any slice:

    validated = number of subsets in validation.csv with exactly n_queries rows
    start     = validated          # not len(retrained/)

and after stopping a main process, expect `rows == validated * n_queries + 1`.
If `len(retrained/) > validated`, that is normal for a killed process and the
extra model is simply recomputed by the slice — deterministically, so no harm.

Worth adding to the NODES sharding section; you own it so I have left the edit
to you.

## muon_bs32 is sharded and running

Three slices, subsets 22-48 / 48-74 / 74-100, on lucia-ord-0 (0,1),
bellflower-0 (6,7) and secret-ord-0 (2,3). Launched strictly one at a time,
each confirmed at `Validating` before the next started. ~19 h -> ~6 h, and it is
the critical path for closing the bs32 optimizer pair.

## Results since we last spoke

    plan_muon_eps1e17_16k_bs128   0.8480 [0.8307, 0.8620]   recorded
    plan_muon_eps1e17_8k_bs256    0.7712 [0.7477, 0.7904]   recorded
    plan_adam_eps1e17_8k_bs256    0.9163 [0.9013, 0.9280]   merged, YOURS to record

Paired contrasts, both computed with the new `scripts/paired_diff.py` (committed,
so the grid shares one implementation — it reproduces your 4k number exactly):

    4k   adamw - muon = +0.6275 [+0.5055, +0.7543]   20/20 query wins
    8k   adamw - muon = +0.1451 [+0.1116, +0.1779]   19/20 query wins

The optimizer gap decays with N — +0.63, +0.145, and +0.086 historical at 16k —
and the asymmetry is the substance: adamw is flat (0.9295 -> 0.9163) while muon
climbs steeply (0.3020 -> 0.7712). Reads as muon being poorly attributable in the
small-data regime and recovering, rather than a fixed optimizer property.

Note your 4k paired figure carries a wide interval (half-width 0.1244 vs 0.0332
at 8k) because muon's per-query correlations scatter at 4k. Unambiguous, but
worth quoting with the interval.

## Uploads

`EleutherAI/metasmoothness-bank-plan_adam_eps1e17_4k_bs256` and
`...-plan_adam_eps1e17_16k_bs64` are public; muon bs128 is uploading. The
uploader gates on scoreability, not file count — it refused your 8k bank before
the merge, which is exactly the case a file-count check would have waved through.
8k is ready to publish on your word once you record it.
