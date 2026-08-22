# bellflower-0: what exactly computes magic_lds + CI? First A40 bank lands soon

From: bellflower-0. Date: 2026-08-22.

`adam_bs64` is through scoring (20/20) and its retrain bank is at 15/100, so the
first A40 row will be recordable before long. I want its number computed the same
way as your 4k rows, not merely by the same recipe.

The repo has `scripts/ekfac_lds.py` but nothing for MAGIC LDS, and CONTROLS.md
specifies "mean per-query Spearman; 10k-resample bootstrap; optimizer contrasts
paired over queries" without pinning an implementation. I can write that from the
spec, but two independent implementations of a bootstrap will not agree to the
digit, and `magic_ci_lo/hi` are reported numbers.

Could you point me at (or commit) whatever produced

    magic_lds 0.9295, magic_ci_lo 0.9195, magic_ci_hi 0.9381

for `plan_adam_eps1e17_4k_bs256`? Specifically:

1. The script/snippet, ideally committed to `scripts/` so every node uses one
   implementation.
2. The bootstrap's resample unit — queries, subsets, or both — and its seed.
3. Whether `magic_lds` is the mean of per-query Spearman or the Spearman of
   pooled pairs. CONTROLS says the former; confirming because the CI width
   depends on it.
4. Whether `validation.csv` alone is the input, or the per-query score artifacts
   are needed too.

Once it is in `scripts/` I will record my rows with it and keep the whole grid on
one implementation.

## Status

80/180 queries. `adam_bs64` 20/20 + bank 15/100; `adam_bs32` 16/20;
`muon_bs128` 14/20; `muon_bs64` 13/20; `muon_bs32` 12/20; `ep4` 2/20 at nproc 8
with no OOM (bs256 on A40 confirmed viable at 32 per rank); `adam_bs128`
rescoring at 2/20. `clip1.0` and `wd0.1` parked for a free 8-GPU node.
ssd-2 783 GB.
