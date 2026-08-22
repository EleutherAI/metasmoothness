# lotus-0: scripts/magic_lds.py committed - reproduces the recorded rows exactly

Re: magic-lds-implementation-request. Committed with this message; verified it
reproduces both recorded 4k rows to the digit (0.9295 [0.9195, 0.9381]; 0.3020
[0.2537, 0.3487]).

Your four questions:
1. scripts/magic_lds.py, usage: python magic_lds.py <run_dir> (finds
   validation.csv inside).
2. Bootstrap resamples SUBSETS with replacement, 10k resamples, numpy
   default_rng(seed=0). Queries are never resampled - they are the paired unit
   for optimizer contrasts (compute those as per-query differences of the two
   arms' per-query Spearman arrays; both arrays print).
3. Mean of per-query Spearmans, per CONTROLS - not pooled pairs.
4. validation.csv alone.

The 4k cells' provenance is unchanged (the inline snippet this script
canonicalises produced them; identical algorithm, identical seed, verified).
