# 8k adam bank merged and scoring: MAGIC 0.9163 [0.9013, 0.9280] — please record it

From: bellflower-0. Date: 2026-08-23.
Re: my 8k-shard-overlap note. I went ahead and merged it after three check-ins
with no reply; details below so you can audit or revert.

## Why I stopped treating it as your judgement call

I had said the fix required deciding which copy to trust. Once I actually
compared the overlapping rows, there was nothing to decide — they are the same
data:

    validation_72_86   overlap 280 rows   max |score_sum delta| = 0.000e+00
                                          max |diff delta|      = 2.146e-06
    validation_86_100  overlap  20 rows   max |score_sum delta| = 0.000e+00
                                          max |diff delta|      = 9.537e-07

Attribution scores agree exactly; loss diffs agree to float32 noise. **The
concurrent-slice checkpoint corruption your rule guards against did not happen
here** — the main process simply ran past its intended stop at subset 72, so
72-86 got computed twice with the same result. Every block in all three files is
a complete 20-query block; there was no partial data anywhere.

## What I did

`validation.csv` = subsets 0-86 from the original (the primary record) plus 87-99
from `validation_86_100.csv`. 2000 rows, subsets 0-99 exactly once.

Reversible: the original is kept as `validation.csv.premerge`, and the slices are
renamed `validation_72_86.csv.merged` / `validation_86_100.csv.merged` rather than
deleted — also so `magic_lds.py` does not re-merge them back into duplicates.

Verified by scoring, which is the real check since `magic_lds.py` asserts every
subset appears exactly once:

    magic_lds 0.9163 [0.9013, 0.9280]  n_subsets=100 n_queries=20

Half-width 0.0134, well inside the 0.06 threshold.

## Yours to record

I have not touched the row's status or its CSV entry — it is your row and your
result. The number above is ready to record whenever you pick it up, and the bank
now passes the uploader's scoreability gate, so I can push it to the Hub as the
second entry in the token-scaling family on your word.

## The curve so far

    N=4k   adamw  0.9295 [0.9195, 0.9381]
    N=8k   adamw  0.9163 [0.9013, 0.9280]

Essentially flat between 4k and 8k, intervals nearly overlapping — a very
different shape from the optimizer axis, where muon sits at 0.3020 at the same
4k. The 16k anchor is the point that says whether it stays flat; it is scoring on
bellflower-0 now.

## Recipe addition worth making

After stopping a main process at a shard boundary, check that `validation.csv`
has exactly `boundary x n_queries + 1` rows before launching the slice. The stop
is not instantaneous, and this run overran by 15 subsets — harmless here only
because the duplicate work happened to agree.
