# N=1M is blocked on an unreproducible dataset recipe, not on compute

Lucia wants the strongest findings to generalise to N=1M. ms is the only probe
that can reach there -- three trainings, no bank -- and 512k already runs. The
blocker is the dataset.

## What is established

The smollm2 ladder is STRICTLY NESTED. train_16k is a prefix of train_32k is a
prefix of ... of train_512k, verified by hashing rows 0, 1, n/2 and n-1 at each
level: every check passed. That nesting is what makes the dataset-size axis a
clean comparison -- a larger run sees exactly what the smaller one saw, plus
more. A 1M set built by resampling would break it silently and every ms value on
the ladder would stop being comparable to its neighbours.

So train_1m must have train_512k as its literal prefix.

## What blocks it

The recipe that produced these files is not in the repo. scripts/ has
prep_london.py and prep_london_heldout.py; there is no smollm2 equivalent.

Reconstructing it empirically failed. The format is clear enough -- 512 GPT-2
tokens per row, columns input_ids and length, contiguous chunks of a concatenated
document stream with NO EOS separator (row 0 contains no eos, and row 1 continues
the same topic mid-sentence). The content is web text: row 0 begins "Pure-bred or
Composite? - How 'pure' is a pure breed?".

But no straightforward streaming reproduction matches. Hashes of the first three
rows, against train_512k's 05bc3fd0fd / 34ebf14360 / f10d4cd2ff:

    smollm-corpus fineweb-edu-dedup, no eos    0979f16206 7ec1085215 d88db503ec
    smollm-corpus fineweb-edu-dedup, with eos  0979f16206 7ec1085215 c50925355d
    smollm-corpus cosmopedia-v2,     no eos    daa289da7d 247d9c46b3 259bbb2574
    smollm-corpus cosmopedia-v2,     with eos  daa289da7d 247d9c46b3 3ad517b983

None match. So a shuffle, a filter, a different split or a different source was
applied and is not recorded anywhere.

## What would unblock it

Any one of:

  * the original prep script or the command line that produced train_*.hf
  * the shuffle seed and source config, if it was a shuffle
  * a decision to accept a NON-NESTED train_1m, which costs the clean
    dataset-size comparison and should be an explicit choice rather than a
    silent one

scripts/prep_smollm2_1m.py is written and works; it verifies its packing against
train_512k before writing and refuses on mismatch, which is what caught this. Give
it the right source and it produces the file.

## Meanwhile

The ladder reaches 256k measured (0.9637) with 512k running, and the trend --
0.9800 / 0.9866 / 0.9869 / 0.9741 / 0.9637 -- extrapolates to roughly 0.94 at
N=1M. That is an estimate, not a measurement, and it is the best available until
the recipe is recovered.
