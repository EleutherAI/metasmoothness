# The london tuning is finished and has nowhere to go

Every london sweep in build_tuning_csv.py declares what it is choosing an lr for:

    selects_lr_for = plan_adam_london16k_bs256
                     plan_muon_london16k_bs256
                     plan_adam_london{32,64,128}k_bs256   (and muon)

**None of those rows exist in experiments.csv.** A grep for "london" over the
whole file returns nothing. So the tuning has settled lrs for experiment rows
that were never defined, and the ablation cannot produce a single ms or LDS
value until they are.

## What is settled and waiting

    london 16k bs256    adamw 8e-4  3.8397     muon 8e-4  3.8394
    london 32k bs256    adamw 8e-4  3.7873     muon 8e-4  3.7842
    london 64k bs256    adamw 1.6e-3 3.6099    muon 1.6e-3 3.5993   (interior)
    london 16k bs16     adamw 2e-4  3.8463     muon 2e-4  3.8240
    london 128k bs256   adamw 1.6e-3 3.4992    muon: sweep running

That is 8 settled lr choices across four corpus sizes and two batch sizes, at a
cost of roughly 30 tuning runs, and none of it has an experiment row to feed.

The two london runs that DID produce ms -- london16k_bs256_adamw at 0.9867 and
london16k_bs256_muon at 0.8547 -- are not experiment rows either. They exist as
run directories under paper_runs and their numbers live in notes, not in
experiments.csv, so they are outside the generator entirely and outside every
consistency check it performs.

## Why this matters for the question being asked

The london ablation exists to test whether the fine-tuning setup is "too easy"
because the corpus is close to pre-training. That test is a comparison of ms and
LDS between smollm2 and london at matched N. The smollm2 side is a full grid in
experiments.csv. The london side is currently two numbers in a note.

## What it needs

Experiment rows for london at 16k/32k/64k bs256 for both optimizers, carrying the
settled lrs above, plus the 16k bs16 pair. Then the ms probes and banks run
through the same generator, the same guards and the same reuse rules as
everything else.

This is a design addition rather than a bug fix -- it changes the experiment grid
-- so it is recorded here rather than done unilaterally at the end of a long
session. It is the single highest-value next step for the london work: without it
roughly 30 completed tuning runs stay unusable.
