# Building a bank at 2000 steps without waiting 40h for MAGIC scoring

The filter-delta vs LDS correlation stops at 2000 steps, because every row above
that is ms-only with no bank. Building one the normal way runs the magic pipeline,
which scores first, and scoring is serial by construction so hardware does not
shorten it. Measured on the ladder rows:

    plan_adam_eps1e17_32k_bs32   4/20 queries, ~2.4 h each   ~38 h remaining
    plan_muon_eps1e17_32k_bs32   3/20 queries, ~2.8 h each   ~48 h remaining
    plan_adam_eps1e17_64k_bs32   1/20 queries, ~5.0 h each   ~95 h remaining
    plan_muon_eps1e17_64k_bs32   1/20 queries, ~5.9 h each  ~112 h remaining

## The route

A bank is 100 leave-1%-out retrains plus their measured query-loss diffs. That is
what `validate` with `method: lds` does, and it does NOT require MAGIC scores --
EK-FAC scores serve just as well, and both 32k_bs32 arms already have them.

So the bank can be built directly through the validate step, skipping scoring
entirely. Roughly 100 retrains x 2000 steps, about 17 GPU-pair-hours against 38 h
of scoring that cannot be parallelised.

## The recipe, with the two things that are easy to get wrong

Start from `gen_filter.py --no-bank --random-n 100`, then fix three fields:

    method: lds              # NOT filter-proponents. Valid values are exactly
                             # lds | filter-proponents | filter-detractors,
                             # and "random" is not one of them.
    save_models: true        # THIS is what writes subsets.json. The write sits
                             # inside `if save_models and global_rank == 0`, so
                             # without it you get diffs and no way to map them
                             # back to removed docs, and ekfac_lds cannot run.
                             # Filter configs leave save_models unset; bank
                             # configs set it true. That difference is the whole
                             # reason a filter run never produces a usable bank.
    scores: <run>/ekfac_scores/scores
                             # method=lds asserts a scores path exists:
                             # "Path to attribution scores must be provided."
                             # Pointing it at the EK-FAC scores makes
                             # validation.csv's score_sum EK-FAC based, which is
                             # what an EK-FAC LDS wants anyway.

Then `ekfac_lds.py --scores <same> --bank <run_path>` reads validation.csv's diff
column and subsets.json and produces the LDS.

## Status

Running on plan_adam_eps1e17_32k_bs32, output at `bank_from_filter/`, secret-ord-0
GPUs 2,3. Retrains are ~21 min each at 2000 steps.

The one thing still unconfirmed: subsets.json appears only once the validation
phase begins, and at the time of writing the run is still in its first training.
Check `bank_from_filter/subsets.json` exists before assuming the whole chain
works.

## What it buys

A genuine 2000-step bank on the step ladder. That gives an EK-FAC LDS immediately
-- the weaker arm of the correlation, +0.129 [-0.189, +0.469] against MAGIC's
+0.561 -- and a MAGIC LDS later for free, once the 38 h of scoring finishes, since
the bank is scorer-independent.
