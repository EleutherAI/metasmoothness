# CONFIRMED: build a bank at any step count without waiting for MAGIC scoring

The filter-delta vs LDS correlation was stuck at 2000 steps because every row
above that is ms-only with no bank, and building one the normal way runs the magic
pipeline, which scores first. Scoring is serial by construction, so hardware does
not shorten it. Measured on the ladder rows:

    plan_adam_eps1e17_32k_bs32   ~38 h of scoring remaining
    plan_muon_eps1e17_32k_bs32   ~48 h
    plan_adam_eps1e17_64k_bs32   ~95 h
    plan_muon_eps1e17_64k_bs32  ~112 h

## The route

A bank is 100 leave-1%-out retrains plus their measured query-loss diffs. That is
`validate` with `method: lds`, and it does NOT need MAGIC scores -- EK-FAC scores
serve, and the ladder rows already have them. About 17 GPU-pair-hours per bank
against 38-112 h of unshortenable scoring.

## The recipe

Start from `gen_filter.py <run_id> --source ekfac --nproc 2 --no-bank
--random-n 100`, then fix three fields in the generated config:

    method: lds
        Valid values are exactly lds | filter-proponents | filter-detractors.
        "random" is not one of them.

    save_models: true
        THIS is what writes subsets.json -- the write sits inside
        `if save_models and global_rank == 0`. Filter configs leave it unset,
        bank configs set it true, and that single difference is why a filter run
        never yields a usable bank: you get diffs with no way to map them back to
        removed docs, so ekfac_lds cannot run.

    scores: <run>/ekfac_scores/scores
        method=lds asserts a scores path exists, otherwise
        "Path to attribution scores must be provided." Pointing it at EK-FAC
        makes validation.csv's score_sum EK-FAC based, which is what an EK-FAC
        LDS wants anyway.

Also drop `retrained_dir` if present -- that is what makes a filter run reuse an
existing bank instead of building one.

Then:

    ekfac_lds.py --scores <same scores> --bank <run_path>

## Confirmed working

plan_adam_eps1e17_32k_bs32, `bank_from_filter/`:

    subsets.json         written once validation starts
    subsets              100
    docs per subset      320   (exactly 1% of 32000)
    doc id range         0 - 31999
    distinct docs        20381 (overlap expected -- independent random draws)

Run reached `Validating: 0/100` with subsets.json in place, which was the last
unverified link.

## What it buys

A genuine bank at any step count the ladder reaches. EK-FAC LDS immediately, on
the weaker arm of the correlation (+0.129 [-0.189, +0.469] against MAGIC's
+0.561 [+0.212, +0.788]), and MAGIC LDS for free later, since a bank is
scorer-independent -- reuse rule 1.

Both 32k_bs32 arms are building now. The 64k arms are the obvious next targets,
and they are where the scoring wait is 95-112 h, so the saving is larger still.
