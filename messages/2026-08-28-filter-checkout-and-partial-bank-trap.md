# 2026-08-28 01:50 — filter runs need a different bergson checkout than training

Two findings from tonight that will cost you time if you hit them cold.

## Filter runs import `/mnt/ssd-2/lucia/bergson-filter`, not the training checkout

Training and EK-FAC scoring use `/mnt/ssd-1/lucia/bergson-main-paper-429` — that is
what `gen_tuning_run.py` and `gen_experiment_run.py` print in their canonical
invocation. **Filter runs do not.** A filter config is a `validate` step carrying a
`method` key, and that field only exists on `Validate` in `bergson-filter`. Launch a
filter against the pinned `-429` checkout and it dies immediately with

    RuntimeError: Couldn't instantiate class <class 'bergson.cli.commands.Validate'>
    ... Validate.__init__() got an unexpected keyword argument 'method'

I lost six shard launches to this. Worse, the older bank-shard message reports the
*silent* version of the same mismatch: against a checkout that tolerates it,
`method: lds` is dropped rather than rejected, so the run completes and means
something different from what the config says. Check the checkout before you trust a
filter result, not just before you launch one.

The safe move is to select it from the config rather than remember it:

    if grep -qa '^ *- *validate:' "$CFG"; then BERG=/mnt/ssd-2/lucia/bergson-filter
    else BERG=/mnt/ssd-1/lucia/bergson-main-paper-429; fi

## Do not read an LDS off the muon 64k bs256 bank — it is 57/100

`plan_muon_eps1e17_64k_bs256` has a `validation.csv`, so `magic_lds.py` will happily
return a number for it. I got **0.8957 [0.8787, 0.9088]** and discarded it, and you
should discard it too: the bank has 57 of 100 subsets.

This row's own comment in `build_experiments_csv.py` already records why. A partial
bank is not a noisy version of the finished one — it can be *precisely wrong* with a
tight-looking interval. Dropping 8 subsets from a finished 100-subset file reproduces
the bogus 0.1085 exactly, where the full bank gives 0.4146. A 57-subset estimate with
a ±0.015 interval is exactly the shape of that trap.

Completing it is 43 more retrains and **A100 only** — D17 makes GPU type part of run
identity and the bank was built on shivam2-0, which is now permanently off-limits, so
lotus-0 is the only candidate. That is a bank build, which D22 rules out, so I did
not start one.

Related: `ekfac_lds.py --scores <row>/scores` on these rows does **not** give you an
EK-FAC LDS. `<row>/scores` is the MAGIC scores directory written by the fused `magic`
step; EK-FAC scoring writes to `<row>/ekfac_scores`, which does not exist for either
64k bs256 row. Pointed at the MAGIC dir the script returns exactly the negation of the
MAGIC LDS — every per-query value flipped in sign. That is the recovery trick, not an
independent measurement, and it is easy to mistake for one.
