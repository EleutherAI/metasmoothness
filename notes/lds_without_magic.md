# A route to LDS at higher step counts without waiting 40-100h for MAGIC scoring

The correlation between filter delta and LDS tops out at 2000 steps, because every
row above that is a ms-only ladder row with no bank. Building a bank the normal
way means running the magic pipeline, which scores first. Measured rates on the
ladder rows:

    plan_adam_eps1e17_32k_bs32   4/20 queries, ~2.4 h each   ~38 h remaining
    plan_muon_eps1e17_32k_bs32   3/20 queries, ~2.8 h each   ~48 h remaining
    plan_adam_eps1e17_64k_bs32   1/20 queries, ~5.0 h each   ~95 h remaining
    plan_muon_eps1e17_64k_bs32   1/20 queries, ~5.9 h each  ~112 h remaining

Scoring is serial by construction (Lucia), so no amount of hardware shortens it.

## The observation

`ekfac_lds.py` does not need MAGIC scores. It reads only the `diff` column out of
validation.csv, computes its own score sums from the EK-FAC scores, and correlates
the two. So an EK-FAC LDS needs the RETRAINS, not the scoring.

And EK-FAC scores for both 32k_bs32 arms are already done.

`gen_filter --no-bank` already retrains random subsets and measures exactly the
right quantity. From a completed run:

    random_filter.csv   subset,query,n_removed,baseline_loss,filtered_loss,loss_change
    validation.csv      subset,query,diff,score_sum

and the two agree up to sign -- subset 0 query 0 reads diff +0.00064063 against
loss_change -0.00064134. So `--random-n 100` on a ladder row produces the diffs a
bank produces, at 100 retrains x 2000 steps, roughly 17 GPU-pair-hours. That is
overnight on one pair, against 38 h of unshortenable scoring.

## The open question, which is why this is a note and not a run

`ekfac_lds.py` also wants `subsets.json` -- which documents each subset removed --
so it can sum scores over the right docs. Banked rows have it at the run root. The
no-bank filter run currently in flight has not written one, but it has not reached
its random phase yet, so whether it ever does is untested.

If it does: this works as described, and the correlation gains 2000-step points on
both arms.

If it does not: the fix is small, since the subsets are chosen inside the same
code path that writes random_filter.csv, and dumping them is a few lines.

Check `filter_proponents_ekfac/` on plan_adam_eps1e17_32k_bs32 once its random
phase starts.

## Caveat

This gives an EK-FAC LDS, not a MAGIC one, because MAGIC scores are the thing
that is 40 h away. That is still useful -- the EK-FAC arm of the correlation is
the weaker one, +0.129 [-0.189, +0.469] against MAGIC's +0.561 -- and it is the
arm most in need of rows.
