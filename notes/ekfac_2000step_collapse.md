# EK-FAC at 2000 steps is 0.4146 -- the collapse I reported was an artefact of an unfinished bank

**This note previously claimed EK-FAC collapses to ~0.1 at 2000 steps. That was
wrong.** The claim came from a bank at 92/100 subsets. At 100/100 the same row,
same scores, same script reads:

    plan_adam_eps1e17_32k_bs32   ekfac_lds 0.4146 [0.3719, 0.4541]   100 subsets

which is inside the 0.404-0.473 band every other EK-FAC row occupies. So the open
question in `notes/ekfac_floor.md` -- robustness or floor -- is NOT settled by
this row, and the flatness now extends to 2000 steps rather than breaking there.

## What actually happened, because it matters more than the retraction

The obvious explanation is that the partial bank was corrupt: shards mid-write,
torn rows. It was not. Dropping those same 8 subsets (21-23, 28-29, 39, 49, 79)
from the finished, verified 100-subset file reproduces the low value exactly:

    100 subsets              0.4146 [0.3719, 0.4541]
    the same data minus 8    0.1085 [0.0476, 0.1655]

Identical to the preliminary number, to four decimals. The data was always fine.
**Eight subsets out of a hundred move this estimate from 0.11 to 0.41.**

That is the real finding here, and it is a methodological one. LDS is a mean over
queries of a Spearman taken ACROSS subsets, so a handful of subsets sitting at the
extremes of the score-sum range can anchor the whole rank correlation. Drop them
and what remains is noise-dominated. This says the n=100 estimate itself is not
obviously stable at this step count, and it says something sharper about process:

**A partial bank is not a noisy version of the finished one. It can be
confidently, precisely wrong.** The 92-subset interval was [0.0476, 0.1655] --
tight, and nowhere near the value the same data produces once complete. Nothing
about that interval warned me. Reporting it as preliminary-but-directional was
the mistake; the direction was the part that was wrong.

## Consequences

  * `plan_muon_eps1e17_32k_bs32` was reported at 0.0907 from 70/100 subsets. That
    number should be treated as meaningless until the bank finishes, not as a
    lower bound or a hint. It is at 88/100.
  * Do not record or report an LDS from an incomplete bank at all. `merge_bank.py`
    already prints `INCOMPLETE: n subset(s) absent`; that line should be treated
    as disqualifying rather than as a caveat to carry alongside the number.
  * The step-count question is still open. Every LDS in the grid, EK-FAC and
    MAGIC alike, now sits at 250 steps or fewer except this one 2000-step EK-FAC
    point, and that point is unremarkable. MAGIC at 2000 steps is still scoring
    (28%) and remains the number that would actually test the setup.

## What stands

The `merge_bank.py` work in the same commit is unaffected and still needed: five
scripts fall back to the pre-shard prefix when no merged file exists, no merged
file existed anywhere, and the fallback gave 0.4147 on 21 prefix subsets. It
reproduces every recorded value exactly at 100 subsets (ep4 0.4730, gpt2-medium
0.4189, 32k_bs256 0.4127, muon 16k_bs16 0.4276, muon 32k_bs256 0.4044).
