"""Normalized QLD: the share of a run's learned capability that the filter undoes.

    (filtered - random) / (base - unfiltered)

Numerator is the QLD every `figures/filter_*.pdf` already plots -- how much
removing the proponents costs the query, over and above removing the same number
of random documents. Denominator is how much training on the corpus bought that
query in the first place: the untrained base model's loss
(scripts/base_query_losses.py) minus the trained, unfiltered model's. So 0.6
reads "removing the proponents takes back 60% of what the corpus taught the
model about these queries", and 1.0 means the proponents account for all of it.

## Ratio of means, not mean of ratios

Per query the denominator crosses zero -- on the GPT-2 AdamW row it spans -0.07
to +0.36 nats at 4k, negative on 10 of 20 queries there (2 at 16k, none from 64k
up), because on a small corpus training leaves some queries *worse* than the base
model. A per-query ratio therefore divides by ~0 and explodes, and its mean is
meaningless. Every point here is instead

    mean_q(filtered - random) / mean_q(base - unfiltered)

which is well defined as long as the *mean* denominator is bounded away from
zero. `drawable()` enforces that: a rung whose denominator is within
MIN_DENOM_SNR standard errors of zero is dropped rather than drawn as a spike,
and the plot scripts print each drop instead of letting it vanish (PROJECT_RULES.md: a
script that only prints a number you should have reacted to is a bug). On the
GPT-2 AdamW row that removes 4k and 8k, where the model has learned almost
nothing about the queries and the share is a division by noise.

## The interval

The CI is a paired bootstrap over queries: resample the 20 query indices once,
then recompute BOTH means from that same resample. Resampling numerator and
denominator independently would destroy the pairing -- a query's loss level
enters `base` and `unfiltered` alike and cancels in the denominator, exactly as
it cancels in the QLD numerator, and that pairing is why a QLD interval is ~12x
tighter than the corresponding absolute-loss interval. The interval is
percentile, on the ratio itself, so it is not symmetric. It is wider in relative
terms than the QLD interval because the denominator carries its own noise.
"""
import csv
import math
import pathlib
import random
import statistics

import absolute_losses as absl
import base_query_losses as bql

ROOT = pathlib.Path(__file__).resolve().parent.parent
BOOT = 10000
# A rung is drawn only when its mean denominator is this many of its own standard
# errors above zero; below that the ratio divides by noise.
MIN_DENOM_SNR = 3.0
_MODELS = {"gpt2": "gpt2", "gpt2-medium": "gpt2-medium", "gpt2-large": "gpt2-large"}


def base_key(run_id, qset="indist"):
    """The base_query_losses key for a run: its own starting model and logit
    scale, on `qset`. Read from experiments.csv rather than hardcoded per figure,
    so a variant row that changes the base model or the logit scale gets the
    matching denominator automatically -- the scale-0.25 row is 3.4998 unfiltered
    against a scaled base of 7.58, and against the unscaled 3.45 its denominator
    would come out negative. Runs absent from experiments.csv (Qwen) pass their
    key explicitly."""
    row = _rows().get(run_id)
    assert row, f"{run_id} not in experiments.csv; pass base_key= explicitly"
    model = _MODELS.get(row["model"])
    assert model, f"{run_id}: unmapped model {row['model']!r}"
    scale = float(row["logit_scale"] or 1.0)
    return f"{model}{'' if scale == 1.0 else f'@{scale:g}'}/{qset}"


def _rows(_cache={}):
    if not _cache:
        _cache.update({r["run_id"]: r
                       for r in csv.DictReader(open(ROOT / "experiments.csv", newline=""))})
    return _cache


def pairs(run, subdir, key, queries=None, **kw):
    """[(qld, denominator)] over the queries this run and the base cache share.

    `queries` restricts to a set of query ids. Restricting to the queries whose
    denominator is positive makes the low rungs drawable, but it is selection on
    the divisor, not noise removal, and it changes the estimand -- see
    notes/normalized_qld.md. Pass a FIXED set across rungs if you use it at all:
    excluding per rung compares a different query set at every N.
    """
    base = bql.load().get(key)
    assert base, f"no base losses for {key!r}; run scripts/base_query_losses.py"
    return [(filtered - rand, base[q] - unfiltered)
            for q, unfiltered, filtered, rand in absl.per_query(run, subdir, **kw)
            if q in base and (queries is None or q in queries)]


def drawable(ps, min_queries=2):
    """Is the mean denominator far enough above zero for a ratio to mean anything?"""
    den = [d for _, d in ps]
    if len(den) < max(2, min_queries):
        return False
    m = statistics.fmean(den)
    return m > MIN_DENOM_SNR * statistics.stdev(den) / math.sqrt(len(den))


def ratio(ps):
    return statistics.fmean([n for n, _ in ps]) / statistics.fmean([d for _, d in ps])


def ci(ps, boot=BOOT, min_queries=2):
    """(ratio, err_lo, err_hi) as a fraction, or None when not drawable."""
    if not drawable(ps, min_queries):
        return None
    r = ratio(ps)
    rnd = random.Random(0)
    bs = sorted(ratio([rnd.choice(ps) for _ in ps]) for _ in range(boot))
    return r, r - bs[int(.025 * boot)], bs[int(.975 * boot)] - r


def point(run, subdir, key=None, *, qset="indist", boot=BOOT, min_queries=2,
          percent=True, queries=None, **kw):
    """(share, err_lo, err_hi) or None -- the shape absolute_losses.point returns,
    so a plot script can hand it straight to ax.errorbar. In percent by default."""
    p = ci(pairs(run, subdir, key or base_key(run, qset), queries, **kw), boot, min_queries)
    return None if p is None else tuple(100 * v for v in p) if percent else p


def report(tag, run, subdir, p, ps=None):
    """One line per point, naming the reason a dropped point was dropped."""
    if p is not None:
        return print(f"{tag} {p[0]:6.1f}%  [-{p[1]:.1f} +{p[2]:.1f}]  ({run}/{subdir})")
    ps = ps if ps is not None else []
    if len(ps) < 2:
        print(f"{tag} dropped: {len(ps)} queries ({run}/{subdir})")
    else:
        den = [d for _, d in ps]
        m = statistics.fmean(den)
        se = statistics.stdev(den) / math.sqrt(len(den))
        print(f"{tag} dropped: denominator {m:+.4f} +-{se:.4f} nats is within "
              f"{MIN_DENOM_SNR:g} SE of 0 (SNR {m / se:.2f}) ({run}/{subdir})")
