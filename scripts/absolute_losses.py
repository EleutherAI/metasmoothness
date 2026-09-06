"""Per-query absolute losses behind a proponent-filter result.

Every QLD figure plots filter_change - random_mean. The `<figure>_absolute.pdf`
companions plot the three losses those differences are made of, averaged over
the same queries with the same 95% CI:

    unfiltered  baseline_loss             filter_proponents.csv   the row's own model
    filtered    filtered_loss             filter_proponents.csv   proponents removed
    random      baseline + random_mean    filter_summary.csv      mean over the control retrains

random_filter.csv carries the control losses per subset, but it is absent for
bank-consumed rows and partial for sharded ones, whereas random_mean sits in
every summary the QLD figures already read, so the control loss is rebuilt from
the summary (loss_change = filtered - baseline, so baseline + mean change is
the mean control loss). Both files are read from the canonical <run>/<subdir>/
or, when that is missing or incomplete, pooled from <subdir>_q<a>_<b>[_suffix]
shards whose LOCAL query i is global a+i (scripts/merge_filter_shards.py).
"""
import csv
import glob
import math
import os
import random
import re
import statistics

from matplotlib.lines import Line2D

ROOTS = ["/mnt/ssd-2/lucia/paper_runs/experiments",
         "/mnt/ssd-1/lucia/paper_runs/experiments"]
# Shard dir = <subdir>_q<a>_<b>, optionally followed by a node suffix
# (filter_proponents_ekfac_q19_20_shared45). Anchored so a sibling result such
# as <subdir>_heldout_q0_5 or a *_qswap_* run cannot be pooled by mistake.
SHARD_RE = re.compile(r"^_q(\d+)_(\d+)(?:$|_)")
SKIP = (".nan", ".invalid", ".partial", ".failed", ".merge_tmp")

# 2026-09-05: the *_absolute companions are switched off. Their interval is set
# by between-query spread (queries differ by ~1.4 nats, a filter moves one by
# ~0.1) and is near-identical on all three series, so they hide the paired
# effect the QLD and *_relative figures show. The plotting code is kept; flip
# this to bring them back and re-add them to make_figures.PRODUCERS.
WRITE_ABSOLUTE = False
# 2026-09-06: the *_relative companions are off by default as well (Lucia: a simpler
# figures directory). QLD_RELATIVE=1 (or make_figures.py --relative) brings them back;
# make_figures.py drops them from EXPECTED_FIGURES when this is False.
WRITE_RELATIVE = os.environ.get("QLD_RELATIVE", "0") == "1"
# 2026-09-06: the logit-scale variants belong to the ms/speculative line of work, not this
# paper; excluded from the variants figure and the LDS-vs-QLD scatter.
EXCLUDE_RUNS = {"plan_adam_eps1e17_16k_scale0.5", "plan_adam_eps1e17_16k_scale0.25"}

# 2026-09-06: QLD_STAT=median swaps the per-query aggregate every QLD point
# draws (and what boot_ci resamples) from the mean to the median.
# scripts/make_figures.py --stat median sets it for every producer and writes
# the set into figures_median/ (FIGURES_DIR) and tables_median/ (TABLES_DIR);
# unset, nothing changes.
STAT = os.environ.get("QLD_STAT", "mean")
if STAT not in ("mean", "median"):
    raise SystemExit(f"QLD_STAT must be 'mean' or 'median', not {STAT!r}")


def agg(vals):
    """The per-query aggregate: mean, or the median under QLD_STAT=median."""
    return statistics.median(vals) if STAT == "median" else statistics.fmean(vals)


def label(text):
    """Axis label naming the aggregate: unchanged for the mean; under
    QLD_STAT=median 'Query loss difference' -> 'Median query loss difference'
    and 'Mean query loss' -> 'Median query loss'."""
    if STAT == "mean":
        return text
    if text.startswith("Mean "):
        return "Median " + text[5:]
    return "Median " + text[0].lower() + text[1:]


SERIES = ("unfiltered", "random", "filtered")
STYLE = {"unfiltered": dict(linestyle=":", marker="^", markersize=4, linewidth=1.6),
         "random": dict(linestyle="--", marker="s", markersize=4, linewidth=1.6),
         "filtered": dict(linestyle="-", marker="o", markersize=5, linewidth=2)}
LABEL = {"unfiltered": "Unfiltered", "random": "Random filter",
         "filtered": "Proponent filter"}
# Single-condition panels colour the unfiltered and random series neutrally so
# they read the way figures/filter_scaling_absolute.pdf does.
NEUTRAL = {"unfiltered": "#222222", "random": "#8f8f8f"}


def run_dir(run):
    return next((os.path.join(r, run) for r in ROOTS
                 if os.path.isdir(os.path.join(r, run))), None)


def _rows(path):
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _read(run, subdir, name, min_rows=20, pool=True):
    """Rows of <name> keyed by global query: the canonical file when it has
    min_rows queries, otherwise the union of the finished shards (canonical
    kept if the shards cover no more)."""
    d = run_dir(run)
    if d is None:
        return {}
    canon = {}
    for r in _rows(os.path.join(d, subdir, name)):
        try:
            canon.setdefault(int(float(r["query"])), r)
        except (KeyError, ValueError):
            pass
    if len(canon) >= min_rows or not pool:
        return canon
    pooled = {}
    for sd in sorted(glob.glob(os.path.join(d, subdir + "_q*_*"))):
        base = os.path.basename(sd)
        if not os.path.isdir(sd) or any(t in base for t in SKIP):
            continue
        m = SHARD_RE.match(base[len(subdir):])
        if not m:
            continue
        a, b = map(int, m.groups())
        for r in _rows(os.path.join(sd, name)):
            try:
                g = a + int(float(r["query"]))
            except (KeyError, ValueError):
                continue
            if a <= g < b:
                pooled.setdefault(g, r)
    return pooled if len(pooled) > len(canon) else canon


def per_query(run, subdir, min_rows=20, pool_summary=True):
    """[(query, unfiltered, filtered, random)] over the summary's queries that
    have finite losses. The unfiltered/filtered pair comes from the query's
    filter_proponents.csv row; a query whose summary row was recovered without
    one (scripts/recover_shard_summary.py) falls back to the baseline_loss in
    random_filter.csv, with filtered = baseline + the summary's filter_change."""
    prop = _read(run, subdir, "filter_proponents.csv", min_rows)
    summ = _read(run, subdir, "filter_summary.csv", min_rows, pool=pool_summary)
    rand = None
    out = []
    for q in sorted(summ):
        try:
            rm = float(summ[q]["random_mean"])
            if q in prop:
                b = float(prop[q]["baseline_loss"])
                f = float(prop[q]["filtered_loss"])
            else:
                if rand is None:
                    rand = _read(run, subdir, "random_filter.csv", min_rows)
                if q not in rand:
                    continue
                b = float(rand[q]["baseline_loss"])
                f = b + float(summ[q]["filter_change"])
        except (KeyError, ValueError):
            continue
        if all(map(math.isfinite, (b, f, rm))):
            out.append((q, b, f, b + rm))
    return out


def sem_ci(vals):
    """(mean, err_lo, err_hi): 1.96 x SEM over queries, as heldout_plot.py and
    qwen_scaling_plot.py draw their QLD bars. A median has no SEM, so under
    QLD_STAT=median this is the bootstrap interval of the median (boot_ci)."""
    if STAT != "mean":
        return boot_ci(vals)
    m = statistics.fmean(vals)
    e = 1.96 * statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return m, e, e


def boot_ci(vals, boot=10000):
    """(mean, err_lo, err_hi): seeded bootstrap of the mean over queries, the
    interval scaling_plot_mpl.py draws on every summary-derived point. Under
    QLD_STAT=median the statistic bootstrapped is the median (agg)."""
    rnd = random.Random(0)
    bs = sorted(agg([rnd.choice(vals) for _ in vals]) for _ in range(boot))
    m = agg(vals)
    return m, m - bs[int(.025 * boot)], bs[int(.975 * boot)] - m


def qld_from_summary(run, subdir, min_rows=20, boot=10000):
    """(agg, err_lo, err_hi) of filter_change - random_mean over the summary's
    finite queries, boot_ci over queries as every summary-derived point, or
    None below min_rows. The summary is the canonical <run>/<subdir>/ file or,
    when that is incomplete, the pooled shards (_read). Under QLD_STAT=median
    this replaces the experiments.csv filter_<m>_delta/_lo/_hi columns, which
    hold only the mean (scripts/filter_deltas.py)."""
    d = []
    for r in _read(run, subdir, "filter_summary.csv", min_rows).values():
        try:
            v = float(r["filter_change"]) - float(r["random_mean"])
        except (KeyError, ValueError):
            continue
        if math.isfinite(v):
            d.append(v)
    if len(d) < min_rows:
        return None
    return boot_ci(d, boot)


def point(run, subdir, *, ci=sem_ci, min_rows=20, min_queries=2, pool_summary=True):
    """{'unfiltered'|'random'|'filtered': (mean, lo, hi), 'n': queries} or None."""
    rows = per_query(run, subdir, min_rows, pool_summary)
    if len(rows) < min_queries:
        return None
    p = {k: ci([r[i] for r in rows]) for i, k in ((1, "unfiltered"), (2, "filtered"), (3, "random"))}
    p["n"] = len(rows)
    return p


def draw(ax, xs, points, color, series=SERIES, neutral=False, labels=False):
    """One errorbar line per series through the non-None points. neutral colours
    the unfiltered/random series black/grey for single-condition panels."""
    kept = [(x, p) for x, p in zip(xs, points) if p]
    if not kept:
        return
    for s in series:
        c = NEUTRAL.get(s, color) if neutral else color
        ax.errorbar([x for x, _ in kept], [p[s][0] for _, p in kept],
                    yerr=[[p[s][1] for _, p in kept], [p[s][2] for _, p in kept]],
                    color=c, capsize=3, capthick=1.2,
                    label=LABEL[s] if labels else None, **STYLE[s])


def handles(conditions=(), series=SERIES, series_color="#666666"):
    """Legend handles: one solid line per (label, colour) condition, then one
    grey line per series showing its linestyle/marker."""
    h = [Line2D([], [], color=c, linewidth=2, label=l) for l, c in conditions]
    h += [Line2D([], [], color=series_color, label=LABEL[s],
                 **{k: v for k, v in STYLE[s].items() if k != "linewidth"})
          for s in series]
    return h


def legend_above(fig, handles, ncols, fontsize=8, top=0.87):
    """Lay out the axes, then put a legend in the strip above them.

    Not ax.legend(..., bbox_to_anchor=(0, 1.08)): an axes-anchored legend counts
    as that axes' own artist, so tight_layout reserves the legend's full width
    inside the LEFT column of the grid and shoves the panels apart -- the wide
    gap the *_absolute figures used to open between (a) and (b). A figure legend
    is laid out against the figure instead, so both panels keep the full width
    the QLD figures get, and `rect` holds the figure at its declared size rather
    than growing it the way bbox_inches="tight" would.
    """
    fig.tight_layout(rect=[0, 0, 1, top])
    fig.legend(handles=handles, frameon=False, fontsize=fontsize, ncols=ncols,
               loc="upper center", bbox_to_anchor=(0.5, 1.0), borderaxespad=0.2)


def legend_above_each(fig, pairs, ncols, fontsize=7, top=0.84):
    """Per-panel legends above their own axes, for figures whose panels legend
    different series (qwen: top x% vs top k, so one shared legend cannot serve).

    Same fix as legend_above, kept per-axes: set_in_layout(False) takes each
    legend out of tight_layout's accounting, so it stops reserving the legend's
    width inside that column and splaying the panels apart. `rect` then leaves
    the strip the legends occupy.

    pairs: [(ax, handles), ...]
    """
    for ax, handles in pairs:
        leg = ax.legend(handles=handles, frameon=False, fontsize=fontsize, ncols=ncols,
                        loc="lower left", bbox_to_anchor=(0, 1.06), borderaxespad=0)
        leg.set_in_layout(False)
    fig.tight_layout(rect=[0, 0, 1, top])


def relative_point(run, subdir, *, ci=boot_ci, min_rows=20, min_queries=2, pool_summary=True):
    """(mean, err_lo, err_hi) of the relative query-loss increase, in percent:

        100 * (filtered_q - random_q) / random_q, averaged over queries

    The `<figure>_relative.pdf` companions: the QLD in percent of the matched
    control's loss. Same query, same number of documents removed, the only
    difference is which ones. Denominator is 2.7-3.4 nats, never near zero, so
    unlike a share of the learning (scripts/normalized_qld.py) it is stable on
    every row. Curves keep the QLD figures' shape (random_q differs from the
    unfiltered loss by ~1e-4); it puts GPT-2 and Qwen on one axis, which nats
    do not."""
    rows = per_query(run, subdir, min_rows, pool_summary)
    if len(rows) < min_queries:
        return None
    return ci([100 * (f - r) / r for _, _, f, r in rows])
