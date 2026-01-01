"""Power-law scaling fits for the filter effect (QLD vs training-set size).

The incumbent model for QLD-vs-N in this repo is log-linear: delta = a + b*log2(N)
(`scale-line` in filter_extrapolation.py). This script asks whether the scaling-law
literature's standard recipe does better:

  * PARAMETERIZATION -- Kaplan et al. (2020), Goyal et al. (2024): the effect of a
    data choice is a power law in the data scale, delta = A * N^alpha, optionally
    with an additive irreducible term, delta = delta0 + A * N^alpha. Goyal's point
    is exactly that a filtering decision has no scale-free number attached to it:
    its value is a curve, so what we report should be an exponent, not a delta at
    one N.
  * FITTING -- Hoffmann et al. (2022): do not OLS on log y. Minimize a Huber loss
    (delta_h = 1e-3) on log-space residuals with L-BFGS from a grid of
    initializations, taking the best final objective, and fit the INDIVIDUAL
    observations rather than per-rung means. Both parts matter here: the per-query
    QLD distribution has a long right tail (one contaminated query can be 5x the
    rung mean), which is what a Huber loss is for.

Six models, all fitted to per-query QLDs unless --means:

    loglin        delta = a + b*log2(N)              OLS               (incumbent)
    kaplan-ols    log delta = log A + alpha*log N    OLS on logs       (incumbent
                                                     `scale-power`)
    kaplan-huber  delta = A * N^alpha                Huber/L-BFGS      (Kaplan +
                                                                       Hoffmann)
    kaplan-smear  kaplan-huber, back-transformed with Duan's smearing factor:
                  a log-space fit targets the geometric mean over queries, and
                  what this project plots is the arithmetic mean
    offset-huber  delta = delta0 + A * N^alpha       Huber/L-BFGS, log-sum-exp
                                                     parameterization as in
                                                     Hoffmann App. D
    sat-huber     delta = delta_inf - A * N^-alpha   Huber/L-BFGS (saturating
                                                     variant: does the filter
                                                     effect approach a ceiling?)

Scored two ways, because they disagree and the disagreement is the point:
RMSE in nats is dominated by the largest rung (QLD spans 20x across the row);
RMSE of log delta weights rungs equally and is the loss the fits optimize.

Reads data/qld_per_query.csv (from qld_query_variance.py). Writes
data/filter_scaling_law.csv. No GPU, no run directories touched. Findings and
the recommendation are in notes/filter_scaling_law.md; the figure is
scripts/scaling_law_plot.py.
"""
import argparse
import collections
import csv
import os

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "data", "qld_per_query.csv")
OUT = os.path.join(HERE, "data", "filter_scaling_law.csv")
HUBER_DELTA = 1e-3          # Hoffmann et al. (2022), App. D
LOG_N0 = np.log(1e4)        # center log N so the intercept is not wildly scaled


# ---------------------------------------------------------------- fitting core
def huber(r, d=HUBER_DELTA):
    a = np.abs(r)
    return np.where(a <= d, 0.5 * r * r, d * (a - 0.5 * d))


def _fit(logpred, inits, logn, logy, bounds=None):
    """L-BFGS from every init in `inits`; keep the best final objective."""
    def obj(p):
        lp = logpred(p, logn)
        if not np.all(np.isfinite(lp)):
            return 1e12
        return float(huber(lp - logy).sum())

    best, best_p = np.inf, None
    for p0 in inits:
        try:
            r = minimize(obj, np.asarray(p0, float), method="L-BFGS-B",
                         bounds=bounds, options={"maxiter": 2000})
        except Exception:
            continue
        if np.isfinite(r.fun) and r.fun < best:
            best, best_p = float(r.fun), r.x
    return best_p, best


def grid(*axes):
    return [np.array(c) for c in np.array(np.meshgrid(*axes)).T.reshape(-1, len(axes))]


# ------------------------------------------------------------------ the models
# Each model: fit(logn, y) -> (predict(N)->delta, params dict)

def fit_loglin(logn, y, p0=None):
    """Incumbent. OLS of delta on log2 N."""
    b, a = np.polyfit(logn / np.log(2), y, 1)
    return (lambda N: a + b * np.log2(N)), {"a": a, "b_per_doubling": b}


def fit_kaplan_ols(logn, y, p0=None):
    """Incumbent `scale-power`. OLS of log delta on log N -- no robustness, and
    the mean of the fitted log is not the fitted mean."""
    m = y > 0
    alpha, c = np.polyfit(logn[m] - LOG_N0, np.log(y[m]), 1)
    return (lambda N: np.exp(c + alpha * (np.log(N) - LOG_N0))), \
        {"A_at_1e4": np.exp(c), "alpha": alpha, "_p": np.array([c, alpha])}


def fit_kaplan_huber(logn, y, p0=None):
    m = y > 0
    ln, ly = logn[m] - LOG_N0, np.log(y[m])
    inits = grid(np.log([1e-3, 3e-3, 1e-2, 3e-2, 1e-1]),   # log delta at N=1e4
                 [0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
    p, loss = _fit(lambda p, x: p[0] + p[1] * x, [p0] if p0 is not None else inits,
                   ln, ly, bounds=[(-30, 5), (-1.0, 3.0)])
    c, alpha = p
    return (lambda N: np.exp(c + alpha * (np.log(N) - LOG_N0))), \
        {"A_at_1e4": np.exp(c), "alpha": alpha, "huber_loss": loss, "_p": p}


def fit_offset_huber(logn, y, p0=None):
    """delta = delta0 + A*N^alpha, via Hoffmann's log-sum-exp form:
    log delta_hat = LSE(log delta0, log A + alpha*log N)."""
    m = y > 0
    ln, ly = logn[m] - LOG_N0, np.log(y[m])

    def lp(p, x):
        return logsumexp(np.stack([np.full_like(x, p[0]), p[1] + p[2] * x]), axis=0)

    inits = grid(np.log([1e-4, 1e-3, 1e-2]),
                 np.log([1e-3, 1e-2, 1e-1]),
                 [0.0, 0.25, 0.5, 1.0])
    p, loss = _fit(lp, [p0] if p0 is not None else inits, ln, ly,
                   bounds=[(-30, 5), (-30, 5), (0.0, 3.0)])
    d0, c, alpha = np.exp(p[0]), np.exp(p[1]), p[2]
    return (lambda N: d0 + c * np.exp(alpha * (np.log(N) - LOG_N0))), \
        {"delta0": d0, "A_at_1e4": c, "alpha": alpha, "huber_loss": loss, "_p": p}


def fit_sat_huber(logn, y, p0=None):
    """delta = delta_inf - A*N^-alpha: growth toward a ceiling. No LSE trick (the
    terms subtract), so predictions are guarded against going non-positive."""
    m = y > 0
    ln, ly = logn[m] - LOG_N0, np.log(y[m])
    ymax = float(y[m].max())

    def pred(p, x):
        return np.exp(p[0]) - np.exp(p[1] - np.exp(p[2]) * x)

    def lp(p, x):
        v = pred(p, x)
        return np.where(v > 1e-12, np.log(np.maximum(v, 1e-12)), -1e6)

    inits = grid(np.log([ymax, 2 * ymax, 5 * ymax, 20 * ymax]),
                 np.log([1e-3, 1e-2, 1e-1, 1.0]),
                 np.log([0.1, 0.3, 1.0]))
    # alpha is bounded away from 0: as alpha -> 0 the ceiling runs off to
    # infinity and the model degenerates into an unidentifiable straight line.
    p, loss = _fit(lp, [p0] if p0 is not None else inits, ln, ly,
                   bounds=[(np.log(ymax), 5), (-30, 5),
                           (np.log(0.02), np.log(3.0))])
    at_bound = abs(p[2] - np.log(0.02)) < 1e-6
    dinf, A, alpha = np.exp(p[0]), np.exp(p[1]), np.exp(p[2])
    return (lambda N: dinf - A * np.exp(-alpha * (np.log(N) - LOG_N0))), \
        {"delta_inf": dinf, "A_at_1e4": A, "alpha": alpha,
         "alpha_at_floor": at_bound, "huber_loss": loss, "_p": p}


def fit_kaplan_huber_smear(logn, y, p0=None):
    """kaplan-huber, then Duan's (1983) smearing correction.

    A log-space fit predicts the GEOMETRIC mean of the per-query QLDs. The
    quantity this project reports and plots is the ARITHMETIC mean over the 20
    queries, and per-query QLD is strongly right-skewed (a single query can be
    5x the rung mean), so the raw back-transform under-predicts what it is
    compared against. Smearing rescales by the mean of the exponentiated
    residuals -- a nonparametric estimate of E[exp(eps)] -- which costs one
    number and no distributional assumption.
    """
    pred, par = fit_kaplan_huber(logn, y, p0=p0)
    m = y > 0
    resid = np.log(y[m]) - np.log(pred(np.exp(logn[m])))
    smear = float(np.mean(np.exp(resid)))
    par = dict(par, smearing=smear)
    return (lambda N: smear * pred(N)), par


MODELS = collections.OrderedDict([
    ("loglin", fit_loglin),
    ("kaplan-ols", fit_kaplan_ols),
    ("kaplan-huber", fit_kaplan_huber),
    ("kaplan-smear", fit_kaplan_huber_smear),
    ("offset-huber", fit_offset_huber),
    ("sat-huber", fit_sat_huber),
])


# -------------------------------------------------------------------- the data
def load(path):
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(path)):
        cells[(r["method"], r["rule"], r["qset"])][int(r["n"])].append(float(r["qld"]))
    return {k: {n: np.array(v) for n, v in sorted(d.items())} for k, d in cells.items()}


def obs_arrays(series, ns, means):
    """(logN, y) over rungs `ns`: one row per query, or one per rung if means."""
    if means:
        return np.log(np.array(ns, float)), np.array([series[n].mean() for n in ns])
    logn = np.concatenate([np.full(len(series[n]), np.log(n)) for n in ns])
    y = np.concatenate([series[n] for n in ns])
    return logn, y


def degenerate(mname, par, y):
    """Say out loud when a fit has collapsed onto a boundary or killed its own
    extra parameter. Both happen here, and both are the answer to "is the extra
    term supported?" -- not something to leave for the reader to spot in a
    parameter string."""
    w = []
    if mname == "offset-huber" and par["delta0"] < 0.01 * float(np.min(y)):
        w.append("delta0 -> 0 (%.2g, vs smallest rung mean %.2g): the additive "
                 "floor is not supported; this IS the pure power law"
                 % (par["delta0"], float(np.min(y))))
    if mname == "sat-huber":
        if par.get("alpha_at_floor"):
            w.append("alpha pinned at the 0.02 bound and delta_inf=%.3g >> any "
                     "observed QLD: no ceiling is identifiable in this range"
                     % par["delta_inf"])
        elif par["delta_inf"] > 5 * float(np.max(y)):
            w.append("delta_inf=%.3g is %.0fx the largest rung mean: the ceiling "
                     "is far outside the data and unconstrained"
                     % (par["delta_inf"], par["delta_inf"] / float(np.max(y))))
    if mname == "kaplan-huber" and abs(par["alpha"]) < 0.05:
        w.append("alpha ~ 0: this series does not scale")
    return w


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def rmse_log(pred, obs):
    """Log-space RMSE, or nan if the model predicts a non-positive QLD -- which
    the log-linear model does at small N, and which is not a large error but a
    statement the model should not be able to make."""
    if np.any(np.asarray(pred) <= 0):
        return float("nan")
    return rmse(np.log(pred) - np.log(obs))


def zero_crossing(pred):
    """Smallest power-of-2 doc count in [1e3, 1e7] where the fit is positive."""
    ns = np.logspace(3, 7, 400)
    v = np.asarray(pred(ns))
    if v[0] > 0:
        return None
    up = np.where(v > 0)[0]
    return float(ns[up[0]]) if len(up) else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--min-rungs", type=int, default=6,
                    help="skip series with fewer rungs (need train + holdout)")
    ap.add_argument("--means", action="store_true",
                    help="fit rung means instead of individual per-query points")
    ap.add_argument("--compare-qsets", nargs=2, metavar=("METHOD", "RULE"),
                    help="bootstrap the in-dist minus held-out difference in "
                         "alpha for one method/rule, on the rungs both have")
    ap.add_argument("--boot", type=int, default=1000,
                    help="bootstrap reps over queries for the alpha CI "
                         "(0 = off; 1000 reps is a few minutes on CPU)")
    args = ap.parse_args()

    cells = load(args.csv)

    if args.compare_qsets:
        method, rule = args.compare_qsets
        a = cells[(method, rule, "in-dist")]
        b = cells[(method, rule, "held-out")]
        ns = sorted(set(a) & set(b))
        print("%s / %s, rungs %s -- paired over the SAME trained runs, so the "
              "two query sets differ only in whether the query text is in the "
              "pool" % (method, rule, [n // 1000 for n in ns]))
        rng = np.random.default_rng(0)
        alpha = lambda d: MODELS["kaplan-ols"](*obs_arrays(d, ns, False))[1]["alpha"]
        for tag, d in (("in-dist", a), ("held-out", b)):
            print("  %-9s alpha = %.3f  (QLD x%.2f per doubling of N)"
                  % (tag, alpha(d), 2 ** alpha(d)))
        diff = []
        for _ in range(max(args.boot, 1)):
            r = [{n: d[n][rng.integers(0, len(d[n]), len(d[n]))] for n in ns}
                 for d in (a, b)]
            diff.append(alpha(r[0]) - alpha(r[1]))
        diff = np.array(diff)
        print("  difference = %.3f  [%.3f, %.3f]  (%d boots over queries)"
              % (diff.mean(), np.percentile(diff, 2.5),
                 np.percentile(diff, 97.5), len(diff)))
        return

    out_rows = []
    for key in sorted(cells):
        series = cells[key]
        ns = sorted(series)
        if len(ns) < args.min_rungs:
            continue
        method, rule, qset = key
        name = "%s / %s / %s" % (method, rule, qset)
        ymean = np.array([series[n].mean() for n in ns])
        npos = sum(int((series[n] <= 0).sum()) for n in ns)
        print("\n=== %s -- %d rungs %s" % (name, len(ns), [n // 1000 for n in ns]))
        if npos:
            print("    %d non-positive per-query QLDs dropped from log-space fits" % npos)

        # ---- full-row fit: parameters and in-sample quality
        logn, y = obs_arrays(series, ns, args.means)
        print("    %-13s %-42s %8s %8s" % ("model", "params (full row)",
                                           "rmse", "rmse_log"))
        fits = {}
        for mname, f in MODELS.items():
            pred, par = f(logn, y)
            p = pred(np.array(ns, float))
            fits[mname] = (pred, par)
            ps = " ".join("%s=%.4g" % (k, v) for k, v in par.items()
                          if k not in ("huber_loss", "_p"))
            lg = rmse_log(p, ymean)
            zc = zero_crossing(pred)
            note = "" if zc is None else "   [QLD<=0 below N=%.0f]" % zc
            print("    %-13s %-42s %8.4f %8s%s"
                  % (mname, ps[:42], rmse(p - ymean),
                     "neg" if np.isnan(lg) else "%.4f" % lg, note))
            if mname == "kaplan-smear" and par["smearing"] > 1.02:
                print("      ! smearing factor %.3f: the raw log-space fit "
                      "under-predicts the mean-over-queries by %.0f%%"
                      % (par["smearing"], 100 * (par["smearing"] - 1)))
            for w in degenerate(mname, par, ymean):
                print("      ! %s" % w)
            out_rows.append({"method": method, "rule": rule, "qset": qset,
                             "fit": "full-row", "n_train_rungs": len(ns),
                             "model": mname, "n_docs": "", "pred": "",
                             "obs": "", "rmse_nats": rmse(p - ymean),
                             "rmse_log": lg, "params": ps})

        # ---- per-rung residuals: a power law that is right on average can
        # still be wrong in a pattern, and the pattern is what a kink looks like
        print("    residual (pred - obs mean), nats, by rung:")
        print("      %-13s %s" % ("rung",
                                  " ".join("%8s" % ("%dk" % (n // 1000)) for n in ns)))
        for mname in ("loglin", "kaplan-huber"):
            r = fits[mname][0](np.array(ns, float)) - ymean
            print("      %-13s %s" % (mname, " ".join("%+8.4f" % v for v in r)))

        # ---- alpha with a bootstrap CI over queries (the unit CONTROLS pairs over)
        if args.boot and not args.means:
            rng = np.random.default_rng(0)
            for mname in ("kaplan-ols", "kaplan-huber"):
                als = []
                warm = fits[mname][1].get("_p")
                for _ in range(args.boot):
                    res = {n: series[n][rng.integers(0, len(series[n]), len(series[n]))]
                           for n in ns}
                    ln, yy = obs_arrays(res, ns, False)
                    try:
                        als.append(MODELS[mname](ln, yy, p0=warm)[1]["alpha"])
                    except Exception:
                        pass
                als = np.array(als)
                print("    %-13s alpha = %.3f  [%.3f, %.3f]  (%d boots over queries)"
                      % (mname, fits[mname][1]["alpha"],
                         np.percentile(als, 2.5), np.percentile(als, 97.5), len(als)))

        # ---- extrapolation: fit the first k rungs, predict the larger ones
        for k in range(3, len(ns)):
            tr, te = ns[:k], ns[k:]
            ltr, ytr = obs_arrays(series, tr, args.means)
            yte = np.array([series[n].mean() for n in te])
            print("    train %s -> test %s"
                  % ([n // 1000 for n in tr], [n // 1000 for n in te]))
            for mname, f in MODELS.items():
                try:
                    pred, par = f(ltr, ytr)
                    p = pred(np.array(te, float))
                except Exception as e:
                    print("      %-13s FAILED %s" % (mname, e))
                    continue
                lg = rmse_log(p, yte)
                det = "  ".join("%dk:%.4f/%.4f" % (n // 1000, pp, oo)
                                for n, pp, oo in zip(te, p, yte))
                print("      %-13s rmse %.4f  rmse_log %5s   %s"
                      % (mname, rmse(p - yte),
                         "neg" if np.isnan(lg) else "%.3f" % lg, det))
                for n, pp, oo in zip(te, p, yte):
                    out_rows.append({"method": method, "rule": rule, "qset": qset,
                                     "fit": "extrapolate", "n_train_rungs": k,
                                     "model": mname, "n_docs": n, "pred": pp,
                                     "obs": oo, "rmse_nats": rmse(p - yte),
                                     "rmse_log": lg, "params": ""})

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "rule", "qset", "fit",
                                          "n_train_rungs", "model", "n_docs",
                                          "pred", "obs", "rmse_nats", "rmse_log",
                                          "params"])
        w.writeheader()
        w.writerows(out_rows)
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
