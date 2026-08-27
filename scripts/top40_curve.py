#!/usr/bin/env python3
"""Fixed-count (top-40 document) filter curve, adamw, bs256.

    python scripts/top40_curve.py            # print
    python scripts/top40_curve.py --readme   # splice into README.md

The main curve removes 1% of the corpus at every N, so "documents removed" grows
with N and is confounded with it. This variant removes a FIXED 40 documents at
every N (fraction = 40/N), which holds the removal constant.

Kept separate from filter_deltas.py on purpose: that script keys a result by
basename(dir).split("_")[-1], and "filter_top40_ekfac" also ends in "ekfac", so
it would collide with the 1% result for the same run and be dropped silently.
"""
import argparse, csv, os, pathlib, random, statistics

ap = argparse.ArgumentParser()
ap.add_argument("--readme", action="store_true")
ap.add_argument("--boot", type=int, default=10000)
args = ap.parse_args()

ROOTS = ["/mnt/ssd-2/lucia/paper_runs/experiments", "/mnt/ssd-1/lucia/paper_runs/experiments"]
ROWS = [(4000, "plan_adam_eps1e17_4k_bs256"),
        (8000, "plan_adam_eps1e17_8k_bs256"),
        (16000, "sm_adamw_eps1e17_16k_bs256"),
        (32000, "plan_adam_eps1e17_32k_bs256"),
        (64000, "plan_adam_eps1e17_64k_bs256")]


def delta(path):
    rows = list(csv.DictReader(open(path)))
    d = [float(r["filter_change"]) - float(r["random_mean"]) for r in rows]
    rnd = random.Random(0)
    boot = sorted(statistics.fmean([rnd.choice(d) for _ in d]) for _ in range(args.boot))
    lo, hi = boot[int(.025 * args.boot)], boot[int(.975 * args.boot)]
    rank1 = sum(1 for r in rows if int(float(r["rank"])) == 1)
    return statistics.fmean(d), lo, hi, len(rows), rank1


out = ["EK-FAC proponent filter, FIXED 40 documents removed (adamw, bs256, 2 epochs)",
       "", "%6s %9s %10s %24s %8s" % ("N", "frac", "n_removed", "delta [95% CI]", "rank1")]
for n, run in ROWS:
    root = next((r for r in ROOTS if os.path.isdir(os.path.join(r, run))), None)
    p = os.path.join(root, run, "filter_top40_ekfac", "filter_summary.csv") if root else None
    frac = 40.0 / n
    if p and os.path.isfile(p):
        m, lo, hi, nq, r1 = delta(p)
        out.append("%6s %9.6f %10d %24s %6d/%d"
                   % ("%dk" % (n // 1000), frac, 40,
                      "%.5f [%.5f,%.5f]" % (m, lo, hi), r1, nq))
    else:
        out.append("%6s %9.6f %10d %24s %8s"
                   % ("%dk" % (n // 1000), frac, 40, "retraining", "-"))
block = "\n".join(out)
print(block)

if args.readme:
    p = pathlib.Path(__file__).resolve().parent.parent / "README.md"
    lines = p.read_text().split("\n")
    HDR = "## Fixed-count filter curve (top 40 documents)"
    if HDR in lines:
        i = lines.index(HDR)
        j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith("## ")), len(lines))
        lines = lines[:i] + lines[j:]
    try:
        anchor = lines.index("## Alternative corpus: london")
    except ValueError:
        anchor = next((k for k, l in enumerate(lines) if l.startswith("## ")), len(lines))
    new = [HDR, "",
           "Same rows as the scaling curve above, but removing a fixed 40 documents at",
           "every N instead of 1%. Regenerate with `python scripts/top40_curve.py --readme`.",
           "", "```", block, "```", ""]
    lines = lines[:anchor] + new + lines[anchor:]
    p.write_text("\n".join(lines))
    print("\n  spliced into README.md")
