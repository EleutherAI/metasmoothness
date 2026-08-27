#!/usr/bin/env python3
"""Render the proponent-filter scaling curve from experiments.csv as ASCII.

    python scripts/scaling_plot.py            # print
    python scripts/scaling_plot.py --readme   # splice into README.md

No matplotlib in the pinned venv, and an ASCII block survives `less` and a
terminal README, so the plot is text. Regenerate whenever experiments.csv is
rebuilt - do NOT hand-edit the block in README.md.
"""
import argparse, csv, math, pathlib

ap = argparse.ArgumentParser()
ap.add_argument("--readme", action="store_true")
ap.add_argument("--height", type=int, default=16)
args = ap.parse_args()

SERIES = [("adam", "A", ("plan_adam_eps1e17_", "sm_adamw_eps1e17_")),
          ("muon", "M", ("plan_muon_eps1e17_", "sm_muon_eps1e17_"))]
NS = [4000, 8000, 16000, 32000, 64000]

rows = list(csv.DictReader(open("experiments.csv")))

def pick(prefixes, n):
    for r in rows:
        rid = r["run_id"]
        if not rid.endswith("_bs256"):
            continue
        if not rid.startswith(prefixes):
            continue
        try:
            if int(float(r["n_docs"])) != n:
                continue
        except (TypeError, ValueError):
            continue
        return r
    return None

data = {}
for name, mark, prefixes in SERIES:
    pts = []
    for n in NS:
        r = pick(prefixes, n)
        if r is None:
            pts.append((n, None, None, None, None, None))
            continue
        def f(k):
            v = (r.get(k) or "").strip()
            try:
                return float(v)
            except ValueError:
                return None
        pts.append((n, f("filter_ekfac_delta"), f("filter_ekfac_lo"),
                    f("filter_ekfac_hi"), f("metasmoothness"), r["run_id"]))
    data[name] = pts

vals = [d for pts in data.values() for (_, d, _, _, _, _) in pts if d is not None]
hi = max(vals) * 1.15 if vals else 0.06
H, COLW = args.height, 9

out = []
out.append("EK-FAC proponent-filter delta vs corpus size (bs256, 2 epochs)")
out.append("A = adamw   M = muon   | = 95% CI   lowercase a/m = ms below 0.98")
out.append("")
for row in range(H, -1, -1):
    y = hi * row / H
    label = ("%.3f" % y).rjust(6)
    line = label + " |"
    for i, n in enumerate(NS):
        cell = " " * COLW
        for name, mark, _ in SERIES:
            n_, d, lo, hi_, ms, rid = data[name][i]
            if d is None:
                continue
            band = hi / H / 2.0
            if abs(d - y) <= band:
                c = mark
                if ms is not None and ms < 0.98:
                    c = mark.lower()
                pos = 3 if name == "adam" else 5
                cell = cell[:pos] + c + cell[pos + 1:]
            elif lo is not None and hi_ is not None and lo - band <= y <= hi_ + band:
                pos = 3 if name == "adam" else 5
                if cell[pos] == " ":
                    cell = cell[:pos] + "|" + cell[pos + 1:]
        line += cell
    out.append(line.rstrip())
out.append("       +" + "-" * (COLW * len(NS)))
out.append("        " + "".join(("%dk" % (n // 1000)).center(COLW) for n in NS))
out.append("")
out.append("  N     steps   adamw delta            muon delta             ms(A)   ms(M)")
for i, n in enumerate(NS):
    a = data["adam"][i]
    m = data["muon"][i]
    def fmt(p):
        _, d, lo, hi_, ms, rid = p
        if d is None:
            return "     retraining     "
        return "%.5f [%.5f,%.5f]" % (d, lo, hi_)
    def msf(p):
        return "%.4f" % p[4] if p[4] is not None else "  -   "
    out.append("  %-5s %-7d %-22s %-22s %s  %s%s"
               % ("%dk" % (n // 1000), 2 * n // 256, fmt(a), fmt(m), msf(a), msf(m),
                  "   <- muon ms collapsed" if (m[4] is not None and m[4] < 0.98) else ""))

block = "\n".join(out)
print(block)

if args.readme:
    p = pathlib.Path("README.md")
    lines = p.read_text().split("\n")
    HDR = "## Proponent filter scaling (EK-FAC)"
    if HDR in lines:
        i = lines.index(HDR)
        j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith("## ")), len(lines))
        lines = lines[:i] + lines[j:]
    anchor = next((k for k, l in enumerate(lines) if l.startswith("## ")), len(lines))
    new = [HDR, "", "```", block, "```", ""]
    lines = lines[:anchor] + new + lines[anchor:]
    p.write_text("\n".join(lines))
    print("\n  spliced into README.md")
