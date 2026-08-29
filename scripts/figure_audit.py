"""Every point each figure needs, and whether it exists / is running / is unscheduled.

Reads the axis constants out of scaling_plot_mpl.py rather than restating them, so
this cannot drift from what the figures actually draw. A point counts as PRESENT
only if the delta is in experiments.csv, which is what the plots read -- a value
sitting in data/filter_deltas.csv but not synced across is still a hole in the
figure, and that is exactly how the Muon 128k point stayed invisible.

Pass a file of live config paths to distinguish RUNNING from MISSING.
"""
import csv
import glob
import os
import re
import sys

ROOT = "/mnt/ssd-2/lucia/metasmoothness"
E = "/mnt/ssd-2/lucia/paper_runs/experiments"
src = open(ROOT + "/scripts/scaling_plot_mpl.py").read()

NS = eval(re.search(r"^NS = (\[[^\]]*\])", src, re.M).group(1))
BATCHES = eval(re.search(r"^BATCHES = (\[[^\]]*\])", src, re.M).group(1))
TOP40 = eval(re.search(r"^TOP40_ROWS = (\[.*?\])\n", src, re.S | re.M).group(1))
SERIES = [("AdamW", ("plan_adam_eps1e17_", "sm_adamw_eps1e17_")),
          ("Muon", ("plan_muon_eps1e17_", "sm_muon_eps1e17_"))]
PREFER = ("plan_muon_eps1e17_4k_bs256_lr2e-4",)

rows = list(csv.DictReader(open(ROOT + "/experiments.csv")))
alive = set()
if len(sys.argv) > 1:
    alive = {l.strip() for l in open(sys.argv[1]) if l.strip()}


def pick(prefixes, suffix):
    for r in sorted(rows, key=lambda r: r["run_id"] not in PREFER):
        if r["run_id"].startswith(prefixes) and r["run_id"].endswith(suffix):
            return r
    return None


def status(run_id, method):
    """PRESENT / RUNNING / PARTIAL / MISSING for one (row, method) point."""
    r = next((x for x in rows if x["run_id"] == run_id), None)
    if r and (r.get(f"filter_{method}_delta") or "").strip():
        return "present", ""
    if not r:
        return "missing", "no row in experiments.csv"
    d = os.path.join(E, run_id)
    running = [c for c in alive if c.startswith(d + os.sep)]
    if running:
        return "running", os.path.basename(running[0])[:34] + (f" (+{len(running)-1})" if len(running) > 1 else "")
    # scores present but no filter yet -> the filter is the missing step
    if method == "ekfac" and os.path.isfile(os.path.join(d, "ekfac_scores/scores/info.json")):
        return "missing", "ekfac scores done, filter not run"
    if method == "magic":
        pq = os.path.join(d, "per_query")
        n = len(glob.glob(pq + "/q*.pt")) if os.path.isdir(pq) else 0
        for sub in ("magic_scores", "magic_scores_ssd2", "magic_scores_only"):
            p = os.path.join(d, sub, "per_query")
            if os.path.isdir(p):
                n = max(n, len(glob.glob(p + "/q*.pt")))
        if n:
            return "partial", f"{n}/20 queries scored"
        return "missing", "no magic scores"
    return "missing", ""


FIGS = []
FIGS.append(("filter_scaling.png  (left: top 1%)",
             [(f"{n//1000}k", pick(SERIES[0][1], f"{n//1000}k_bs256"), "ekfac") for n in NS]))
FIGS.append(("filter_scaling.png  (right: top 40)",
             [(f"{n//1000}k", next((x for x in rows if x["run_id"] == rid), None), "top40")
              for n, rid in TOP40]))
FIGS.append(("filter_scaling_appendix.png  (AdamW vs Muon)",
             [(f"{name} {n//1000}k", pick(pre, f"{n//1000}k_bs256"), "ekfac")
              for name, pre in SERIES for n in NS]))
FIGS.append(("filter_batch_appendix.png  (batch sweep at 16k)",
             [(f"{name} bs{b}", pick(pre, f"16k_bs{b}"), "ekfac")
              for name, pre in SERIES for b in BATCHES]))
FIGS.append(("filter_method_appendix.png  (EK-FAC vs MAGIC)",
             [(f"{m.upper()} {n//1000}k", pick(SERIES[0][1], f"{n//1000}k_bs256"), m)
              for m in ("ekfac", "magic") for n in NS]))

for title, points in FIGS:
    holes = []
    n_present = 0
    for label, r, method in points:
        if r is None:
            holes.append((label, "missing", "no row"))
            continue
        if method == "top40":
            p = os.path.join(E, r["run_id"], "filter_top40_ekfac", "filter_summary.csv")
            if os.path.isfile(p):
                n_present += 1
            else:
                holes.append((label, "missing", "no top-40 summary"))
            continue
        st, note = status(r["run_id"], method)
        if st == "present":
            n_present += 1
        else:
            holes.append((label, st, note or r["run_id"]))
    print(f"  {title}: {n_present}/{len(points)} points")
    for label, st, note in holes:
        print(f"     {label:16s} {st.upper():8s} {note}")
