#!/usr/bin/env python3
"""Regenerate every paper figure into figures/ and check the set is exact.

    python scripts/make_figures.py
    python scripts/make_figures.py --stat median   # figures_median/, tables_median/

Runs each plot script in turn (all write PDF), then requires figures/ to hold
exactly EXPECTED_FIGURES -- a missing figure or a stray .png/.pdf fails the
build -- and finishes with scripts/figure_audit.py, which re-checks the file set
and reports the data points each figure still lacks.

--stat median (or QLD_STAT=median in the environment) rebuilds the whole set
with the per-query median in place of the mean (absolute_losses.STAT; the
EK-FAC/MAGIC top-1% points are then recomputed from the merged summaries, since
experiments.csv holds only means) into figures_median/ and tables_median/,
leaving figures/ and tables/ untouched. The producers and the audit take their
output dirs from FIGURES_DIR / TABLES_DIR, which this script exports.

Not generated here, on purpose: filter_batch_appendix and filter_scaling_appendix
(folded into filter_muon_appendix), filter_scaling_law (scaling_law_plot.py) and
ood_mean_scaling_diagnostic (plot_ood_mean_scaling.py). Those scripts still run
by hand.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAT = os.environ.get("QLD_STAT", "mean")
_suffix = lambda stat: "" if stat == "mean" else f"_{stat}"
FIGURES = os.environ.get("FIGURES_DIR") or os.path.join(ROOT, "figures" + _suffix(STAT))
TABLES = os.environ.get("TABLES_DIR") or os.path.join(ROOT, "tables" + _suffix(STAT))

# script -> figures it writes: each QLD figure beside its *_relative companion
# (the same difference as a percent of the matched control's loss,
# absolute_losses.relative_point). The *_absolute companions are switched off
# (absolute_losses.WRITE_ABSOLUTE) and a share-of-learning normalization was
# tried and rejected (notes/normalized_qld.md); both code paths are kept but
# produce nothing here.
PRODUCERS = [
    ("scaling_plot_mpl.py", ["filter_scaling", "filter_muon_appendix", "filter_muon_appendix_relative",
                             "filter_method_appendix",
                             "filter_method_appendix_relative", "filter_variants_appendix",
                             "filter_variants_appendix_relative"]),
    ("plot_filter_absolute_losses.py", ["filter_scaling_relative"]),
    ("heldout_plot.py", ["filter_heldout",
                         "filter_heldout_relative"]),
    ("qwen_scaling_plot.py", ["filter_scaling_qwen",
                              "filter_scaling_qwen_relative"]),
    ("qwen15b_heldout_trend.py", ["qwen15b_heldout_trend"]),
    ("filter_vs_lds_plot.py", ["filter_vs_lds"]),
    ("lds_vs_size_plot.py", ["lds_vs_size"]),  # LDS vs training-set size, MAGIC and EK-FAC panels (2026-09-06)
    # writes tables/qld_lds_adamw_muon.tex, no figure (replaces the corpus-scaling
    # panel formerly in filter_muon_appendix)
    ("qld_lds_table.py", []),
    # first four points of filter_method_appendix as a table (tables/qld_methods_4k_32k.tex)
    ("qld_methods_table.py", []),
    # every number quoted in the Results section (tables/results_analysis.md)
    ("results_analysis.py", []),
]
WRITE_RELATIVE = os.environ.get("QLD_RELATIVE", "0") == "1"  # mirrors absolute_losses.WRITE_RELATIVE
if not WRITE_RELATIVE:  # the *_relative companions are gated off in every producer
    PRODUCERS = [(s, [n for n in names if not n.endswith("_relative")]) for s, names in PRODUCERS]
EXPECTED_FIGURES = sorted(f"{name}.pdf" for _, names in PRODUCERS for name in names)


def check_figure_set(verbose=True):
    """(missing, stray) among the top-level .pdf/.png files of figures/."""
    present = {f for f in os.listdir(FIGURES)
               if os.path.isfile(os.path.join(FIGURES, f)) and f.endswith((".pdf", ".png"))}
    missing = sorted(set(EXPECTED_FIGURES) - present)
    stray = sorted(present - set(EXPECTED_FIGURES))
    if verbose:
        for f in EXPECTED_FIGURES:
            size = os.path.getsize(os.path.join(FIGURES, f)) if f in present else 0
            print(f"  {'ok     ' if f in present else 'MISSING'} {f:44s} {size:>8d} B")
        for f in stray:
            print(f"  STRAY   {f}  (not produced by make_figures.py; remove it)")
    return missing, stray


def main():
    global STAT, FIGURES, TABLES
    ap = argparse.ArgumentParser(description="Regenerate every paper figure.")
    ap.add_argument("--stat", choices=("mean", "median"), default=None,
                    help="per-query aggregate (default: QLD_STAT or mean); median "
                         "writes to figures_median/ and tables_median/")
    ap.add_argument("--relative", action="store_true", help="also write the *_relative companions (sets QLD_RELATIVE=1)")
    a = ap.parse_args()
    if a.relative:
        os.environ["QLD_RELATIVE"] = "1"
    if a.stat and a.stat != STAT:
        STAT = a.stat
        FIGURES = os.path.join(ROOT, "figures" + _suffix(STAT))
        TABLES = os.path.join(ROOT, "tables" + _suffix(STAT))
    os.environ.update(QLD_STAT=STAT, FIGURES_DIR=FIGURES, TABLES_DIR=TABLES)
    os.makedirs(FIGURES, exist_ok=True)
    for script, _ in PRODUCERS:
        print(f"== {script}", flush=True)
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", script)],
                       cwd=ROOT, check=True)
    print("== figure set")
    missing, stray = check_figure_set()
    print("== figure_audit.py", flush=True)
    audit = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "figure_audit.py")],
                           cwd=ROOT)
    if missing or stray or audit.returncode:
        sys.exit(f"FAILED: missing={missing} stray={stray} audit_rc={audit.returncode}")
    print(f"OK: {len(EXPECTED_FIGURES)} figures in {FIGURES}")


if __name__ == "__main__":
    main()
