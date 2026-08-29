"""Which figure each result feeds, so results can be named by their purpose.

Derives membership from the plot scripts' own axis constants rather than a
hand-kept list, so it cannot drift from what is drawn.
"""
import csv
import re
import sys

ROOT = "/mnt/ssd-2/lucia/metasmoothness"
src = open(ROOT + "/scripts/scaling_plot_mpl.py").read()
NS = eval(re.search(r"^NS = (\[[^\]]*\])", src, re.M).group(1))
BATCHES = eval(re.search(r"^BATCHES = (\[[^\]]*\])", src, re.M).group(1))
TOP40 = eval(re.search(r"^TOP40_ROWS = (\[.*?\])\n", src, re.S | re.M).group(1))
VARIANTS = eval(re.search(r"^VARIANT_ROWS = (\[.*?\])\n", src, re.S | re.M).group(1))

rows = list(csv.DictReader(open(ROOT + "/experiments.csv")))
PREFER = ("plan_muon_eps1e17_4k_bs256_lr2e-4",)
ADAM = ("plan_adam_eps1e17_", "sm_adamw_eps1e17_")
MUON = ("plan_muon_eps1e17_", "sm_muon_eps1e17_")


def pick(pre, suf):
    for r in sorted(rows, key=lambda r: r["run_id"] not in PREFER):
        if r["run_id"].startswith(pre) and r["run_id"].endswith(suf):
            return r["run_id"]
    return None


membership = {}


def add(run, fig, col):
    if run:
        membership.setdefault(run, set()).add((fig, col))


for n in NS:
    add(pick(ADAM, f"{n//1000}k_bs256"), "filter_scaling (1%)", "ekfac delta")
    add(pick(ADAM, f"{n//1000}k_bs256"), "filter_method_appendix", "ekfac delta")
    add(pick(ADAM, f"{n//1000}k_bs256"), "filter_method_appendix", "magic delta")
    for name, pre in (("AdamW", ADAM), ("Muon", MUON)):
        add(pick(pre, f"{n//1000}k_bs256"), "filter_scaling_appendix", "ekfac delta")
for _, rid in TOP40:
    add(rid, "filter_scaling (top-40)", "top-40 delta")
for b in BATCHES:
    for pre in (ADAM, MUON):
        add(pick(pre, f"16k_bs{b}"), "filter_batch_appendix", "ekfac delta")
for _, rid in VARIANTS:
    add(rid, "filter_variants_appendix", "ekfac delta")
# The LDS-vs-delta scatter takes every row carrying both halves, either scorer.
for r in rows:
    for sc in ("ekfac", "magic"):
        if (r.get(f"{sc}_lds") or "").strip() and (r.get(f"filter_{sc}_delta") or "").strip():
            add(r["run_id"], "filter_vs_lds", f"{sc} lds+delta")

want = sys.argv[1] if len(sys.argv) > 1 else None
print("  rows carrying MAGIC results, and what they feed:")
for r in sorted(rows, key=lambda r: int((r.get("steps") or "0") or 0)):
    run = r["run_id"]
    if not ((r.get("magic_lds") or "").strip() or (r.get("filter_magic_delta") or "").strip()):
        continue
    figs = sorted({f for f, c in membership.get(run, set()) if "magic" in c or f.startswith("filter_method")})
    print("    %-34s steps=%-5s %s" % (run, r.get("steps", ""),
                                       ", ".join(figs) if figs else "NON-FIGURE DATA"))
print()
print("  in-progress rows:")
for run in ("plan_muon_eps1e17_32k_bs32", "plan_muon_eps1e17_64k_bs32",
            "plan_adam_eps1e17_64k_bs32", "plan_adam_eps1e17_128k_bs256",
            "plan_adam_eps1e17_256k_bs256"):
    figs = sorted({f for f, c in membership.get(run, set())})
    print("    %-34s %s" % (run, ", ".join(figs) if figs else "NON-FIGURE DATA"))
