#!/usr/bin/env python3
"""Print the metasmoothness results grid from experiments.csv as an ASCII table.

    python scripts/results_table.py                 # print to stdout
    python scripts/results_table.py -o results.txt  # also write to a file (scp it home)
    python scripts/results_table.py --all           # include rows with no results yet

Re-run any time experiments.csv is regenerated. Stdlib only.
"""
import argparse, csv, sys
from pathlib import Path

COLS = [  # (header, csv column, formatter)
    ("run",            "run_id",         str),
    ("optimizer",      "optimizer",      str),
    ("lr",             "lr",             lambda v: f"{float(v):.0e}"),
    ("N docs",         "n_docs",         lambda v: f"{int(float(v)):,}"),
    ("bs",             "batch_size",     str),
    ("N epochs",       "num_epochs",     str),
    ("N steps",        "steps",          str),
    ("metasmoothness", "metasmoothness", lambda v: f"{float(v):.4f}"),
    ("EK-FAC LDS",     "ekfac_lds",      lambda v: f"{float(v):.4f}"),
    ("MAGIC LDS",      "magic_lds",      lambda v: f"{float(v):.4f}"),
    # Tail-filter power, z against the row's own bank as the random control.
    # The raw nat deltas are not comparable across dataset sizes; z is.
    ("MAGIC filt z",   "filter_prop_magic_z", lambda v: f"{float(v):.1f}"),
    ("EK-FAC filt z",  "filter_prop_ekfac_z", lambda v: f"{float(v):.1f}"),
    ("train loss",     "train_loss",     lambda v: f"{float(v):.4f}"),
    ("heldout loss",   "heldout_loss",   lambda v: f"{float(v):.4f}"),
    ("delta L2",       "delta_l2",       lambda v: f"{float(v):.2f}"),
    ("status",         "status",         str),
]
IDX = {c: i for i, (_, c, _) in enumerate(COLS)}  # name -> column index
RESULT_COLS = ("metasmoothness", "ekfac_lds", "magic_lds")

def fmt(r, col, f):
    v = r.get(col, "")
    if v in ("", None):
        return "-"
    try:
        return f(v)
    except ValueError:
        return v

def ci(r, prefix):
    lo, hi = r.get(f"{prefix}_ci_lo", ""), r.get(f"{prefix}_ci_hi", "")
    return f" [{float(lo):.3f},{float(hi):.3f}]" if lo and hi else ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=Path(__file__).resolve().parent.parent / "experiments.csv")
    ap.add_argument("-o", "--out", help="also write the table to this file")
    ap.add_argument("--all", action="store_true", help="include rows with no results yet")
    ap.add_argument("--ci", action="store_true", help="append 95%% CIs to the LDS columns")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv, newline="")))
    if not a.all:
        rows = [r for r in rows if any(r.get(c) for c in RESULT_COLS)]
    rows.sort(key=lambda r: (r["optimizer"], float(r["n_docs"] or 0), float(r["batch_size"] or 0), float(r["num_epochs"] or 0)))

    table = []
    for r in rows:
        line = [fmt(r, c, f) for _, c, f in COLS]
        if a.ci:
            line[IDX["ekfac_lds"]] += ci(r, "ekfac")
            line[IDX["magic_lds"]] += ci(r, "magic")
        table.append(line)

    hdr = [h for h, _, _ in COLS]
    w = [max(len(h), *(len(t[i]) for t in table)) if table else len(h) for i, h in enumerate(hdr)]
    right = {IDX[c] for c in ("n_docs", "batch_size", "num_epochs", "steps", "metasmoothness",
                              "ekfac_lds", "magic_lds", "filter_prop_magic_z",
                              "filter_prop_ekfac_z", "train_loss", "heldout_loss",
                              "lr", "delta_l2")}
    def row(cells): return " | ".join(c.rjust(w[i]) if i in right else c.ljust(w[i]) for i, c in enumerate(cells))
    lines = [row(hdr), "-+-".join("-" * x for x in w)] + [row(t) for t in table]
    n_done = sum(all(r.get(c) for c in RESULT_COLS) for r in rows)
    lines.append(f"\n{len(rows)} rows shown, {n_done} with all three metrics. '-' = not yet measured.")
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")
        print(f"\nwrote {a.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
