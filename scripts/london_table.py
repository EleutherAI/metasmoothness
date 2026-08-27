#!/usr/bin/env python3
"""Render london.csv as a README section.

    python scripts/london_table.py            # print
    python scripts/london_table.py --readme   # splice into README.md

London is the distribution-shift corpus. It carries metasmoothness only -- no
bank, so no LDS and no filter delta. Regenerate after rebuilding london.csv; do
not hand-edit the block in README.md.
"""
import argparse, csv, os, pathlib

ap = argparse.ArgumentParser()
ap.add_argument("--readme", action="store_true")
args = ap.parse_args()

HERE = pathlib.Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE.parent / "london.csv")))

def n(r):
    try:
        return int(float(r["n_docs"]))
    except (TypeError, ValueError):
        return 0

out = ["metasmoothness on london (distribution-shift corpus), 2 epochs",
       "", "%-24s %-6s %8s %6s %8s %10s" % ("run_id", "opt", "N", "bs", "steps", "ms")]
for r in sorted(rows, key=lambda r: (n(r), r.get("batch_size", ""), r.get("optimizer", ""))):
    ms = (r.get("metasmoothness") or "").strip()
    out.append("%-24s %-6s %8d %6s %8s %10s"
               % (r["run_id"][:24], r.get("optimizer", ""), n(r),
                  r.get("batch_size", ""), r.get("n_steps", ""), ms or "pending"))
block = "\n".join(out)
print(block)

if args.readme:
    p = HERE.parent / "README.md"
    lines = p.read_text().split("\n")
    HDR = "## Alternative corpus: london"
    if HDR in lines:
        i = lines.index(HDR)
        j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith("## ")), len(lines))
        lines = lines[:i] + lines[j:]
    try:
        anchor = lines.index("## Results")
    except ValueError:
        anchor = next((k for k, l in enumerate(lines) if l.startswith("## ")), len(lines))
    new = [HDR, "",
           "No bank was built for these rows, so there is no LDS and no filter delta.",
           "ms only. Regenerate with `python scripts/london_table.py --readme`.", "",
           "```", block, "```", "",
           "At bs256 ms holds up to 32k and then collapses at 64k (0.674 adamw, 0.447 muon)",
           "where smollm2 stays near 0.99. That collapse is confounded with learning rate:",
           "the 64k rows ran at lr 1.6e-3 and re-running at 8e-4 recovers most of it",
           "(+0.26 adamw, +0.42 muon), leaving a smaller genuine N effect at fixed lr",
           "(-0.041 adamw, -0.093 muon from 32k to 64k).", ""]
    lines = lines[:anchor] + new + lines[anchor:]
    p.write_text("\n".join(lines))
    print("\n  spliced into README.md")
