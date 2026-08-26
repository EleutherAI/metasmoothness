"""Merge a sharded bank's validation slices into validation_merged.csv.

ekfac_lds.py, magic_lds.py, ekfac_paired.py, paired_diff.py and ci_vs_queries.py
all look for `validation_merged.csv` and fall back to `validation.csv` when it is
absent. On a sharded bank `validation.csv` holds only the PRE-SHARD PREFIX -- the
subsets the base build finished before shards took over -- so the fallback
quietly evaluates on a fraction of the bank.

No merged file existed anywhere on disk, so that fallback was the only path ever
taken. plan_adam_eps1e17_32k_bs32 had 91 retrained subsets and scored LDS on 21.

The estimate was not biased by this, and the scripts do print the n_subsets they
used, so nothing recorded is wrong. It was simply built on a quarter of the
evidence that exists, with an interval to match.

Duplicates: a base build that overran its stop can write subsets a shard also
covered. Shard copies win, and any disagreement above --tol is reported rather
than averaged away, because a real disagreement means the two runs were not
bit-comparable and that is worth knowing.

    python merge_bank.py <bank_dir> [--tol 1e-5] [--force]
"""
import argparse
import csv
import re
import sys
from pathlib import Path

AP = argparse.ArgumentParser()
AP.add_argument("bank", type=Path)
AP.add_argument("--tol", type=float, default=1e-5)
AP.add_argument("--force", action="store_true",
                help="overwrite an existing validation_merged.csv")
args = AP.parse_args()

bank = args.bank
out = bank / "validation_merged.csv"
if out.exists() and not args.force:
    sys.exit("refusing: %s exists (use --force)" % out)

base = bank / "validation.csv"
shards = sorted(p for p in bank.glob("validation_*.csv")
                if re.fullmatch(r"validation_\d+_\d+\.csv", p.name))
if not shards:
    sys.exit("no shard files in %s -- nothing to merge" % bank)

fields = None
rows = {}          # (subset, query) -> row
origin = {}        # (subset, query) -> filename
clashes = []

# Base first so shards overwrite it.
for path in ([base] if base.is_file() else []) + shards:
    with open(path) as f:
        r = csv.DictReader(f)
        if fields is None:
            fields = r.fieldnames
        for row in r:
            key = (int(row["subset"]), int(row["query"]))
            if key in rows and origin[key] != path.name:
                a, b = rows[key], row
                for col in ("diff", "score_sum"):
                    if col in a and col in b:
                        try:
                            d = abs(float(a[col]) - float(b[col]))
                        except (TypeError, ValueError):
                            continue
                        if d > args.tol:
                            clashes.append((key, col, origin[key], path.name, d))
            rows[key] = row
            origin[key] = path.name

subsets = sorted({k[0] for k in rows})
queries = sorted({k[1] for k in rows})

with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for key in sorted(rows):
        w.writerow(rows[key])

n_base = sum(1 for v in origin.values() if v == "validation.csv")
print("merged %d rows: %d subsets x %d queries"
      % (len(rows), len(subsets), len(queries)))
print("  sources: validation.csv + %d shard(s)" % len(shards))
print("  rows kept from the pre-shard prefix: %d" % n_base)
if base.is_file():
    with open(base) as f:
        only_base = sum(1 for _ in csv.DictReader(f))
    print("  validation.csv alone would have given %d rows -- the fallback path"
          % only_base)

missing = [s for s in range(max(subsets) + 1) if s not in set(subsets)]
if missing:
    runs, prev = [], None
    for s in missing:
        if prev is not None and s == prev + 1:
            runs[-1][1] = s
        else:
            runs.append([s, s])
        prev = s
    print("  INCOMPLETE: %d subset(s) absent: %s"
          % (len(missing), ", ".join("%d-%d" % (a, b) for a, b in runs)))

if clashes:
    print("  %d DISAGREEMENT(S) above tol=%g between duplicate copies:"
          % (len(clashes), args.tol))
    for (key, col, f1, f2, d) in clashes[:8]:
        print("    subset %d query %d %s: %s vs %s differ by %.3g"
              % (key[0], key[1], col, f1, f2, d))
    print("  shard copies were kept; investigate before trusting this bank")
print("wrote %s" % out)
