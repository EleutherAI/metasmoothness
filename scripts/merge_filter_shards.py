#!/usr/bin/env python3
"""Merge per-query filter shards into the canonical filter_proponents_<src> dir.

    python scripts/merge_filter_shards.py <run_id> [--source ekfac] [--force]

scripts/filter_deltas.py discovers summaries with

    glob("<root>/*/filter_*_*/filter_summary.csv")
    src = basename(dir).split("_")[-1]

so a shard directory named filter_proponents_ekfac_q0_7 is keyed as src="7",
not "ekfac", and its delta never registers. It also takes only the FIRST dir per
(run, src), so four shards would collapse to one even if they keyed correctly.

This writes the union of the shards to filter_proponents_<src>/filter_summary.csv,
which is the path the discovery already expects.

Query indices in a shard's summary are LOCAL (0..k-1); shard q<a>_<b> holds global
queries a..b-1, so local i is rewritten to a+i. Without that every shard claims
query 0 and the merged file has duplicates.

Each shard carries its OWN 3 random controls (num_subsets=0 is rejected without a
bank), so random_mean/random_sd stay per-shard rather than pooled. The per-query
delta is unaffected; only the shared variance across queries differs from an
unsharded run, which is why a sharded delta is not bit-comparable with one.
"""
import argparse, csv, glob, os, re, sys

ap = argparse.ArgumentParser()
ap.add_argument("run_id")
ap.add_argument("--source", default="ekfac")
ap.add_argument("--prefix", default="filter_proponents",
                help="shard dir prefix; use filter_top40 for the fixed-40 runs")
ap.add_argument("--force", action="store_true")
args = ap.parse_args()

ROOTS = ["/mnt/ssd-2/lucia/paper_runs/experiments",
         "/mnt/ssd-1/lucia/paper_runs/experiments"]
root = next((r for r in ROOTS if os.path.isdir(os.path.join(r, args.run_id))), None)
if root is None:
    sys.exit("run dir not found: %s" % args.run_id)
run = os.path.join(root, args.run_id)

pat = re.compile(r"%s_%s_q(\d+)_(\d+)$" % (re.escape(args.prefix), re.escape(args.source)))
shards = []
for d in sorted(glob.glob(os.path.join(run, "%s_%s_q*_*" % (args.prefix, args.source)))):
    m = pat.search(d)
    s = os.path.join(d, "filter_summary.csv")
    if m and os.path.isfile(s):
        shards.append((int(m.group(1)), int(m.group(2)), s))
if not shards:
    sys.exit("no finished shards for %s (%s)" % (args.run_id, args.source))

rows, fields, covered = [], None, set()
for a, b, s in sorted(shards):
    with open(s, newline="") as f:
        rd = csv.DictReader(f)
        fields = fields or rd.fieldnames
        got = 0
        for r in rd:
            local = int(float(r["query"]))
            g = a + local
            if g >= b:
                sys.exit("shard q%d_%d has local query %d out of range" % (a, b, local))
            if g in covered:
                sys.exit("duplicate global query %d -- overlapping shards" % g)
            covered.add(g)
            r["query"] = g
            rows.append(r)
            got += 1
    print("  q%d_%d: %d/%d queries" % (a, b, got, b - a))

rows.sort(key=lambda r: int(r["query"]))
missing = sorted(set(range(max(covered) + 1)) - covered)
if missing:
    print("  INCOMPLETE: %d query(ies) absent: %s" % (len(missing), missing))
    if not args.force:
        sys.exit("refusing to write a partial merge (use --force)")

out_dir = os.path.join(run, "%s_%s" % (args.prefix, args.source))
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "filter_summary.csv")
if os.path.exists(out) and not args.force:
    sys.exit("refusing to overwrite %s (use --force)" % out)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print("  wrote %s: %d queries from %d shards" % (out, len(rows), len(shards)))
