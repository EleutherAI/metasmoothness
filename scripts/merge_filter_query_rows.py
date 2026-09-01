
#!/usr/bin/env python3
import argparse, csv, re, shutil
from pathlib import Path
import numpy as np

SHARD_RE = re.compile(r"^(?P<family>.+)_q(?P<start>\d+)_(?P<end>\d+)(?:_.+)?$")


def read_rows(path):
    with path.open(newline="") as f:
        yield from csv.DictReader(f)


def gather_filter_rows(base, family):
    winners = {}
    sources = {}
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        m = SHARD_RE.match(d.name)
        if not m or m.group("family") != family:
            continue
        csv_path = d / "filter_proponents.csv"
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            continue
        start = int(m.group("start"))
        for row in read_rows(csv_path):
            local_q = int(row["query"])
            global_q = start + local_q
            if global_q in winners:
                # Prefer the first row in lexical path order for reproducibility.
                continue
            out = dict(row)
            out["query"] = str(global_q)
            winners[global_q] = out
            sources[global_q] = str(csv_path)
    return winners, sources


def load_random(bank_csv):
    by_subset = {}
    with bank_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            by_subset.setdefault(int(row["subset"]), {})[int(row["query"])] = float(row["diff"])
    if not by_subset:
        raise SystemExit(f"no random rows in {bank_csv}")
    n = max(by_subset) + 1
    qmax = max(q for rows in by_subset.values() for q in rows) + 1
    arr = np.full((n, qmax), np.nan, dtype=float)
    for s, rows in by_subset.items():
        for q, val in rows.items():
            arr[s, q] = val
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path)
    ap.add_argument("family")
    ap.add_argument("--bank", type=Path, default=None)
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    bank = args.bank or args.base / "bank_from_filter" / "validation.csv"
    rows, sources = gather_filter_rows(args.base, args.family)
    missing = [q for q in range(args.queries) if q not in rows]
    print(f"{args.family}: have {len(rows)}/{args.queries}; missing {missing}")
    for q in sorted(rows):
        print(f"q{q}: {sources[q]}")
    if missing or not args.write:
        return 1 if missing else 0

    outdir = args.base / args.family
    outdir.mkdir(exist_ok=True)
    tmpdir = args.base / (args.family + ".merge_tmp")
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir()

    random = load_random(bank)
    filter_csv = tmpdir / "filter_proponents.csv"
    summary_csv = tmpdir / "filter_summary.csv"
    fieldnames = ["query","n_removed","baseline_loss","filtered_loss","loss_change"]
    with filter_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for q in range(args.queries):
            row = {k: rows[q][k] for k in fieldnames}
            row["query"] = q
            w.writerow(row)
    with summary_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query","n_removed","filter_change","random_mean","random_sd","random_n","rank"])
        for q in range(args.queries):
            col = random[:, q]
            col = col[~np.isnan(col)]
            fc = float(rows[q]["loss_change"])
            sd = float(np.std(col, ddof=1)) if len(col) > 1 else float("nan")
            rank = 1 + int((col > fc).sum())
            w.writerow([q, rows[q]["n_removed"], fc, float(np.mean(col)), sd, len(col), rank])
    shutil.copy2(filter_csv, outdir / "filter_proponents.csv")
    shutil.copy2(summary_csv, outdir / "filter_summary.csv")
    print(f"wrote {outdir}")

if __name__ == "__main__":
    raise SystemExit(main())
