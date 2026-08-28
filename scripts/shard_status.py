"""Per-shard completion state for a sharded filter run.

Three states matter and they are not the same:
  SUMMARY   the shard finished and wrote filter_summary.csv -- ready to merge
  per-query retrains done but no summary. This is the state every muon 128k shard
            reached because its bank had no validation_merged.csv. Recoverable
            with recover_shard_summary.py, no GPU time.
  empty     still retraining
"""
import glob
import os
import sys

E = "/mnt/ssd-2/lucia/paper_runs/experiments"
TARGETS = [("plan_adam_eps1e17_128k_bs256", "filter_top40_ekfac"),
           ("plan_adam_eps1e17_256k_bs256", "filter_proponents_ekfac")]

for run, pre in TARGETS:
    ds = sorted(glob.glob(os.path.join(E, run, pre + "_q*_*", "")))
    n_sum = n_pq = 0
    lines = []
    for d in ds:
        name = os.path.basename(os.path.dirname(d))
        if os.path.isfile(os.path.join(d, "filter_summary.csv")):
            st = "SUMMARY"; n_sum += 1
        elif os.path.isfile(os.path.join(d, "filter_proponents.csv")):
            st = "per-query (recoverable)"; n_pq += 1
        else:
            st = "empty"
        lines.append("     %-34s %s" % (name, st))
    bank = os.path.join(E, run, "bank_top40" if "top40" in pre else "bank_from_filter",
                        "validation_merged.csv")
    print("  %s / %s: %d/%d summaries, %d per-query-only" % (run, pre, n_sum, len(ds), n_pq))
    print("     bank merged: %s" % ("yes" if os.path.isfile(bank) else "NO -- summaries will fail"))
    print("\n".join(lines))
