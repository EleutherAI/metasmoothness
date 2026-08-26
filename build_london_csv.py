#!/usr/bin/env python3
"""Build london.csv -- the distribution-shift arm, kept out of the paper grid.

Lucia's ruling 2026-08-26: london gets its own table. build_experiments_csv.py
asserts `dataset == "smollm2"` for every admitted row, which is a deliberate
statement about what the paper grid contains, so the london arm lives here
instead of being forced through that assert.

Why the arm exists: our fine-tuning ms sits at 0.98-0.99 almost everywhere,
which is higher than the wikitext experience, and the suspicion is that smollm2
is simply too close to GPT-2 pre-training for the probe to be stressed.
london-llm-1800 is pre-1931 text -- the same task at a real distribution shift.
The comparison of interest is ms and LDS against the smollm2 row at MATCHED N and
batch size, never the absolute loss, which is not comparable across two different
held-out sets.

Results are recorded here the same way build_experiments_csv.py does it: as a
dict keyed on run_id, applied over rows defined below, with a guard that refuses
to silently drop a measurement whose row does not exist. That guard exists
because the identical bug has now bitten both other generators.

    python build_london_csv.py        # writes london.csv
"""
import csv
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "london.csv")

COLS = ["run_id", "optimizer", "n_docs", "batch_size", "n_epochs", "n_steps",
        "lr", "metasmoothness", "fd_step", "heldout_loss", "magic_lds",
        "ekfac_lds", "hardware", "world_size", "status", "run_dir", "notes"]

RUNS = "/mnt/ssd-2/lucia/paper_runs/experiments"

# lrs are the measured INTERIOR winners from tuning.csv, evaluated on
# london_heldout_4k (4000 docs x 512 tok, source rows 40000+, verified
# zero-overlap against london_128k). Scoring a london model against the smollm2
# held-out set would select whichever lr best fits the wrong distribution.
#
#   16k bs256  8e-4    32k bs256  8e-4    64k/128k bs256  1.6e-3    16k bs16  2e-4
#
# The lr optimum is flat at 8e-4 from 16k to 32k and then climbs to 1.6e-3 by
# 64k, so it is not transferable from the smollm2 grid, which is why these
# sweeps were worth running.
TUNED = {
    (16000, 256): 8e-4, (32000, 256): 8e-4,
    (64000, 256): 1.6e-3, (128000, 256): 1.6e-3,
    (16000, 16): 2e-4,
}

# heldout CE of the winning lr in each sweep group, for reference alongside ms.
HELDOUT = {
    "london16k_bs256_adamw": 3.8397, "london16k_bs256_muon": 3.8394,
    "london32k_bs256_adamw": 3.7873, "london32k_bs256_muon": 3.7842,
    "london64k_bs256_adamw": 3.6099, "london64k_bs256_muon": 3.5993,
    "london128k_bs256_adamw": 3.4992, "london128k_bs256_muon": 3.4845,
}

# Measured ms. Only 16k exists so far; 32k/64k/128k are probing now.
#
# THE HEADLINE, and the reason this arm was worth the compute: at 16k bs256 the
# two optimizers separate by 0.13, which nothing in the smollm2 grid comes close
# to -- there adamw and muon sit at 0.9930 and 0.9963 at the same setting. So the
# corpus, not the optimizer, is what was hiding the difference.
MS = {
    "london16k_bs256_adamw": 0.9867,
    "london16k_bs256_muon": 0.8547,
    # bs16 measured 2026-08-26. These invert the bs256 ordering, which is the
    # most surprising thing the arm has produced so far:
    #
    #                  adamw    muon
    #   london  bs16   0.9058   0.9640
    #   london  bs256  0.9867   0.8547
    #   smollm2 bs16   0.9133   0.9939
    #   smollm2 bs256  0.9930   0.9963
    #
    # On smollm2 both optimizers prefer the LARGER batch (adamw 0.9133 -> 0.9930,
    # muon 0.9939 -> 0.9963). On london adamw does the same (0.9058 -> 0.9867)
    # but muon goes the OTHER WAY, 0.9640 down to 0.8547.
    #
    # So the 0.13 optimizer gap at bs256 is not a plain corpus effect. At bs16
    # london and smollm2 look alike for both optimizers; the gap only opens at
    # large batch, and only for muon. That is a corpus x optimizer x batch
    # interaction, not "london is harder".
    #
    # Treat london16k_bs256_muon at 0.8547 as the value to re-check first: it is
    # the single outlier carrying the whole story, and it is one measurement.
    "london16k_bs16_adamw": 0.9058,
    "london16k_bs16_muon": 0.9640,
    # 32k bs256, measured 2026-08-26. These undercut the 16k reading badly.
    #
    #   london  16k bs256   adamw 0.9867   muon 0.8547   gap 0.130
    #   london  32k bs256   adamw 0.9732   muon 0.9536   gap 0.020
    #   smollm2 32k bs256   adamw 0.9937   muon 0.9948   gap 0.001
    #
    # The 0.13 optimizer gap does NOT survive doubling the corpus. muon reads
    # 0.9536 at 32k, not ~0.85, so london16k_bs256_muon = 0.8547 looks like an
    # outlier rather than the start of a trend. The seed probes now running will
    # say whether it is direction noise.
    #
    # What DOES survive is smaller and cleaner: london sits below smollm2 for
    # BOTH optimizers at 32k, by 0.021 (adamw) and 0.041 (muon). A modest corpus
    # effect in the same direction for both, which is a much more ordinary claim
    # than the one the 16k cell suggested.
    "london32k_bs256_adamw": 0.9732,
    "london32k_bs256_muon": 0.9536,
}

rows = []
for n in (16000, 32000, 64000, 128000):
    for opt in ("adamw", "muon"):
        rid = "london%dk_bs256_%s" % (n // 1000, opt)
        steps = (n * 2) // 256
        rows.append({
            "run_id": rid, "optimizer": opt, "n_docs": n, "batch_size": 256,
            "n_epochs": 2, "n_steps": steps, "lr": TUNED[(n, 256)],
            "fd_step": 0.1, "hardware": "", "world_size": 2,
            "run_dir": "%s/%s" % (RUNS, rid),
            "notes": "Distribution-shift arm, pre-1931 corpus. Compare ms with "
                     "the smollm2 row at N=%d bs256; the losses are not "
                     "comparable across held-out sets." % n,
        })
# bs16 is the sharp probe: batch moves ms hardest on smollm2 (0.9930 at bs256
# down to 0.9133 at bs16), so it is where a corpus effect should show most.
for opt in ("adamw", "muon"):
    rid = "london16k_bs16_%s" % opt
    rows.append({
        "run_id": rid, "optimizer": opt, "n_docs": 16000, "batch_size": 16,
        "n_epochs": 2, "n_steps": 2000, "lr": TUNED[(16000, 16)],
        "fd_step": 0.1, "hardware": "", "world_size": 2,
        "run_dir": "%s/%s" % (RUNS, rid),
        "notes": "Distribution-shift arm at bs16 (2000 steps). Batch is the axis "
                 "that moves ms hardest on smollm2, so a corpus effect should be "
                 "most visible here.",
    })

by_id = {r["run_id"]: r for r in rows}

# Apply measurements, and refuse to drop one whose row does not exist. The same
# silent-drop bug has now been found in build_experiments_csv.py (BANK_RESULTS)
# and build_tuning_csv.py (LONDON_HELDOUT); it is not getting a third chance.
for table, field in ((MS, "metasmoothness"), (HELDOUT, "heldout_loss")):
    orphans = sorted(set(table) - set(by_id))
    if orphans:
        raise SystemExit(
            "%d %s result(s) name a run with no row, so they would be silently "
            "dropped:\n  %s" % (len(orphans), field, "\n  ".join(orphans)))
    for rid, val in table.items():
        by_id[rid][field] = val

for r in rows:
    rd = r["run_dir"]
    done = os.path.isfile(os.path.join(rd, "ms", "metasmoothness.json"))
    r["status"] = "done" if r.get("metasmoothness") else ("running" if done or os.path.isdir(rd) else "planned")
    for c in COLS:
        r.setdefault(c, "")

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    for r in sorted(rows, key=lambda r: (r["n_docs"], r["batch_size"], r["optimizer"])):
        w.writerow(r)

n_ms = sum(1 for r in rows if r["metasmoothness"] != "")
print("wrote %s: %d rows (%d with ms, %d with heldout)"
      % (OUT, len(rows), n_ms, sum(1 for r in rows if r["heldout_loss"] != "")))
for r in sorted(rows, key=lambda r: (r["n_docs"], r["batch_size"], r["optimizer"])):
    print("  %-26s N=%-7s bs=%-4s lr=%-8s ms=%-8s heldout=%s"
          % (r["run_id"], r["n_docs"], r["batch_size"], r["lr"],
             r["metasmoothness"] or "-", r["heldout_loss"] or "-"))
