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

# Measured ms, against the matched smollm2 row at the same N and batch.
#
# There is NO optimizer headline here, and there nearly was one. A 0.13 gap
# appeared at 16k bs256 and turned out to be a single bad probe direction.
#
# ms perturbs data weights along ONE random direction v. Re-probing
# london16k_bs256_muon at two further directions, with data, lr, optimizer,
# batch, fd_step and world size all identical and only direction_seed changed:
#
#   london16k_bs256_muon    seed 0  0.8547   seed 1  0.9858   seed 2  0.9890
#   london16k_bs256_adamw   seed 0  0.9867   seed 1  0.9827
#
# Two independent directions put muon at ~0.987; adamw moves 0.004 across
# directions. The seed-0 draw disagreed with itself by 0.13. That also withdraws
# the "corpus x optimizer x batch interaction" I read off the bs16 pair -- the
# inversion existed only because of that one cell.
#
# The table as it actually stands:
#
#                  adamw    muon
#   london  bs16   0.9058   0.9640
#   london  bs256  0.9867   0.9858
#   smollm2 bs16   0.9133   0.9939
#   smollm2 bs256  0.9930   0.9964
#
# Both optimizers prefer the larger batch on both corpora. london sits below
# smollm2 everywhere, and against the matched smollm2 row: -0.006 and -0.011 at
# 16k bs256, -0.021 and -0.041 at 32k bs256. A real corpus effect, consistent in
# direction, widening slightly with N -- and far too small to explain why ms sits
# at 0.98-0.99 across the whole grid.
#
# Raw seed values in notes/ms_direction_seed_variance.md.
MS = {
    "london16k_bs256_adamw": 0.9867,
    # SIX directions measured, and this cell is a DISTRIBUTION, not a point:
    #
    #   0.8547  0.9858  0.9890  0.9711  0.8262  0.9659
    #   mean 0.9321  median 0.9685  sd 0.0700  range 0.826-0.989
    #
    # Two of six collapse to ~0.83-0.85; the other four sit at 0.966-0.989. That
    # is not a rare tail, it is about a third of directions, so the seed-0 value
    # was neither a fluke nor representative.
    #
    # Controls, same probe, same everything but the direction:
    #   london  adamw 16k   0.9867 0.9827 0.9861          sd 0.0020
    #   smollm2 muon  16k   0.99636 0.99660 0.99672 0.99462  sd 0.0010
    #
    # So the direction-sensitivity is specific to muon ON LONDON. Recording the
    # mean, because no single direction represents this cell.
    "london16k_bs256_muon": 0.9321,
    "london16k_bs16_adamw": 0.9058,
    "london16k_bs16_muon": 0.9640,
    "london32k_bs256_adamw": 0.9712,
    # 32k muon over THREE directions: 0.9536 0.9623 0.9699, sd 0.0082.
    # That is tight. The 16k cell had sd 0.0721 with two of six directions
    # collapsing; at 32k nothing collapses in three draws. So the bimodality is a
    # 16k phenomenon, not a property of muon-on-london generally, and the 32k
    # values can be read as ordinary measurements.
    # adamw at 32k: 0.9732 0.9692, sd 0.0028.
    "london32k_bs256_muon": 0.9619,
    # 64k, and ms falls off a cliff:
    #
    #   london  16k  adamw 0.9867  muon ~0.986   lr 8e-4
    #   london  32k  adamw 0.9732  muon 0.9536   lr 8e-4
    #   london  64k  adamw 0.7000  muon 0.4434   lr 1.6e-3
    #   smollm2 64k  adamw 0.9876  muon 0.9947   lr 1e-4
    #
    # DO NOT read this as an N effect yet. The 64k rows run at a DIFFERENT
    # LEARNING RATE -- the london sweep chose 1.6e-3 by held-out loss, against
    # 8e-4 at 16k/32k and 1e-4 for smollm2 at the same N. lr is a known ms lever
    # (the scale0.25 row in the paper grid turned out to be an lr effect, not a
    # logit-scale one), so N and lr move together across this ladder and the
    # cliff could be either.
    #
    # london64k_bs256_{adamw,muon}_lr8e-4 are running to separate them: same 64k
    # corpus, same everything, lr held at the 32k value. If ms recovers toward
    # 0.97 the cliff is the learning rate; if it stays near 0.70/0.44 it is the
    # corpus at scale.
    #
    # Note this cuts both ways for the tuning. 1.6e-3 is the right lr by held-out
    # loss and apparently a terrible one for ms, which means "tune on loss, then
    # measure ms" can select configurations that are bad for the thing being
    # studied.
    "london64k_bs256_adamw": 0.7000,
    "london64k_bs256_muon": 0.4434,
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
