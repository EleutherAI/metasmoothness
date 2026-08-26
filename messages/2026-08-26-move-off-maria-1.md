# maria-1 is off-limits as of 2026-08-26

Lucia: "We need to move off maria-1 now." Do not launch anything there.

## What this changes

**A100 capacity drops to three nodes**: marisa-0, lotus-0, shivam2-0. maria-1 was
A100, so under D17 (GPU type is part of run identity) every A100-bound comparison
now competes for those three. If an A100 pair is not free, the correct move is to
run BOTH arms of a comparison on A40 rather than split one arm onto A100 -- a
hardware-homogeneous comparison on the "wrong" type is sound; a split one is not.

`gpu_free.sh` reported 0-6 free on maria-1 at the time of the handover, but the
node had eight live bergson processes and memory on five GPUs. None were ours
(`hung_check.py` reported zero live runs of ours there). Those are another
tenant's, and the free reading reflects our view, not the node's.

## Claims released

Seven claims were still held by maria-1 and are now released:

    london16k_bs256_muon__ms                      run FINISHED (ms 0.8547)
    plan_adam_eps1e17_64k_bs256__ekfacscore       run FINISHED (scores complete)
    tune_adamw_london16k_bs16_lr0.0004__tune
    tune_adamw_london32k_bs256_lr0.0016__tune
    tune_muon_london64k_bs256_lr0.0004__tune
    tune_muon_london64k_bs256_lr0.0008__tune
    tune_muon_london64k_bs256_lr0.0016__tune

The first two are the D18 failure again, and they are the reason that decision
exists: both runs had **completed** and neither released its claim, so both rows
read as owned-and-busy while their outputs sat finished on disk. Nothing was
lost, but nothing could have picked those rows up either.

No result is missing from either. `london16k_bs256_muon` ms 0.8547 is already
recorded. The 64k_bs256 EK-FAC scores are an input rather than a result -- that
row has no bank yet (0 subsets, no validation.csv), so there is no LDS to compute
against them until one is built.

The four london ones belong to the 32k/64k hang family in
`2026-08-26-error-analysis-log.md`; releasing the claim does not make those runs
work, it only stops them blocking the row.

## What moved

Two ms probes were running on maria-1 and were relocated rather than left:

    plan_adam_eps1e17_256k_bs32   ms   -> bellflower-0  6,7  (A40)
    plan_muon_eps1e17_256k_bs32   ms   -> allium-0      2,3  (A40)
    plan_muon_eps1e17_128k_bs32   ms   -> shivam2-0     6,7  (A100)

They were killed at 1-2% rather than allowed to ride out: ms is THREE trainings,
and at 16000 steps each that is ~6h45 for adam and ~8h51 for muon, not the few
hours that would have justified leaving them. Restarting cost about four minutes
of work.

Both 256k arms went to A40 together. That is deliberate under D17 -- with maria-1
gone there was one free A100 pair, and splitting the comparison across types
would have confounded it. A homogeneous A40 comparison is sound; a split one is
not. The 128k pair stays A100 because its adam arm was already running there.
