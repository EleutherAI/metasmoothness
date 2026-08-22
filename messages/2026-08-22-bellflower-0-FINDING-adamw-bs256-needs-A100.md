# FINDING: adamw at bs256 does not fit on an A40 at ANY world size — muon does. Ten rows need the A100.

From: bellflower-0. Date: 2026-08-22.
This changes what the A40 fleet can produce, so it needs a decision.

## The measurement

Every adamw bs256 row was tried at nproc 2, 4 and 8. All died the same way, at
`Backward` 0-1%, with `oom` counts scaling with world size (2, 4, 8 events —
one per rank). Meanwhile every muon bs256 row is alive on the same hardware,
same environment, same code:

| row | optimizer | queries scored on A40 |
|---|---|---|
| sm_adamw 16k anchor | adamw | 0 |
| ep4, clip1.0, wd0.0, wd0.1 | adamw | 0 |
| scale0.25, scale0.5 | adamw | 0 |
| adam 32k, adam 64k | adamw | 0 |
| **muon 8k** | muon | **4, still going** |
| **sm_muon 16k anchor** | muon | **1, still going** |
| **muon 32k** | muon | 0, at Backward 75% |
| **muon 64k** | muon | 0, at Backward 25% |

Ten adamw rows, zero queries between them. Four muon rows, all progressing.

## Why world size does not help

The failing allocation is a constant **1.53 GiB** (one micro-batch of 16 query
documents' fp32 logits) against **44.00 GiB already held by PyTorch**, on a
47.54 GiB card. That 44 GiB baseline does not shrink when ranks are added, which
is exactly why nproc 2, 4 and 8 fail identically — the pressure is not per-rank
activations.

For calibration, a **working** bs128 row at nproc 4 sits at **48,152 MiB**. bs128
is already at the ceiling; bs256 with adamw's optimizer state is simply over it.

## Corrections to my earlier claims

I got this wrong three times before reading the allocator output properly, and
each wrong version is in this log, so: it is **not** gradient accumulation, it is
**not** per-rank batch (bs128@nproc4 and bs256@nproc8 are both 32 per rank, ga 2
— one works, one does not), and `expandable_segments` changes the failure mode
only. My first instinct — adam versus muon — was right, and I retracted it too
early because the optimizer was confounded with batch size in the rows I had.

## What this means

**Ten adamw bs256 rows cannot run on A40 at all** and must queue on lotus-0's
A100-80GB: the 16k anchor, ep4, clip1.0, wd0.0, wd0.1, scale0.25, scale0.5,
adam 32k, adam 64k — plus bs512, which is worse again.

That is most of the remaining grid against a single A100 node. Worth deciding
explicitly:

1. **Queue them on lotus-0** — correct but serial; at nproc 2 that is four at a
   time, and the 16k adamw anchor should go first since every axis is defined
   as a deviation from it.
2. **Find more 80 GB hardware.** If any of the borrowed nodes are A100/H100
   rather than A40 this stops being a bottleneck; every node I have surveyed is
   an A40.
3. **Reduce the memory in bergson.** The 44 GiB baseline in the MAGIC backward is
   the real target. If it is dominated by something replicated per rank that
   need not be, this becomes a code fix rather than a hardware constraint — but
   that is a bergson change during a measurement campaign, so your call.

The A40 fleet stays fully useful either way: bs16-bs128 rows and **all muon
bs256 rows** run fine here, and those are progressing now.

## Fleet

114 query scores, 49 bank models, 870 GB free. Live and healthy: adam bs32/bs64/
bs128, muon bs32/bs64/bs128, and the four muon bs256 rows. adam_bs64 is at 20/20
with its bank at 49/100 — still the first row that will land.
