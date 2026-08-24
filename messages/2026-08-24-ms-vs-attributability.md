# 2026-08-24 — metasmoothness vs measured attributability: what 11 rows can and cannot say

## The numbers

| run | ms | MAGIC | EK-FAC |
|---|---|---|---|
| sm_muon (anchor) | 0.9964 | 0.8379 | 0.4237 |
| muon 8k | 0.9962 | 0.7712 | 0.3881 |
| adam 4k | 0.9946 | 0.9295 | 0.3975 |
| muon bs128 | 0.9944 | 0.8480 | 0.4635 |
| adam bs128 | 0.9935 | 0.9441 | 0.4551 |
| adam wd0.1 | 0.9930 | 0.9414 | 0.4244 |
| sm_adamw (anchor) | 0.9930 | 0.9411 | 0.4253 |
| adam wd0.0 | 0.9930 | 0.9410 | 0.4235 |
| adam 8k | 0.9924 | 0.9163 | 0.3869 |
| adam clip1.0 | 0.9896 | 0.8982 | 0.4176 |
| **muon 4k** | **0.9037** | **0.3020** | 0.3031 |

    spearman(ms, MAGIC)  = -0.045   p = 0.89
    spearman(ms, EK-FAC) = +0.355   p = 0.29

## Do NOT read that as "metasmoothness does not predict attributability"

**The predictor barely varies.** Ten of the eleven rows sit between 0.9896 and
0.9964 — a spread of **0.007** — while MAGIC ranges over 0.30 to 0.94. A rank
correlation computed over a band that thin is measuring rank noise, not
relationship. This is textbook range restriction: with essentially no variance
in the predictor, near-zero correlation is the expected output whether or not a
relationship exists.

**The single row where ms actually moves is consistent with a relationship.**
muon 4k is the only row where ms leaves the band (0.9037), and it also has by
far the lowest attributability in the entire grid (MAGIC 0.3020, next lowest
0.7712). One point cannot establish a trend, but it points the right way.

## The within-pair comparison, which is cleaner

Optimizer pairs hold everything else fixed, so the sign is meaningful even when
the magnitudes are tiny:

| pair | ms (adamw - muon) | MAGIC (adamw - muon) | same direction? |
|---|---|---|---|
| 4k | +0.0910 | +0.6275 | yes |
| 8k | -0.0038 | +0.1451 | **no** |
| bs128 | -0.0009 | +0.0961 | **no** |
| anchor | -0.0033 | +0.1032 | **no** |

In three of four pairs muon has the HIGHER metasmoothness and the LOWER
attributability. The three disagreeing ms gaps are 0.001-0.004, i.e. inside the
band where ms is not resolving anything, so this is weak evidence — but it is
weak evidence pointing the wrong way, and it is worth knowing before the paper
leans on ms as a predictor.

## What would actually settle it

The grid needs rows whose ms is not ~0.99. Everything measured so far is a
gently-tuned fine-tune, and those all look alike to the probe. The two
configurations that already break attributability hard — **bs16 (MAGIC 0.1796)**
and **logit scale 0.25 (MAGIC 0.0456)** — are exactly the places ms is most
likely to move, and neither has an ms value yet. Both are running now, along
with bs32/bs64/bs512, scale0.5 and the muon arms.

If ms drops on those rows, the probe tracks attributability where it matters and
the near-zero correlation above is just range restriction. If ms stays at ~0.99
while MAGIC sits at 0.05, that is a real negative result about the probe, and it
would be visible in a single number rather than a correlation over a thin band.
