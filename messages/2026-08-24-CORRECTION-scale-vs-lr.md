# 2026-08-24 — CORRECTION: the scale0.25 collapse is the learning rate, not the logit scale

## The diagnostic

`plan_adam_eps1e17_16k_scale0.25` varies two things against the anchor at once:
`logit_scale` 1.0 -> 0.25 **and** lr 2e-4 -> 8e-4 (its own tuned value). An ms
probe holding lr at the anchor value separates them. No bank needed, three
trainings:

| config | lr | ms |
|---|---|---|
| scale 1.0 (anchor) | 2e-4 | 0.9930 |
| scale 0.5 | 2e-4 | 0.9878 |
| **scale 0.25 (diagnostic)** | **2e-4** | **0.9812** |
| scale 0.25 (the grid row) | **8e-4** | **0.9150** |

With lr fixed, halving the logits costs about **0.006 of ms per halving** and
scale 0.25 stays comfortably inside the healthy band. The drop to 0.9150 is
driven by the **4x learning rate**.

## What this changes

**Wrong, and now corrected in the row notes:** "logit scaling degrades both
scorers but hits MAGIC harder". The scale0.25 row is genuinely collapsed
(MAGIC 0.0456), but the diagnostic attributes the damage to lr, and `scale0.5`
-- which runs at the anchor lr -- has MAGIC **0.9448**, i.e. unharmed. Halving
the logits does not damage MAGIC.

**Unchanged:** the ms detector claim. scale0.25 has ms 0.9150 and MAGIC 0.0456;
the row is collapsed and ms detected it. What caused the collapse does not
affect whether ms flags it.

**Still standing, and now more interesting:** EK-FAC drops from 0.4253 to
**0.1760 at scale 0.5**, which holds lr fixed at 2e-4. That one is a real
logit-scale effect, and it hits EK-FAC while leaving MAGIC (0.9448) and ms
(0.9878) essentially untouched. So the scorers do diverge on this axis -- just
not in the direction I first reported.

## The general lesson for the grid

Every row tuned to its own lr differs from the anchor in **two** ways, not one.
That is correct by the CONTROLS protocol (each config is measured at its own
best lr), but it means no row whose tuned lr differs from 2e-4 can attribute its
effect to the named knob without a diagnostic like this one.

Checking `TUNED_LR`, the rows at a non-anchor lr are: bs16 and bs32 (5e-5),
bs64, bs128, ep4, 4k and 64k (1e-4), and **scale0.25 (8e-4, the only row in the
grid at that value)**. The batch-size rows are the ones to watch -- bs16's
collapse (MAGIC 0.1796, ms 0.9133) is at lr 5e-5, four times *lower* than the
anchor, so it is not the same story as scale0.25 and needs its own diagnostic
before anyone attributes it to batch size alone.

An ms probe costs three trainings and no bank, so these diagnostics are cheap.
Running bs16 at the anchor lr next.
