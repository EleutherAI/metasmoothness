# STOP plan_adam_eps1e17_16k_scale0.25 - badly mistuned (sweep verdict)

From: lotus-0, 2026-08-22. Sweep results, heldout evaluated WITH the scale
applied (raw logits of scaled models are miscalibrated by design - see
heldout_eval.py --logit-scale, commit 87c870c):

- scale0.5: 3.3168 / 3.3020 / 3.3012 at 1e-4/2e-4/4e-4 -> tie rule selects
  2e-4. YOUR RUNNING scale0.5 BANK IS CORRECTLY TUNED - continue.
- scale0.25: 3.6272 / 3.5374 / 3.4733 -> endpoint win by 0.064, optimum above
  the grid; 8e-4 extension running on lotus-0 now. YOUR scale0.25 BANK AT 2e-4
  IS MISTUNED BY >=2x - stop it; relaunch at the sweep winner once the
  extension lands (also note only the 4e-4 point beats untrained gpt2, and
  barely - this row may end up flagged under CONTROLS rule 4).

Mechanism: strong logit scaling flattens head gradients, pushing the effective
lr optimum far up - the sweep-centers prediction (scale is lr-neutral) held for
0.5 and failed for 0.25.

## Final verdict: scale0.25 tuned lr = 8e-4

Extensions: 8e-4 = 3.4338, 1.6e-3 = 3.4341 (0.0003 tie, curve flat). 8e-4
selected and written to TUNED_LR. Full curve from 1e-4: 3.6272 / 3.5374 /
3.4733 / 3.4338 / 3.4341 - the optimum sits 4x above the anchor. Relaunch your
scale0.25 bank at 8e-4 (regenerate; the row carries the lr now). scale0.5 at
2e-4 confirmed - no action.
