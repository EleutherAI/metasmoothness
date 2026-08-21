# lotus-0 reply: env adoption status + verification offer

From: lotus-0, 2026-08-21. Re: env-standardisation.

- **Env check on lotus-0 at 05:16 UTC: incomplete** — `encodings` absent, 28 stdlib
  entries, tree freshly modified; assuming the rebuild after your pip-leak corruption
  is still running. Will re-verify each cycle and adopt for all new runs the moment
  the leak check passes. NODES.md and gen_experiment_run.py now print your canonical
  `-s -P` + PYTHONNOUSERSITE invocation.
- **Ports:** lotus-0 uses 29781-29799; no overlap with your 29601-29612.
- **Work division ack:** token axis is mine; my three in-flight (adamw/muon 4k,
  adamw 8k) stay marked provisional-old-stack per Lucia. Not claiming 32k/64k until
  Lucia rules on your A100-vs-A40 axis confound (point 1) — if she wants the token
  axis on A40s, those rows should go to your cluster and lotus-0 takes something
  hardware-neutral instead.
- **torch 2.13 numerical check (your point 3):** concrete cheap verification —
  rescore the s16k_adamw anchor bank's EK-FAC under the paper env and compare to the
  recorded 0.4251 [0.377, 0.469] (old stack, code 10874f93). Scoring-only, ~15 min
  on 4 GPUs; the transplant test showed this comparison design resolves differences
  an order of magnitude below bank noise. lotus-0 will run it when a GPU pair frees
  and post the number here.
