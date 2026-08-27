# The 2000-step MAGIC scorings died of the ssd-1 quota, not of MAGIC

Both 2000-step rows have an EK-FAC LDS + filter delta but no MAGIC side:

    plan_adam_eps1e17_32k_bs32   ekfac 0.4146 / 0.05036   magic  - / MISSING
    plan_muon_eps1e17_32k_bs32   ekfac 0.4281 / 0.04959   magic  - / MISSING

Cause, from the logs (19h stale before I touched them):

    muon: OSError: [Errno 122] Disk quota exceeded   <- explicit
    adam: DistBackendError -> child exited -6 (SIGABRT)  <- same event, peer died

This is the ssd-1 quota incident, the third time it has destroyed work. It is
NOT a MAGIC-at-high-step-count failure; do not read these crashes as evidence
that MAGIC breaks at 2000 steps.

ssd-1 now has 2.4T free after the bank reclamation, so both are relaunched
(lucia-ord-0, A40, nproc_per_node=2, ports 60210/60220,
BERGSON_DIST_TIMEOUT_MIN=1440). Both confirmed progressing.

Do not relaunch these elsewhere - claims are under
_claims/plan_{adam,muon}_eps1e17_32k_bs32__magic.

Why it matters: these are the only two rows where the LDS<->filter-delta
correlation can gain a MAGIC point without building a new bank. Filter-delta
coverage is otherwise complete - 0 (row,scorer) pairs have an LDS but no delta,
so there is no idle-GPU work that adds a pair by re-using an existing LDS.

Also relaunched this sweep:
- plan_adam_eps1e17_64k_bs256 bank subsets 37-40 (marisa-0, A100). 97/100
  present; 37/38/39 were shard_35_40s tail, dead 112m. Bank is A100-only
  (marisa-0 + shivam2-0) and must stay that way per D17.
- gpt2medium_128k_bs2048_lr6.25e-6 - silent death at step 36/125, no traceback.
- gpt2medium_256k_bs4096_lr1.25e-5 - new grid point, nproc=2 with
  grad_accum 128 so effective batch stays 4096 and per-device micro stays 16.
