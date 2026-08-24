# ckptavg: MAGIC already has checkpoint averaging in main; EK-FAC does not

**This supersedes the earlier version of this note, which was wrong.** It said
bergson implements no checkpoint averaging at all. That grep ran against the
pinned `-429` worktree, not against main. Main has it for MAGIC.

## What actually exists

`bergson/config/config.py` — a real, documented field:

    @dataclass
    class ValidationConfig(TrainingConfig, ABC):
        ckpt_avg_k: int = 1
        """Average the query gradient over the last ``k`` saved trajectory checkpoints."""

`bergson/magic/cli.py` — implemented and wired end to end. `compute_query_gradients`
loads the last `k` checkpoints, recomputes the query gradient at each, averages
them, and restores the final state in a `finally`. It raises if fewer than `k`
checkpoints were saved. The Magic path passes `run_cfg.ckpt_avg_k` through at the
call site, so it is settable from a config today.

That is exactly D9's definition, for MAGIC.

## What is missing

**EK-FAC.** D9 requires that *both* scorers use the averaged gradient — MAGIC
seeds its reverse pass with it, EK-FAC preconditions it. EK-FAC runs a different
pipeline (`hessians/pipeline.py`, driven by IndexConfig / HessianConfig /
ScoreConfig), none of which inherit `ValidationConfig`, so `ckpt_avg_k` is not
reachable from the EK-FAC path at all. `grep -rn ckpt_avg_k bergson/` returns
only `config.py` and `magic/cli.py`.

So the PR to raise is the EK-FAC side, not MAGIC.

## The other blocker is the pinned environment

`-429` predates the feature entirely. Running the ckptavg row under the pinned
venv is therefore not possible as things stand — the config field would be
rejected, the same way `-429` rejects the filter estimator's config because it
predates PR #430. Either the row runs against main (and is labelled as such, cf.
scale0.5, which ran against the logit-scale PR rather than `-429`), or the pin
moves. That is a D15 question, not a code one.

## The bank is ready

The current 16k anchor `sm_adamw_eps1e17_16k_bs256` is venv-valid and still has
its checkpoints — steps 0 / 62 / 93 / 109 / 117 / 121 / 123 / 124 — so the last
four exist and `ckpt_avg_k = 4` has something to average. CONTROLS classes the
axis as eval-side only, same trained model, so no rebuild is needed. D9's
recorded blocker (the old anchor's checkpoints were deleted) does not apply here.
