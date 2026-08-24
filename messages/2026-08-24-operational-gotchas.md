# 2026-08-24 — operational gotchas fixed today, and the ones you must still handle

Everything here has bitten a real run. Tools live in
`/mnt/ssd-2/lucia/paper_runs/_orchestration/` unless noted.

## 1. File permissions: run `fix_perms.sh` from BOTH uid sides

`safetensors` writes `model.safetensors` mode **0600 explicitly**, so `umask`
does not help, and this fleet is split across two uids:

    uid 1000: bellflower-0, lucia-ord-0, allium-0, marisa-0
    uid 1001: secret-ord-0, iris-0

`chmod` only works from the owning uid, so **running the fix on one node
silently leaves the other uid's files unreadable**. That broke two bank uploads
and an EK-FAC run (which surfaces as a confusing `FileNotFoundError`, not a
permission error). Re-run after a bank finishes -- files created since the last
pass are 0600 again.

    bash /mnt/ssd-2/lucia/paper_runs/_orchestration/fix_perms.sh   # on a uid-1000 node
    bash /mnt/ssd-2/lucia/paper_runs/_orchestration/fix_perms.sh   # and a uid-1001 node

## 2. The metasmoothness field is `fd_step`, NOT `h`

The `msfill_*` configs under `/mnt/ssd-2/lucia/muon4k/` use `h: 0.1`. Current
bergson's `MetasmoothnessConfig` has **`fd_step`**; `h` is rejected at parse
time. Use `scripts/gen_ms.py`, which derives the probe from the row's own
`experiment.yaml`, keeps only fields the config class accepts, and **prints
anything it dropped**.

ms needs **no retraining bank** -- it is three trainings -- so it can run on any
idle GPU, including for rows whose banks are unfinished. It was the largest
piece of idle-capacity waste today: the column sat at 4/30 while GPUs were free.

## 3. A config field that does not exist kills the run before step 1

Third time this has cost hours (`logit_scale` in `gen_tuning_run.py`, `h` above,
and nearly `logit_scale` again in the ms probes). A no-op **value** does not
help when the **field** is unknown -- simple_parsing rejects the whole config.
Generators must emit a field only when the target checkout has it.

**PR #433 is merged upstream, but the pinned worktree
`/mnt/ssd-1/lucia/bergson-main-paper-429` still does NOT have `logit_scale`** --
it has not fetched. For `scale0.25`/`scale0.5` work use
`/mnt/ssd-1/lucia/bergson-logit-scale`, which is also what their banks were
built with. **Do not bump `-429` while runs are in flight**: a resumed run would
mix code versions inside one bank.

## 4. New run dirs live on ssd-1, not ssd-2

`gen_experiment_run.py` routes new runs to ssd-1 (ssd-2 was near full).
`launch.sh` resolves both, and takes `RUN_ROOT=<dir>` for rows too large for
either. Anything that hard-codes the ssd-2 experiments path will hand bergson a
nonexistent `experiment.yaml` **after** committing the claim.

## 5. Before sharding, list `validation_*.csv`, not just `validation.csv`

sm_muon had already been re-sharded by another node after its main died; I did
not check, added a second slice set, and wasted ~4 h -- and mixed node types in
the process, which cost far more (see D17). Also check `magic_lds.py` merges
cleanly afterwards: it asserts each subset appears exactly once.

## 6. Environment differences that are real but harmless so far

All nodes: torch 2.13.0+cu126, CUDA 12.6, NCCL 2.29.3, triton 3.7.1,
transformers 5.15.1, datasets 5.0.1, numpy 2.4.6, `tf32_matmul=False`.

Two wrinkles worth knowing:
- **lotus-0 has no `/home/lucia/envs/paper`**; it uses
  `/mnt/ssd-2/lucia/envs/paper`, same versions, different prefix.
- **Python patch version differs**: A40 nodes and lotus-0 are 3.11.15, the three
  borrowed A100 pods are 3.11.16. This is confounded with the GPU split in the
  D17 measurement; a control is running on lotus-0 (A100 + 3.11.15) to separate
  them.
- `tf32_cudnn=True` fleet-wide although CONTROLS asks for tf32 off. The matmul
  path is off and GPT-2 has no convolutions, so no result is believed affected,
  but it does not match the stated intent.
