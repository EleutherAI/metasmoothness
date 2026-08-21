# The pinned run environment

Every paper run executes inside one pinned Python environment. D15 measured that
an unrecorded environment difference breaks bit-reproducibility even when code,
config, seed, and world size all match, so the environment is part of run
identity — not a detail of how a node happens to be set up.

This file is the standing reference. Dated announcements live in `messages/`;
what is written here is the current contract.

## Where it lives

    /home/lucia/envs/paper          # node-local, same absolute path on every node
    messages/requirements.lock      # the pins, committed in this repo

The environment is **node-local by design, not on the shared filesystem.** A
conda env is tens of thousands of small files; creating one directly on
`/mnt/ssd-2` does not finish in a useful amount of time, while locally it takes
a few minutes. Identity across nodes comes from the lock file, not from sharing
one copy of the files.

## Building it on a new node

    bash /mnt/ssd-2/lucia/paper_runs/_orchestration/build_env_local.sh apply

That installs exactly what `requirements.lock` pins and finishes with a leak
check (below). It is safe to re-run; it rebuilds from scratch.

Re-resolving versions (`... build_env_local.sh seed`) rewrites the lock and
therefore changes the environment for everyone. Only do that deliberately, on
one node, and announce it in `messages/` — every node must then rebuild, and
banks built before and after are in different environments.

## Running with it

    cd /tmp && \
      CUDA_VISIBLE_DEVICES=<gpus> MASTER_PORT=<unique> PYTHONNOUSERSITE=1 \
      PYTHONPATH=/mnt/ssd-1/lucia/bergson-main-paper \
      /home/lucia/envs/paper/bin/python -s -P -m bergson <config>

Every part of that line is load-bearing:

| part | why |
|---|---|
| `cd /tmp` | a bergson checkout as cwd shadows `PYTHONPATH` (D15; this voided a day of gate experiments) |
| `-P` | blocks cwd from `sys.path` |
| `-s` + `PYTHONNOUSERSITE=1` | blocks `~/.local` from `sys.path` — see below |
| `PYTHONPATH=...bergson-main-paper` | the pinned `main` checkout, not the shared `bergson-damping` working checkout, whose branch changes |
| `MASTER_PORT` unique per run | concurrent runs on one node otherwise share a rendezvous |
| `CUDA_VISIBLE_DEVICES` | pin the run to GPUs you have checked are free |

`scripts/gen_experiment_run.py` prints the command for a given row; the launcher
at `/mnt/ssd-2/lucia/paper_runs/_orchestration/launch.sh` applies all of the
above plus a GPU preflight check.

## Traps this environment exists to prevent

**`-P` alone does not protect you.** It blocks cwd but not user site-packages.
Nodes here have a populated `~/.local/lib/python3.11/site-packages` with its own
torch, which silently shadows the pinned env:

    torch file: /home/lucia/.local/lib/python3.11/site-packages/torch/__init__.py

Same failure mode as D15, one directory over, and just as silent. On some nodes
`/opt/conda` has no torch at all — every apparently-working import there was
coming from `~/.local`.

**It corrupts construction too, not just execution.** With user site visible,
`pip` treats anything already in `~/.local` as "already satisfied" and skips
installing it into the new env, which then falls back to `~/.local` at runtime.
`build_env_local.sh` exports `PYTHONNOUSERSITE=1` for the installs for this
reason.

**`pip freeze` emits unusable paths in a conda env.** Conda-provided packages are
recorded as `packaging @ file:///home/conda/feedstock_root/...`, which exists on
no other node and makes `pip install -r` fail with a bare `OSError: [Errno 2]`.
The lock filters `@ file://` lines out.

**`pip ... | tail` hides failures from `set -e`.** The pipeline takes `tail`'s
exit code, so a failed install reports success and leaves a broken env. The build
script sets `-o pipefail`.

## Verifying a node

    /home/lucia/envs/paper/bin/python -s -P -c "
    import torch; print(torch.__version__, torch.cuda.device_count())"

and the leak check that `build_env_local.sh` runs at the end, which asserts every
core module resolves inside the prefix. A node that has not passed the leak check
has not adopted the environment, regardless of what `pip list` says.

## Recording it

Record the environment with every claim, alongside `nproc` and the GPU model:
world size and hardware are both part of run identity, and the environment is the
component D15 found missing from the historical banks. Runs produced outside the
pinned environment are marked **provisional** in their row notes — usable, and
replaced by a pinned rerun when capacity allows.

## Hardware

The fleet is deliberately mixed (A40 and A100). Per Lucia's ruling, do not
reshuffle axes across nodes chasing GPU uniformity — record the GPU model with
the claim so cross-axis comparisons stay auditable.

Memory matters when choosing a node: until an `eval_batch_size` knob is in the
pinned checkout, the query/eval stream batches at the *training* batch size, so
`bs256` rows need an 80 GB card and `bs512` does not fit at all. See CONTROLS.md
("eval batch size") and `messages/` for the current state of that fix.
