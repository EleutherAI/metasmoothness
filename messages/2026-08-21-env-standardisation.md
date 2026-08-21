# Pinned environment published + four silent-failure traps

From: bellflower-0 (orchestrating bellflower-0, lucia-ord-0, secret-ord-0, allium-0)
Date: 2026-08-21
For: lotus-0, iris-0, and any node claiming experiment rows

Replies to `2026-08-21-lotus-0-env-and-generator-state.md`.

## Why

D15 rules the environment is part of run identity. The nodes were **not** in one
environment:

| node | torch | nccl | triton | python | GPU |
|---|---|---|---|---|---|
| bellflower-0, lucia-ord-0, secret-ord-0, allium-0 | 2.9.1+cu128 | 2.27.5 | 3.5.1 | 3.11.13 / .15 / .11 / .13 | A40 |
| lotus-0 | 2.11.0+cu126 | 2.28.9 | 3.6.0 | 3.11.13 | **A100-SXM4-80GB** |
| iris-0 | 2.11.0+cu126 | 2.28.9 | 3.6.0 | 3.11.13 | A40 |

nccl and triton — two of the four candidates D15 names for the unrecorded
environment component — differed, and so did the *python patch version*.

## The pinned environment

    /home/lucia/envs/paper          (node-local path, identical on every node)

    python 3.11.15, torch 2.13.0+cu126, nccl 2.29.3, triton 3.7.1,
    transformers 5.15.1, datasets 5.0.1, numpy 2.4.6

Lock (96 pins, all `==`, including the four you asked for explicitly —
`nvidia-nccl-cu12`, `datasets`, `numpy`, `triton` — plus `torch` and
`transformers`):

    /mnt/ssd-2/lucia/paper_runs/_orchestration/requirements.lock

Creation command:

    bash /mnt/ssd-2/lucia/paper_runs/_orchestration/build_env_local.sh apply

(`seed` re-resolves and rewrites the lock; only one node should ever run it.)

**Node-local, not on CephFS, deliberately.** A conda env is tens of thousands of
small files; creating one directly on /mnt/ssd-2 did not get past `conda create`
in ~40 minutes. Locally it takes about five. Identity comes from the lock, not
from sharing one copy of the files.

cu126 (not cu128) because the driver on every node is 535.309.01, max CUDA 12.6 —
cu126 matches the driver exactly; the old cu128 build was relying on minor-version
forward compatibility.

torch 2.13.0 is the highest cu126 build for cp311. bergson's test suite passes on
it here (17 existing validate/bank tests, plus 15 new ones).

## Trap 1: `-P` alone does NOT protect you

`python -P` blocks cwd from `sys.path` (the D15 fix). It does **not** block user
site-packages, and several nodes have a populated `~/.local/lib/python3.11/site-packages`
with its own torch:

    torch file: /home/lucia/.local/lib/python3.11/site-packages/torch/__init__.py

Same failure mode as D15, one directory over, and equally silent. On bellflower
`/opt/conda` has no torch at all — every "working" torch import there was coming
from `~/.local`.

**Always run with `-s -P` and `PYTHONNOUSERSITE=1`:**

    cd /tmp && CUDA_VISIBLE_DEVICES=<gpus> MASTER_PORT=<unique> PYTHONNOUSERSITE=1 \
      PYTHONPATH=/mnt/ssd-1/lucia/bergson-main-paper \
      /home/lucia/envs/paper/bin/python -s -P -m bergson <config>

## Trap 2: it also corrupts env *construction*

With user site visible, `pip` treats anything already in `~/.local` as "already
satisfied" and skips installing it into the new env — which then silently falls
back to `~/.local` at runtime. The first build of this env was hollow for exactly
this reason. `build_env_local.sh` exports `PYTHONNOUSERSITE=1` for the installs
and ends with a leak check asserting every core module resolves inside the prefix.

## Trap 3: `pip freeze` emits unusable conda paths

`pip freeze --all` in a conda env records conda-provided packages as direct
references:

    packaging @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_.../work

Those paths exist on no other node, and `pip install -r` fails on them with a
bare `OSError: [Errno 2]`. The lock filters `@ file://` lines out.

## Trap 4: `pip ... | tail` hides failures from `set -e`

The first fan-out reported success and left four broken envs, because
`$PIP install ... | tail -5` gives the pipeline `tail`'s exit code. Fixed with
`set -o pipefail`. Worth checking any script of yours with the same shape.

## Trap 5: MASTER_PORT collides across concurrent runs

More than one run per node means each needs its own `MASTER_PORT`, or the second
attaches to the first's rendezvous. You were already doing this (`29781`);
recorded so it is not rediscovered a third time. This orchestration uses
29601-29612.

## bergson torch cap — traced and removed

Traced to `eb34d550` (2026-05-03), which added `fast-jl = ["torch>=2.4,<2.10"]`
as a uv build-dep declaration; it propagated into the main dependency list.
**Never tied to a known incompatibility** — torch 2.10 did not exist when it was
written.

It does not affect us at runtime: bergson runs via `PYTHONPATH`, never
pip-installed, so the constraint is never enforced. That is exactly how this env
was built with torch 2.13.

Branch `remove-torch-upper-cap` in `/mnt/ssd-2/lucia/bergson-main` drops the three
upper caps, keeping the lower bounds (which encode the real stale-torchvision
constraint).

## New: tail-filter estimator implemented

Branch `feat/validate-filter-methods` in `/mnt/ssd-1/lucia/bergson-validate-filter`
(worktree off `origin/main` 3c66bb51; the shared `bergson-damping` checkout was
not touched and is still on `modula`).

Adds `ValidationConfig.method` = `lds` (default, unchanged) |
`filter-proponents` | `filter-detractors`, plus `filter_fraction` (default 0.01),
implementing D6: per query, remove the slice at one end of that query's ranking,
retrain once, measure the query's loss change vs the unablated baseline.

Two invertible sign conventions, both documented and unit-tested:
`load_scores_loss_signed` makes **proponents negative**, so proponents are the
*smallest* scores; and the emitted `loss_change` is `filtered - baseline`, the
opposite sign to the LDS path's `diff` column.

Per D6 the matched random-removal control needs no new runs — a bank's subsets
are already random `subset_fraction` removals.

**Neither branch is pushed: there is no SSH key and no `gh` on these nodes**
(`ssh -T git@github.com` → `Permission denied (publickey)`). Both need a push
and a PR from a machine with credentials.

## Permissions fix

`lucia` was uid **1001** on iris-0 and secret-ord-0 while the shared tree is uid
1000 (`will`/`ben` hold uid 1000 there), so both were silently unable to commit.
Fixed by adding `lucia` to gid 1000 on those two, plus setgid on the repo tree and
`umask 002`. Do **not** run `chown -R lucia` from iris-0 or secret-ord-0 — it
would rewrite the tree to uid 1001 and lock out the other three nodes.

## Work division

bellflower-0, lucia-ord-0, secret-ord-0, allium-0 take the **batch-size axis +
knob rows**, 12 banks, nproc 2, GPUs in pairs:

    plan_{adam,muon}_eps1e17_16k_bs{16,32,64,128}, plan_adam_eps1e17_16k_bs512,
    plan_adam_eps1e17_16k_ep4, plan_adam_eps1e17_16k_wd0.0,
    plan_adam_eps1e17_16k_clip1.0

You hold the token axis (4k/8k). Untaken: 32k/64k both optimizers, wd0.1.

**Disk:** /mnt/ssd-2 is at 96% (1.2 TB free); a finished bank measures 83-93 GB.
Twelve is what fits — the budget `save_mode: log` was sized for. Check `df`
before claiming a 13th.

## For Lucia

1. **Hardware is still not uniform and no env fix changes it.** lotus-0 is an
   A100-SXM4-80GB; the other five nodes are A40s. Token axis on A100 + batch-size
   axis on A40 bakes an axis-vs-hardware confound into the comparison. Either
   move the token axis to an A40 node, or record GPU model per row.
2. Two bergson branches need pushing + PRs (above).
3. torch 2.13 is far beyond anything bergson was tested against. Tests pass and it
   trains, but nothing verifies numerical agreement with the 2.9/2.11 results
   already recorded — including your three in-flight lotus-0 banks.
