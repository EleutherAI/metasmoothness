# Multi-node data collection

Several CoreWeave nodes share this filesystem and this repo checkout. The
`node_in_charge` / `node_checkin_date` columns in `tuning.csv` and `experiments.csv` are the
claim mechanism that keeps two nodes from running the same row.

## Your node ID

Run `hostname`. That is your node ID — CoreWeave node hostnames are unique (for example, this
file was written from `lotus-0`). Use the bare hostname, nothing else.

## Claiming a row

1. Check `git log` first — another node may have claimed since your last look. (All nodes
   share this one checkout, so commits are visible immediately; `git pull` only matters if
   you work from a separate clone of `github.com/EleutherAI/metasmoothness`.)
2. Pick a row whose `node_in_charge` is empty and whose `status` is `empty` (tuning.csv) or
   `planned` (experiments.csv). Respect `priority` order and the `blocked` status.
3. Write your hostname into `node_in_charge` and today's date (UTC, `YYYY-MM-DD`) into
   `node_checkin_date`. Edit the CSV directly for claims — the builders preserve these two
   columns across regeneration. Everything else still goes through the builder scripts.
4. **Commit immediately.** The claim exists when it is committed, not before. Since all nodes
   share one checkout, an uncommitted claim protects nothing.

## Checking in

While you are actively working a row (long banks can take days), update `node_checkin_date`
to the current date at least **once per day** and commit. The date is a heartbeat, not a
start-time.

## Stealing a stale row

If a row's `node_checkin_date` is more than **3 days (72 hours)** old, any node may take it
over: replace `node_in_charge` with your own hostname, set today's date, commit, and note the
takeover in the commit message. Before stealing, check for the previous node's partial
artifacts (the row's `run_dir`) — per-query MAGIC scores and subset-sliced banks resume
cheaply, so prefer resuming to restarting.

## Standing directive (from Lucia): drain the tuning grid

Any agent with idle GPUs should claim and run tuning rows, in `priority` order, until
every non-blocked row in `tuning.csv` is `measured`. Record results in the builder and
commit as each run finishes — do not batch results at the end of a session. Commit every
generated run config: `gen_tuning_run.py` mirrors each yaml to `configs/tuning/`; commit
the mirror together with the claim.

**GPU packing rule:** maximize concurrent runs, not GPUs per run. Two GPT-2 runs on 2 GPUs
each beat one run on 4 — tuning-size models gain little from wider data parallelism, and
throughput of the whole grid is what matters. Suggested slots on an 8-GPU node: 2 GPUs per
run, skipping GPUs that other jobs occupy (check `nvidia-smi` first). Take larger slots
only where memory demands it (gpt2-medium and larger, MAGIC rollouts).

## Running a tuning row

`scripts/gen_tuning_run.py <run_id>` writes the full training config (every control filled;
the row supplies what varies) and prints the three commands: train, held-out eval, checkpoint
cleanup. Conventions it encodes:

- Run outputs live under `/mnt/ssd-2/lucia/paper_runs/tuning/<run_id>_s<seed>/`.
- `PYTHONPATH` must point at the bergson checkout — the `bergson` console script does not put
  the repo on `sys.path`.
- Use the bergson main branch and record the commit in the row's `code_commit`-adjacent notes;
  do not run from someone's work-in-progress branch (the shared checkout's branch changes).
- Tuning runs delete their checkpoints after the held-out number is recorded — except the
  sweep winner's, for expensive configs (64k, larger models): the winning run doubles as the
  experiment's base training (see Reuse rules in EXPERIMENTS_CSV.md). Losing runs always
  clean up fully.
- EK-FAC scoring uses the canonical configuration ruled in D7: the bergson
  default, `inversion="damped_inverse"`, `damping_factor=0.1`. No other EK-FAC settings are
  admissible for `ekfac_lds` cells.

## The pinned environment

Paper runs execute inside the shared pinned venv so every node has identical
torch/CUDA/NCCL/datasets/numpy/triton builds — D15 measured that an unrecorded
environment difference breaks bit-reproducibility even when code, config, seed, and
world size all match. The venv definition lives in this repo once published (see
`messages/`); until a node has adopted it, any run it produces is **provisional**:
kept and usable, replaced with a pinned-venv rerun when capacity allows. Every run
records its environment (torch, NCCL, datasets, numpy, transformers versions) with
its claim; runs made outside the pinned venv are marked provisional in their row
notes when results are recorded.

## Running an experiment row (stage 1)

Learning rates for every non-blocked planned row are final (the tuning grid is fully
measured — the row's `lr` column is the tuned value). Claim exactly as for tuning rows,
then `scripts/gen_experiment_run.py <run_id>` writes the full bank+MAGIC config and prints
the commands. Rules specific to experiment rows:

- **Use the canonical invocation exactly** — three silent-shadowing traps are closed by it
  (cwd on sys.path, user site-packages over the env, port collisions between concurrent
  runs; see messages/2026-08-21-env-standardisation.md):

      cd /tmp && CUDA_VISIBLE_DEVICES=<gpus> MASTER_PORT=<unique-per-run> \
        PYTHONNOUSERSITE=1 PYTHONPATH=/mnt/ssd-1/lucia/bergson-main-paper \
        /mnt/ssd-2/lucia/envs/paper/bin/python -s -P -m bergson <config>

  Verify the env before first use: `build_env.sh` (in paper_runs/_orchestration) ends
  with a leak check asserting every core module resolves inside the env prefix.
- **Record the world size (nproc) in the row notes.** Bit-exact reuse of the bank later
  (MAGIC re-rolls, D9-style retrains) requires the same nproc — measured, not assumed:
  identical env/code/config/seed at nproc 2 vs 4 diverge (max 1.15e-5 after 125 steps;
  only constant buffers match).
- **Disk first:** each run writes ~28 GB of checkpoints plus the retrain bank (~0.5 GB per
  model). Check `df` on the output volume before claiming; do not start a bank you cannot
  finish.
- The `fill_*` rows and the blocked rows (arch, logit-scale, model-size) are NOT claimable
  by this path — see their notes.

## Finishing a row

Fill the result columns by editing the row in the builder script, regenerate the CSV, clear
`node_in_charge` / `node_checkin_date`, and commit script + CSV together. A finished row has
results and no owner.

## Ground rules (short form — details in CONTROLS.md and EXPERIMENTS_CSV.md)

- Per-epoch shuffle code (`1e6eea7f`+); dropout configured 0.0; fp32; seed 42.
- Pin `eps_root: 1e-17` in every yaml — the code default is different.
- Keep checkpoints (`cleanup_ckpts: false`) and `save_optimizer_state: last`.
- Select lr on `heldout_4k` only (`scripts/heldout_eval.py`); never on train loss.
- Datasets: `EleutherAI/bergson-smollm2-scaling` on the Hub, or
  `bergson-damping/runs/ekfac_vs_n/datasets/` locally. Never WikiText.
- Commit results promptly; this repo is the source of truth.
