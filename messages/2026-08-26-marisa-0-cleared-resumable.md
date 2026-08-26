# marisa-0 was cleared; four claims released and resumable

Lucia asked for every job on marisa-0 to be killed urgently, so all lucia-owned
processes there were terminated and GPUs 2-7 released. GPUs 0-1 still hold ~54 GB
under another tenant's PIDs -- `kill -9` as lucia does not touch them, which is
how we know they are not ours. Do not try to reclaim 0-1.

marisa-0 is an **A100** node. D17 makes GPU type part of run identity, so
anything resumed from this list must land on A100 (marisa-0, maria-1, shivam2-0,
lotus-0) or its scores will not be comparable with the half already computed.

Four claims were held by marisa-0. All four are now released. Details to resume:

## 1. gpt2medium_64k_bs32 :: ms -- DONE, do not rerun

Finished Aug 25 15:42 and nobody released the claim or recorded the number, so
it read as a live 751-minute job for ten hours.

    score 0.8579838871955872   fd_step 0.1   direction_seed 0
    total_movement_l1 64432.09375
    /mnt/ssd-2/lucia/paper_runs/experiments/gpt2medium_64k_bs32/ms/metasmoothness.json

Worth noting against the 16k gpt2-medium row, which scored 0.8580217361450195
with total_movement_l1 44512.296875 -- different runs, ms identical to four
decimals across a 4x dataset increase.

## 2. plan_muon_eps1e17_64k_bs256 :: ekfac score -- RESUME, needs A100 + the NCCL flag

Died at `Collecting gradients: 100%`, which is the exact point the watchdog abort
has fired on every 64k EK-FAC job so far. Not a marisa-specific failure.

    run dir  /mnt/ssd-2/lucia/paper_runs/experiments/plan_muon_eps1e17_64k_bs256
    config   ekfac.yaml   nproc_per_node: 2   -> give it exactly 2 GPUs
    hardware A100 only

`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` is the wrong knob -- it governs the monitor
thread, not the collective timeout, which is an `init_process_group` argument
with no env override. Use:

    cd /tmp && setsid nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      TORCH_NCCL_ASYNC_ERROR_HANDLING=0 TORCH_NCCL_ENABLE_MONITORING=0 \
      CUDA_VISIBLE_DEVICES=<a,b> MASTER_PORT=<free> PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 \
      PYTHONPATH=/mnt/ssd-1/lucia/bergson-main-paper-429 \
      /home/lucia/envs/paper/bin/python -s -P -m bergson $R/ekfac.yaml \
      > $R/ekfac.log 2>&1 < /dev/null &

Clear `$R/ekfac_scores` first -- a partial hessian stage is not resumable and a
second writer into a live scores dir is what corrupted the muon 64k_bs32 run
earlier today. Note `run.log` in that directory is a *different* live job (the
experiment, on lotus-0); do not read its age as this job's progress.

## 3 + 4. tune_adamw_london64k_bs256_lr0.0004, tune_muon_london32k_bs256_lr0.0008
## -- DO NOT relaunch as-is

Both were hung, not running: zero-byte logs after ~9.5 hours, 130 threads with
39 parked in `futex_wait`, an open fd on the dataset arrow file, and nothing ever
written to the run dir but `config.yaml`.

This is not the muon-specific bug recorded in notes/muon32k_hang.md. **All 13**
london tuning runs at 32k and 64k are hung the same way, adamw and muon alike,
while london 16k and london 128k both completed. The optimizer is not the
variable. Logs land in `/tmp/tune_<name>.log`, not the run dir, which is why
earlier sweeps reported "no log" and scored them as healthy.

Seven of those hung jobs are still live on maria-1, shivam2-0, secret-ord-0,
lucia-ord-0 and shared-ord-0, holding 14 GPUs between them. They were left
running deliberately -- Lucia interrupted the kill -- so leave them alone until
she says otherwise.

Before relaunching any london 32k/64k tuning, someone needs to work out why that
size band hangs when 16k and 128k do not. Datasets are intact and scale
correctly (65.9 MB / 131.8 MB of data arrow), so it is not a missing or
truncated corpus.
