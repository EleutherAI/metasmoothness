# Both 4000-step EK-FAC scorings wedged writing to ssd-1 (~9.5h lost, 2 GPUs gone)

This is the SAME failure that took marisa-0, and it is no longer a one-node
curiosity: it killed both halves of the 4000-step scoring plan simultaneously.

## What happened

    plan_adam_eps1e17_64k_bs32/ekfac.yaml   iris-0       wedged 9h16m
    plan_muon_eps1e17_64k_bs32/ekfac.yaml   lucia-ord-0  wedged 9h39m

Identical signature on both. Each run has two workers:

    worker A   state D (DNl)   4% cpu    wchan = ceph_mdsc_wait_request
    worker B   state R (RNl)  100% cpu   spinning in the collective, waiting on A

The D worker holds an open fd on its own output:

    /mnt/ssd-1/.../plan_adam_eps1e17_64k_bs32/ekfac_scores/scores.part/scores.bin

So it wedged **writing scores to ssd-1**, blocked in a CephFS MDS request that
never returned. Both logs froze at "Collecting gradients: 16000/16000" - the
gradient pass finished and the write hung.

`ls -d /mnt/ssd-1/...` on the same node returns in **2ms**. The filesystem is not
down. It is a single client request that never completes, exactly like marisa-0.

## Why it fooled me for 9 hours

A frozen log is the documented NORMAL appearance of this job: the 4000-step
EK-FAC rank-0 section legitimately runs 6+ hours with no output. I had written
that down, so a 9h silent log looked like the known-good case. It is not
distinguishable from a hang by log age, GPU memory, or process count - every one
of those looks identical.

The check that DOES separate them, in one command:

    ps -eo pid,ppid,stat,pcpu,args= | awk '$2==<launcher>'

Healthy long serial phase: all workers R or S, at least one near 100% cpu.
Wedged: one worker in **D** with `wchan = ceph_mdsc_wait_request`, its twin
pinned at 100% doing nothing but waiting for it.

Always confirm with `cat /proc/<pid>/wchan`. `ceph_mdsc_wait_request` is the
tell; `do_wait` on the launcher is normal and means nothing.

## Cost

  - ~9.5h x 2 runs of scoring, thrown away (scores.part never became scores)
  - iris-0 gpu3 and lucia-ord-0 gpu0 are now held at 9464 MiB by D-state workers
    that cannot be killed. Both nodes are down a GPU until they restart.
  - The entire 4000-step filter plan is blocked: no scores means no proponent
    filter, under D22 or otherwise.

## Mitigation: stop writing run outputs to ssd-1

Both wedges, and marisa-0, were writes/lookups under /mnt/ssd-1. ssd-2 is a
DIFFERENT Ceph cluster (different mon addresses, different csi volume), so it is
a genuinely separate failure domain, and it has 1.1T free.

  - `gen_bank.py` already rewrites DATASET paths to the ssd-2 mirror for this
    reason. That is not enough - the wedge here was on the OUTPUT path.
  - New runs on rows that live under /mnt/ssd-1 should set `run_path` to an ssd-2
    location. The row directory being on ssd-1 is fine for reading configs; it is
    sustained writes that hang.

Not done yet, and it is the obvious next fix: I do not want to relaunch two more
9-hour scorings into the same trap.
