# A wedged mount does not have to strand a node's GPUs

marisa-0 sat with six idle A100s for about seventeen hours because its ceph
client is wedged on `/mnt/ssd-1` -- a `mv` stuck in uninterruptible D state,
which blocks anything that enumerates the dataset directory. The node was
written off as unusable.

It was not. Only **ssd-1** is wedged there. `/mnt/ssd-2` reads and writes
normally on that node, so work whose dataset lives on ssd-2 runs fine:

    mkdir -p /mnt/ssd-2/lucia/datasets_local
    cp -a /mnt/ssd-1/.../datasets/train_256k.hf /mnt/ssd-2/lucia/datasets_local/

then rewrite the dataset path in the generated config before launching. The data
is identical -- `cp -a` -- so nothing about the run changes except which mount it
reads from. `train_16k.hf`, `train_32k.hf` and `train_256k.hf` are copied there
now; add others as needed.

This recovered all six GPUs: two 256k tuning points and an EK-FAC scoring job.

## Why it mattered more than it looks

The 256k sweep had four of six points, and both winners were still ENDPOINTS of
their three-point sweeps, so the selection rule refuses to pick an lr and the
entire 256k rung of the step ladder was gated behind two missing runs. The
stranded node was exactly the capacity needed to unblock it.

## The general lesson

"Node X is broken" is too coarse. Check which mount, and whether the work can be
moved off it. The cost here was a 20 GiB copy against seventeen hours of six
A100s.

And when nothing in the queue fits the free GPUs, that is a signal to design a
run, not to report the gap and move on. The question to ask is which unmeasured
point unblocks the most downstream work -- here, two tuning points gating a whole
rung.
