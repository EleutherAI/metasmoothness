# marisa-0 is occupied by another tenant - do not launch there

State as of this sweep: all 8 GPUs at a uniform 54432 MiB and 76-100% util,
and 63 of my bergson processes on the node are <defunct> zombies.

The occupying PIDs return NO cmdline even to root:

    nvidia-smi --query-compute-apps=pid  ->  263534 263535 263549 ... 304237
    ps -o args= -p <pid>                 ->  empty

That means they live in another container PID namespace - another pod sharing
the node hardware. The uniform 54432 MiB across all 8 GPUs is one 8-GPU job.

## How I got this wrong

My idle check is `nvidia-smi memory.used < 500`. That raced the other tenant
coming up: I sampled while their allocator had not yet grown, called 4 GPUs
idle, and launched into them. Everything I put on marisa-0 this sweep died:

  - plan_adam_eps1e17_64k_bs256 bank_shard_37_40  (died at step 20/500)
  - gpt2medium_128k_bs2048_lr6.25e-6              (relaunch, died)
  - gpt2medium_128k_bs2048_lr3.1e-6               (died)

Nothing was corrupted - the bank shard writes per-subset and got no further
than its own base training, so the bank is still a clean 97/100.

## CORRECTION: the cmdline test gives false positives

I first wrote that an empty cmdline for a compute-app PID proves the GPU
belongs to another container. That is WRONG and I checked it against a node
where I know the answer:

    allium-0: gpu4-7 busy, 4 compute-app PIDs, ALL with empty cmdline
              ...and 6 live bergson processes. They are MY OWN workers.

torch spawns distributed workers whose argv does not survive into `ps`, so
"empty cmdline" describes my own jobs just as well as a foreign tenant's.
Using it as the test would have made me abandon nodes I am happily using.

What actually distinguished marisa-0 was the conjunction, not any single fact:

    - ALL 8 GPUs busy at one uniform size (54432 MiB) = one 8-GPU job
    - ZERO live bergson processes (63 of mine, every one <defunct>)
    - my logs stop advancing while the GPUs stay pinned at 100%

Zero-live-bergson-plus-busy-GPUs is the signal. Count live (non-Z) bergson
processes and compare against the jobs you believe you are running there:

    ps -eo stat=,args= | grep "[b]ergson" | grep -vc "^Z"

memory.used alone is still not enough on its own - it is a point sample and
reads near zero for a job that is still coming up, which is how I raced this
tenant in the first place.

## Rehomed / still needs a home

  - bank_shard_37_40 -> relaunched on lotus-0 [4,5]. lotus-0 is A100, same
    class as the rest of this bank (marisa-0 + shivam2-0), so D17 holds.
  - the two gpt2medium bs2048 points (lr6.25e-6, lr3.1e-6) are DEAD and
    unclaimed. They need an A40 pair each. Claims released.

Healthy right now: shivam2-0 (7 live bergson, 0 defunct), lucia-ord-0, lotus-0.
