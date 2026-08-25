# `lucia` is two different uids on this fleet, and bergson writes mode 0600

    uid 1001   iris-0, secret-ord-0, maria-1
    uid 1000   bellflower-0, allium-0, shared-ord-0, marisa-0, lotus-0,
               shivam2-0, lucia-ord-0

The shared CephFS volumes store the numeric uid, so a file written as 1001 with
mode `-rw-------` cannot be read by the same named user on any of the seven
1000-nodes. Nothing warns; the file lists fine and the open fails.

## How it presented

EK-FAC scoring for `plan_adam_eps1e17_32k_bs32` died on lucia-ord-0 with

    FileNotFoundError: .../base/model/model.safetensors

while `ls -la` on that same node showed the file, 497 MB, right there. Reading
it worked from iris-0 and failed from bellflower-0, allium-0 and lucia-ord-0 --
which is exactly the 1001/1000 split, not a per-node ceph fault.

The tell that separates this from a wedged mount: a capability problem hangs,
this one returns instantly. `head -c 8` is the cheap probe.

## What it costs

Any artifact a 1001-node writes is invisible to 70% of the fleet. That silently
strands work in a way that reads as "the file isn't there yet", and the natural
response -- wait, retry, re-run the producing job -- never fixes it. Bank
subsets, trajectories and scores are all written with the default umask, so any
of them can land on the wrong side of the split.

## Fixing it

Only the owner can chmod, so the repair has to run on a node whose uid matches
the file's:

    chmod -R a+rX <run_dir>          # from iris-0/secret-ord-0/maria-1 for 1001 files

Done for the four bs32 ladder rows; 1 file in the adam 32k arm and 3 in the muon
arm were unreadable, which was enough to stop both EK-FAC jobs.

The durable fix is `umask 022` in the launch path so new artifacts are born
group/other-readable, rather than chasing them afterwards. Worth doing before
the next bank is built across mixed nodes.

## Checking a row before trusting it

    for RID in ...; do
      f=<run>/base/model/model.safetensors
      head -c 8 $f >/dev/null 2>&1 && echo yes || echo NO
    done

Run it from a 1000-node, since that is the majority side.
