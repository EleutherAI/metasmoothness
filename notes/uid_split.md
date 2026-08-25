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

## It had already reached the retrain banks

A read-only audit across every run directory on both volumes found 43 files on
the wrong side of the split, and they were not incidental:

    plan_adam_eps1e17_32k_bs256    42 x retrained/subset_N/model.safetensors
    plan_muon_eps1e17_32k_bs256     1 x retrained/subset_N/model.safetensors

42 of that bank 100 subsets could not be opened from seven of the ten nodes. A
filter run or a HuggingFace publish launched from any of them would have read
58 models and reported success, because a bank is enumerated by directory and
nothing checks that each model actually opens.

Relaxed to a+r from the owning node; all 100 now open from a uid-1000 node. No
content was touched -- this adds a read bit and nothing else. The measured LDS
is unaffected either way, since magic_lds reads validation.csv rather than the
models.

Worth re-running the audit before the next publish:

    find <run_dir> -user 1001 ! -perm -o+r     # from a 1001 node
    find <run_dir> -user 1000 ! -perm -o+r     # from a 1000 node

## It reaches the repo too

The checkout on the shared volume is subject to the same split. 20 tracked files
are owned by uid 1000 with no group/other write, so editing one from a 1001 node
fails with PermissionError -- which is how the umask fix first failed to apply,
silently leaving behind a commit whose message claimed it had been made.

`.git` itself is clean (no problematic directories), so commits, branches and
bundles work from any node. Only in-place edits of those files break. This note
is on the other side of the same split: it was written from a 1001 node, so a
1000 node cannot append to it.

The rule: edit a repo file from a node whose uid owns it, or chmod it first.
Making the tree world-writable would fix it but is a worse trade on a source tree
than remembering which side a file came from.

## The durable fix, applied

`scripts/run_filter_slot.sh` now sets `umask 022`, so artifacts it creates are
born group/other readable. That covers the filter path. Anything launched by hand
still inherits the login shell umask 077, so prefer the slot scripts, and run the
audit above before trusting a bank you did not build.

## CORRECTION: umask 022 is not sufficient

The note above says the durable fix is `umask 022` in the launch path. That is
wrong, and it kept producing unreadable models after the fix went in.

huggingface/safetensors saves through a temporary file, and Python creates temp
files 0600 **regardless of umask**. The temp is then renamed into place, so the
final `model.safetensors` is 0600 even under a permissive umask. Measured on
`tune_muon_london16k_bs256_lr0.0002_s42`, launched with umask 022 and still
written `uid=1000 mode=600`.

So the reliable repair is chmod after the fact: `scripts/fix_perms.py`. Only the
owning uid can chmod, so it has to run from a node of EACH uid. First run fixed
488 paths from the 1000 side and 26 from the 1001 side.

Keep umask 022 anyway -- it does help for everything that is not written through
a temp file -- but do not rely on it alone. Run fix_perms before any publish, and
after any batch of runs that another node will need to read.
