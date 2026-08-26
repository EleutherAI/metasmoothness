"""Re-probe a finished ms run at different direction seeds.

ms perturbs the data weights along ONE random direction v and scores the
agreement of three trainings at 1, 1+h*v, 1+2h*v. Every value in the grid uses
direction_seed 0, so nothing so far says how much of an ms number is the
configuration and how much is the direction that happened to be drawn.

That matters right now for exactly one cell. london16k_bs256_muon reads 0.8547
against 0.9867 for adamw at the same setting, and against 0.9963 for muon on
smollm2. It is the only value in the london/smollm2 x adamw/muon x bs16/bs256
table that breaks the pattern, and the whole "the corpus was hiding an optimizer
difference" reading rests on it. One draw of v is not enough to carry that.

Writes <run>_seed<N>/ms.yaml pointing at the same data, same lr, same optimizer,
same fd_step -- only direction_seed differs. Anything else changing would make
the comparison meaningless.

    python gen_ms_seeds.py <run_id> <seed> [<seed> ...]
"""
import copy
import os
import sys

import yaml

EXP_ROOTS = ["/mnt/ssd-2/lucia/paper_runs/experiments",
             "/mnt/ssd-1/lucia/paper_runs/experiments"]


def find_run(run_id):
    """Runs live on either volume; a hardcoded root silently misses half of them."""
    for root in EXP_ROOTS:
        if os.path.isfile(os.path.join(root, run_id, "ms.yaml")):
            return root
    return None


if len(sys.argv) < 3:
    sys.exit(__doc__)
run_id, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]

EXP = find_run(run_id)
if EXP is None:
    sys.exit("no ms.yaml for %s under %s" % (run_id, " or ".join(EXP_ROOTS)))
src = os.path.join(EXP, run_id, "ms.yaml")
base = yaml.safe_load(open(src))


def setk(node, key, value):
    hit = False
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                node[k] = value
                hit = True
            elif setk(v, key, value):
                hit = True
    elif isinstance(node, list):
        for v in node:
            if setk(v, key, value):
                hit = True
    return hit


def getk(node, key):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                return v
            r = getk(v, key)
            if r is not None:
                return r
    elif isinstance(node, list):
        for v in node:
            r = getk(v, key)
            if r is not None:
                return r
    return None


for s in seeds:
    tag = "%s_seed%d" % (run_id, s)
    root = os.path.join(EXP, tag)
    cfg = copy.deepcopy(base)
    if not setk(cfg, "direction_seed", s):
        sys.exit("no direction_seed key in %s" % src)
    ms_path = os.path.join(root, "ms")
    setk(cfg, "run_path", ms_path)
    cfg["run_path"] = ms_path
    if os.path.isfile(os.path.join(ms_path, "metasmoothness.json")):
        print("  %-40s already has a result" % tag)
        continue
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "ms.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print("  wrote %-38s seed=%d lr=%s opt=%s data=%s"
          % (tag, s, getk(cfg, "lr"), getk(cfg, "optimizer"),
             os.path.basename(str(getk(cfg, "dataset")))))
