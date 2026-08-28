"""Find launched runs that are neither alive nor finished.

health_check.sh catches a job that HOLDS GPUs without doing work. It cannot catch
the opposite failure: a job that exits and leaves nothing behind. london_adamw
completed all 20 retrains, exited 1 without writing filter_summary.csv, released
its GPUs, and sat invisible for hours -- three hours of finished retrains that
nothing was looking for.

So: walk the launch registry, and for each entry ask two questions.
    is a process still running this config?      -> ALIVE
    did it leave the output it exists to produce? -> DONE
Neither means the run died silently and its work is sitting unclaimed on disk.

The expected output depends on the step, so this maps run_path to what should be
there rather than guessing:
    filter/validate -> filter_summary.csv, or filter_proponents.csv for a run that
                       finished its retrains but died before summarising (RECOVERABLE
                       -- recover_filter_summary.py rebuilds it with no GPU time)
    bank            -> retrained/subset_N for the range it was given
    magic scoring   -> scores/info.json, or per_query/*.pt for partial progress

Reads the registry rather than scanning directories so a run that never wrote
anything at all still appears.
"""
import os
import sys

import yaml
from collections import OrderedDict

REG = "/mnt/ssd-2/lucia/paper_runs/_logs/launch_registry.tsv"
# The registry lives on the shared volume and kubectl lives on the laptop, so the
# live-process list is gathered outside and passed in as a file.
ALIVE_FILE = sys.argv[1]

# Latest registry entry per config -- a config relaunched after a failure should be
# judged on its most recent launch, not its first.
entries = OrderedDict()
for line in open(REG):
    f = line.rstrip("\n").split("\t")
    if len(f) < 7:
        continue
    entries[f[6]] = {"when": f[0], "host": f[2], "gpus": f[3], "name": f[4], "cfg": f[6]}

alive = {l.strip() for l in open(ALIVE_FILE) if l.strip()}


def run_path_of(cfg):
    """Where this config actually writes.

    NOT derivable from the filename. bank_shard_0_5.yaml writes into
    bank_from_filter/, ekfac.yaml writes into ekfac_scores/, and a magic config
    can write to the row root. Inferring the directory from the config name is a
    mistake I have now made three times (fix_shard_ckpts, recover_filter_summary,
    and the first version of this) -- read run_path out of the YAML instead.
    """
    try:
        doc = yaml.safe_load(open(cfg))
    except Exception:
        return os.path.splitext(cfg)[0]
    if isinstance(doc, dict):
        step = doc.get("steps", [{}])[0]
        if isinstance(step, dict):
            for v in step.values():
                if isinstance(v, dict) and v.get("run_path"):
                    return v["run_path"]
        if doc.get("run_path"):
            return doc["run_path"]
    return os.path.splitext(cfg)[0]


def outputs(cfg):
    """(done, note) for the run this config drives."""
    run = run_path_of(cfg)
    if os.path.isfile(os.path.join(run, "filter_summary.csv")):
        return True, "filter_summary.csv"
    if os.path.isfile(os.path.join(run, "filter_proponents.csv")):
        return False, "RECOVERABLE: retrains done, no summary"
    if os.path.isfile(os.path.join(run, "scores", "info.json")):
        return True, "scores/info.json"
    pq = os.path.join(run, "per_query")
    if os.path.isdir(pq):
        n = len([f for f in os.listdir(pq) if f.endswith(".pt")])
        if n:
            return False, f"partial: {n} query file(s)"
    rt = os.path.join(run, "retrained")
    if os.path.isdir(rt):
        n = len([d for d in os.listdir(rt) if d.startswith("subset_")])
        if n:
            return True, f"{n} subset(s) retrained"
    return False, "no output"


dead = []
for cfg, e in entries.items():
    if cfg in alive:
        continue
    done, note = outputs(cfg)
    if not done:
        dead.append((e, note))
    # A config whose run_path is shared by many shards (a bank) reports the whole
    # bank's progress, so a finished shard shows as done even though this entry is
    # not individually traceable. That is the correct answer for "is work lost".

for e, note in dead:
    print("  %-22s %-13s gpu %-8s %s   [%s]" % (e["name"][:22], e["host"], e["gpus"], e["when"], note))
print("  %d launched run(s) neither alive nor finished, of %d registered configs"
      % (len(dead), len(entries)))
