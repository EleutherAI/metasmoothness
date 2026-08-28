"""Split MAGIC scoring across queries, which is exact and needs no score merger.

MAGIC runs one full backward over the training trajectory PER QUERY: 20 queries x
2000 steps x ~5 s/it is ~33 h for a single row, and the 4000-step rows are ~111 h.
Sharding across queries fixes that, and bergson already supports the assembly:

    for qi in range(num_query_docs):
        qpath = .../per_query/q{qi}.pt
        if os.path.exists(qpath):        # resume: already scored
            per_query.append(torch.load(qpath)); continue
        ...
    return torch.stack(per_query, dim=-1)

So a shard writes per_query/q{i}.pt for its own queries, and a final pass over the
full 20-query config finds all 20 files, skips every backward, and stacks them.
Nothing numerical is reimplemented here -- only file placement.

The one trap: a shard handed a 2-query slice numbers its outputs q0.pt and q1.pt,
LOCAL to the slice. They must be renamed to their global index before assembly, or
the stack silently gets the wrong query in the wrong column. gather=True does that
rename; it refuses to overwrite a file that already exists.

Usage:
    shard_magic_queries.py <run_id> <cfg-name> emit          # write shard configs
    shard_magic_queries.py <run_id> <cfg-name> gather        # local -> global names
"""
import os
import shutil
import sys
from pathlib import Path

import yaml

run_id, cfg_name, action = sys.argv[1], sys.argv[2], sys.argv[3]
ROOT = Path("/mnt/ssd-2/lucia/paper_runs/experiments") / run_id
SLICES = Path("/mnt/ssd-2/lucia/datasets_local")
base = yaml.safe_load((ROOT / f"{cfg_name}.yaml").read_text())
main_run = Path(base["steps"][0]["magic"]["run_path"])
PAIRS = [(a, a + 2) for a in range(0, 20, 2)]

if action == "emit":
    made = 0
    for lo, hi in PAIRS:
        # Queries already scored by the unsharded run are global-indexed and reusable.
        if all((main_run / "per_query" / f"q{i}.pt").is_file() for i in range(lo, hi)):
            print(f"  q{lo}_{hi}: already scored by the main run, skipping")
            continue
        qds = SLICES / f"query_20_q{lo}_{hi}.hf"
        if not qds.is_dir():
            sys.exit(f"missing query slice {qds}")
        doc = yaml.safe_load((ROOT / f"{cfg_name}.yaml").read_text())
        m = doc["steps"][0]["magic"]
        m["query"] = dict(m.get("query", {}))
        m["query"]["dataset"] = str(qds)
        out = ROOT / f"{cfg_name}_q{lo}_{hi}"
        m["run_path"] = str(out)
        doc["run_path"] = str(out)
        # The trajectory lives in the main run and is read, never rewritten.
        link_src = main_run / "checkpoints"
        out.mkdir(parents=True, exist_ok=True)
        link = out / "checkpoints"
        if not link.exists():
            link.symlink_to(link_src)
        (ROOT / f"{cfg_name}_q{lo}_{hi}.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
        made += 1
        print(f"  wrote {cfg_name}_q{lo}_{hi}.yaml  (queries {lo}-{hi-1}, checkpoints symlinked)")
    print(f"  {made} shard(s); each does {2}/20 of the backwards")

elif action == "gather":
    dst = main_run / "per_query"
    dst.mkdir(parents=True, exist_ok=True)
    moved = skipped = 0
    for lo, hi in PAIRS:
        sd = ROOT / f"{cfg_name}_q{lo}_{hi}" / "per_query"
        if not sd.is_dir():
            continue
        for local in range(hi - lo):
            src, tgt = sd / f"q{local}.pt", dst / f"q{lo + local}.pt"
            if not src.is_file():
                continue
            if tgt.exists():
                skipped += 1
                continue
            shutil.copy2(src, tgt)
            moved += 1
            print(f"  {sd.parent.name}/q{local}.pt -> per_query/q{lo + local}.pt")
    have = sorted(int(p.stem[1:]) for p in dst.glob("q*.pt"))
    print(f"  copied {moved}, skipped {skipped} already present")
    print(f"  per_query now holds {len(have)}/20 queries: {have}")
    print("  when all 20 are present, run the FULL config once -- it skips every "
          "backward and just writes scores.bin")
else:
    sys.exit("action must be emit or gather")
