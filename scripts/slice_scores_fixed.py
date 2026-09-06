#!/usr/bin/env python3
"""Slice a bergson scores.bin into per-query-column shards. CORRECT alignment.

BUG FIXED 2026-09-04: bergson writes the structured scores 8-byte-aligned
(itemsize 160 for 20 cols). Reading it with an unaligned dtype (itemsize 100)
misreads every record past row 0 -> garbage slices whose top-40 overlaps the
truth by ~1/40. Always build the source dtype from info.json's offsets/itemsize.

    slice_scores_fixed.py <scores_dir> <out_prefix> <shard_size> [<n_queries>]
      e.g. .../ekfac_scores_heldout/scores  heldout_scores  5   20
"""
import json, sys
import numpy as np
from pathlib import Path

src = Path(sys.argv[1]); out_prefix = sys.argv[2]; step = int(sys.argv[3])
info = json.load(open(src / "info.json"))
dd = info["dtype"]
nq = info["num_scores"]; nrows = info["num_rows"]
nqs = int(sys.argv[4]) if len(sys.argv) > 4 else nq
# CORRECT source dtype from info.json (aligned)
sdt = np.dtype({"names": dd["names"], "formats": [np.dtype(f) for f in dd["formats"]],
                "offsets": dd["offsets"], "itemsize": dd["itemsize"]})
mm = np.memmap(src / "scores.bin", dtype=sdt, mode="r", shape=(nrows,))
outdir = src.parent  # write shards next to the row's scores dir's parent (the run dir)
# actually write into the run experiment dir
run_dir = src.parent.parent if src.name == "scores" else src.parent
for lo in range(0, nqs, step):
    hi = min(lo + step, nqs)
    out = run_dir / f"{out_prefix}_q{lo}_{hi}"
    out.mkdir(exist_ok=True)
    on, of = [], []
    for j in range(hi - lo): on += [f"score_{j}", f"written_{j}"]; of += ["float32", "bool"]
    odt = np.dtype({"names": on, "formats": of}, align=True)  # aligned output too
    om = np.memmap(out / "scores.bin", dtype=odt, mode="w+", shape=(nrows,))
    for j in range(hi - lo):
        om[f"score_{j}"] = mm[f"score_{lo + j}"]
        om[f"written_{j}"] = mm[f"written_{lo + j}"]
    om.flush()
    oi = dict(info); oi["num_scores"] = hi - lo
    oi["dtype"] = {"names": on, "formats": of,
                   "offsets": [odt.fields[n][1] for n in on], "itemsize": odt.itemsize}
    json.dump(oi, open(out / "info.json", "w"), indent=2)
    (out / "config.yaml").write_text(
        "steps:\n- score:\n    score_cfg:\n      score: individual\n      higher_is_better: true\n")
    print(f"sliced {out.name} cols {lo}:{hi} (src itemsize {sdt.itemsize}, out {odt.itemsize})")
