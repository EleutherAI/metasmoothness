#!/usr/bin/env python3
"""Slice bergson scores.bin into per-query-column shards (CORRECT alignment).
    slicefix.py <scores_dir> <out_dir> <out_prefix> <shard_size> [<n_queries>]
Source dtype is read from info.json offsets (8-byte aligned); reconstructing it
unaligned misreads every record past row 0 (the 2026-09-04 bug)."""
import json, sys
import numpy as np
from pathlib import Path
src = Path(sys.argv[1]); out_dir = Path(sys.argv[2]); prefix = sys.argv[3]; step = int(sys.argv[4])
info = json.load(open(src / "info.json")); dd = info["dtype"]
nrows = info["num_rows"]; nqs = int(sys.argv[5]) if len(sys.argv) > 5 else info["num_scores"]
sdt = np.dtype({"names": dd["names"], "formats": [np.dtype(f) for f in dd["formats"]],
                "offsets": dd["offsets"], "itemsize": dd["itemsize"]})
mm = np.memmap(src / "scores.bin", dtype=sdt, mode="r", shape=(nrows,))
for lo in range(0, nqs, step):
    hi = min(lo + step, nqs); out = out_dir / f"{prefix}_q{lo}_{hi}"; out.mkdir(parents=True, exist_ok=True)
    on, of = [], []
    for j in range(hi - lo): on += [f"score_{j}", f"written_{j}"]; of += ["float32", "bool"]
    odt = np.dtype({"names": on, "formats": of}, align=True)
    om = np.memmap(out / "scores.bin", dtype=odt, mode="w+", shape=(nrows,))
    for j in range(hi - lo): om[f"score_{j}"] = mm[f"score_{lo+j}"]; om[f"written_{j}"] = mm[f"written_{lo+j}"]
    om.flush()
    oi = dict(info); oi["num_scores"] = hi - lo
    oi["dtype"] = {"names": on, "formats": of, "offsets": [odt.fields[n][1] for n in on], "itemsize": odt.itemsize}
    json.dump(oi, open(out / "info.json", "w"), indent=2)
    # Copy the source config verbatim so the slice inherits the source score
    # convention (higher_is_better). Hardcoding true here silently flipped
    # MAGIC (which records false) and inverted proponent selection (2026-09-04b).
    src_cfg = src / "config.yaml"
    if src_cfg.is_file():
        (out / "config.yaml").write_text(src_cfg.read_text())
    else:
        (out / "config.yaml").write_text("steps:\n- score:\n    score_cfg:\n      score: individual\n")
    # Self-check: the slice's column j MUST equal the source's column lo+j
    # exactly. A silent alignment mismatch (the 2026-09-04 bug) selects wrong
    # docs and produces garbage filter deltas.
    _v = np.memmap(out / "scores.bin", dtype=odt, mode="r", shape=(nrows,))
    for j in range(hi - lo):
        assert np.array_equal(np.asarray(_v[f"score_{j}"]),
                              np.asarray(mm[f"score_{lo+j}"])), \
            f"SLICE MISMATCH col {j} in {out} -- alignment bug"
    print(f"sliced {out} [verified]")
