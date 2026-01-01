#!/usr/bin/env python3
"""Slice an EK-FAC/MAGIC scores directory down to a range of query columns.

    python scripts/shard_scores.py <scores_dir> <start> <stop> <out_dir>

Why this is needed: slicing only the query DATASET is not enough. bergson checks
that the score matrix has exactly one column per query document and dies with

    ValueError: scores has 20 query columns but the query dataset has 6 documents

so a query shard needs a scores directory sliced to match. scores.bin is a flat
structured array, num_rows x (score_i, written_i) for i in [0, num_scores), and
the fields are renumbered from 0 in the output so the shard sees a dense 0..k-1.
"""
import json, shutil, sys
from pathlib import Path
import numpy as np

src, a, b, dst = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
info = json.load(open(src / "info.json"))
n_rows, n_scores = int(info["num_rows"]), int(info["num_scores"])
if not (0 <= a < b <= n_scores):
    raise SystemExit("bad range %d:%d for %d scores" % (a, b, n_scores))

dt_in = np.dtype({k: info["dtype"][k] for k in ("names", "formats", "offsets", "itemsize")})
arr = np.memmap(src / "scores.bin", dtype=dt_in, mode="r", shape=(n_rows,))

k = b - a
names, formats = [], []
for i in range(k):
    names += ["score_%d" % i, "written_%d" % i]
    formats += ["float32", "bool"]
dt_out = np.dtype({"names": names, "formats": formats})

dst.mkdir(parents=True, exist_ok=True)
out = np.memmap(dst / "scores.bin", dtype=dt_out, mode="w+", shape=(n_rows,))
for i in range(k):
    out["score_%d" % i] = arr["score_%d" % (a + i)]
    out["written_%d" % i] = arr["written_%d" % (a + i)]
out.flush()

info2 = dict(info)
info2["num_scores"] = k
info2["dtype"] = {"names": names, "formats": formats,
                  "offsets": [j * 5 for j in range(2 * k)], "itemsize": dt_out.itemsize}
# offsets must come from the real dtype, not guessed
info2["dtype"]["offsets"] = [dt_out.fields[n][1] for n in names]
json.dump(info2, open(dst / "info.json", "w"))

for f in ("data.hf", "hessians.pth", "hessians_eigen.pth", "normalizers.pth",
          "processor_config.yaml", "total_processed.pt", "config.yaml"):
    s = src / f
    if s.is_dir():
        if not (dst / f).exists():
            shutil.copytree(s, dst / f)
    elif s.is_file():
        shutil.copy2(s, dst / f)

# verify against the source
chk = np.memmap(dst / "scores.bin", dtype=dt_out, mode="r", shape=(n_rows,))
ok = all(np.array_equal(chk["score_%d" % i], arr["score_%d" % (a + i)]) for i in range(k))
print("  wrote %s  queries %d:%d -> 0:%d  itemsize %d  verify=%s"
      % (dst, a, b, k, dt_out.itemsize, "OK" if ok else "MISMATCH"))
if not ok:
    raise SystemExit("verification failed")
