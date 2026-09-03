#!/usr/bin/env python3
"""Numeric gates for the EK-FAC scoring pipeline.

The qwen15b 4k/8k/16k scores were all zero: an interrupted inverse application
left a full-size, all-zero kfac_query/gradients.bin, `resume: true` saw the file
and skipped the stage, and scoring ran zeros to completion. Existence is not
validity; these gates check content.

    python gate_ekfac.py inverse <score_run_dir> [--purge]
    python gate_ekfac.py scores  <score_run_dir>
    python gate_ekfac.py all     <score_run_dir> [--purge]

inverse: kfac_query/gradients.bin must be finite, nonzero-std, <50% exact zeros.
  --purge moves an invalid file to gradients.bin.invalid so a resumed run
  recomputes the stage instead of scoring from it. Run this BEFORE launching
  any scoring run that could resume.
scores: scores/scores.bin -- every written flag true, scores finite, nonzero
  std, <90% exact zeros. Run AFTER scoring, and again at consumption
  (gen_filter.py calls it) so nothing downstream ever ranks on zeros.

Layout: scores.bin is interleaved (score_i: f32, written_i: bool) pairs,
8-byte aligned (bergson score_writer._score_struct_dtype), so f32 view even
indices are scores and byte 4 of each pair is the flag. kfac_query is flat f32.

Exit 0 = pass. Exit 1 = fail (message on stdout, GATE FAIL prefix).
"""
import argparse
import os
import sys

import numpy as np


def sample(a, n=2_000_000):
    # Three contiguous slabs, not a stride: a strided read of a 100GB memmap
    # over Ceph fetches millions of scattered pages and takes minutes.
    if a.size <= n:
        return np.asarray(a, dtype=np.float64)
    k = n // 3
    mid = a.size // 2
    return np.concatenate([np.asarray(a[:k], dtype=np.float64),
                           np.asarray(a[mid:mid + k], dtype=np.float64),
                           np.asarray(a[-k:], dtype=np.float64)])


def check_inverse(run, purge=False):
    p = os.path.join(run, "kfac_query", "gradients.bin")
    if not os.path.exists(p):
        print("inverse: no kfac_query/gradients.bin yet -- nothing to gate")
        return True
    if os.path.getsize(p) == 0:
        print("inverse: gradients.bin is EMPTY (0 bytes) -> GATE FAIL")
        if purge:
            d = os.path.dirname(p)
            os.rename(d, d + ".invalid")
            print(f"purged: {d} -> .invalid")
        return False
    s = sample(np.memmap(p, dtype=np.float32, mode="r"))
    std, zeros = float(np.std(s)), float(np.mean(s == 0))
    finite = bool(np.isfinite(s).all())
    ok = finite and std > 1e-12 and zeros < 0.5
    print(f"inverse: std={std:.3e} zeros={zeros:.3f} finite={finite} -> "
          + ("PASS" if ok else "GATE FAIL"))
    if not ok and purge:
        # bergson's resume checks the DIRECTORY, not file content -- purging only
        # gradients.bin leaves kfac_query/ "complete" and scoring runs gradient-less.
        d = os.path.dirname(p)
        os.rename(d, d + ".invalid")
        print(f"purged: {d} -> .invalid (resume will recompute the stage)")
    return ok


def check_scores(run):
    p = os.path.join(run, "scores", "scores.bin")
    if not os.path.exists(p):
        print(f"GATE FAIL scores: {p} missing")
        return False
    raw = np.memmap(p, dtype=np.uint8, mode="r")
    if raw.size % 8:
        print(f"GATE FAIL scores: size {raw.size} not a multiple of the 8-byte pair")
        return False
    pairs = raw.reshape(-1, 8)
    scores = pairs[:, :4].copy().view(np.float32).ravel()
    written = pairs[:, 4]
    s = sample(scores)
    w_ok = float(np.mean(sample(written) != 0))
    std, zeros = float(np.std(s)), float(np.mean(s == 0))
    finite = bool(np.isfinite(s).all())
    ok = finite and std > 1e-12 and zeros < 0.9 and w_ok == 1.0
    print(f"scores: std={std:.3e} zeros={zeros:.3f} finite={finite} "
          f"written={w_ok:.3f} -> " + ("PASS" if ok else "GATE FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["inverse", "scores", "all"])
    ap.add_argument("run_dir")
    ap.add_argument("--purge", action="store_true")
    a = ap.parse_args()
    ok = True
    if a.stage in ("inverse", "all"):
        ok &= check_inverse(a.run_dir, purge=a.purge)
    if a.stage in ("scores", "all"):
        ok &= check_scores(a.run_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
