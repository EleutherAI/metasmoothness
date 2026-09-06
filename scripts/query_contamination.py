"""Does each query's own text appear in the training pool?

query_20 chunks were excluded from the train chain by index, but the corpus was
re-chunked, so a query's text can still sit inside a training chunk at a different
offset (README 2026-09-04). This measures that directly: for each query, take K
probes of P consecutive tokens and report the fraction found verbatim in train_N.

    python scripts/query_contamination.py --n 32000 64000 --sets query_20 query_20_heldout
"""
import argparse
import numpy as np
from datasets import load_from_disk

D = "/mnt/ssd-2/lucia/datasets_local"
BASE = np.uint64(1000003)
BLOCK = 4_000_000


def probes(qds, k, p):
    """-> (hashes, list of (query_idx, probe token array))"""
    out = []
    for i, row in enumerate(qds):
        ids = np.asarray(row["input_ids"], dtype=np.int64)
        for off in np.linspace(0, len(ids) - p, k, dtype=int):
            out.append((i, ids[off:off + p]))
    return np.array([roll(t.reshape(1, -1))[0] for _, t in out], dtype=np.uint64), out


def roll(x):
    """Polynomial hash of each length-P window; x is (rows, P) or a 1-D stream handled by caller."""
    h = np.zeros(x.shape[0], dtype=np.uint64)
    for j in range(x.shape[1]):
        h = h * BASE + x[:, j].astype(np.uint64)
    return h


def stream_hashes(tok, p):
    """Rolling hashes of every length-p window of a 1-D token array."""
    n = len(tok) - p + 1
    h = np.zeros(n, dtype=np.uint64)
    t = tok.astype(np.uint64)
    for j in range(p):
        h = h * BASE + t[j:j + n]
    return h


def main(ns, sets, k, p):
    for name in sets:
        qds = load_from_disk(f"{D}/{name}.hf")
        ph, plist = probes(qds, k, p)
        pset = set(ph.tolist())
        print(f"\n=== {name}: {len(qds)} queries x {k} probes of {p} tokens ===")
        for n in ns:
            train = load_from_disk(f"{D}/train_{n // 1000}k.hf")
            found = np.zeros(len(plist), dtype=bool)
            carry = np.empty(0, dtype=np.int64)
            buf = []
            size = 0
            for row in train:
                buf.append(np.asarray(row["input_ids"], dtype=np.int64))
                size += len(buf[-1])
                if size >= BLOCK:
                    carry = scan(np.concatenate([carry] + buf), ph, pset, plist, found, p)
                    buf, size = [], 0
            if buf:
                scan(np.concatenate([carry] + buf), ph, pset, plist, found, p)
            per_q = {}
            for (qi, _), f in zip(plist, found):
                per_q.setdefault(qi, []).append(f)
            frac = {qi: float(np.mean(v)) for qi, v in per_q.items()}
            hit = [qi for qi, f in frac.items() if f > 0]
            print(f"train_{n // 1000}k: {len(hit)}/{len(frac)} queries have text in train; "
                  f"mean probe-recall {np.mean(list(frac.values())):.2f}")
            print("   per query: " + " ".join(f"{qi}:{frac[qi]:.2f}" for qi in sorted(frac)))


def scan(tok, ph, pset, plist, found, p):
    h = stream_hashes(tok, p)
    cand = np.isin(h, ph)
    for pos in np.flatnonzero(cand):
        w = tok[pos:pos + p]
        for i, (_, probe) in enumerate(plist):
            if not found[i] and np.array_equal(w, probe):
                found[i] = True
    return tok[-(p - 1):]


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--n", type=int, nargs="+", default=[32000, 64000])
    a.add_argument("--sets", nargs="+", default=["query_20", "query_20_heldout"])
    a.add_argument("--probes", type=int, default=16)
    a.add_argument("--len", type=int, default=32)
    g = a.parse_args()
    main(g.n, g.sets, g.probes, g.len)
