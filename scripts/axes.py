#!/usr/bin/env python3
"""Which axis a row belongs to, whether it is cut, and how much we care.

Ruling (Lucia, 2026-08-25): weight decay, gradient clipping and logit scale are
CUT. Not deprioritised -- no further results are wanted for them, ever. GPU time
goes to batch scaling, and to as much token / step-count data at fixed 2 epochs
as we can get, including on larger models.

This exists because the opposite was happening: filter runs for wd0.0, wd0.1,
clip1.0 and scale0.5 were holding fourteen GPUs while the token ladder waited.
Encoding the ruling here means the queue tools can refuse a cut row instead of
each sweep re-deciding by hand.

    from axes import axis_of, is_cut, sort_key
    is_cut("plan_adam_eps1e17_16k_wd0.0")   -> True
    axis_of("plan_adam_eps1e17_32k_bs256")  -> ("token", 1)

Sort key is priority first, then LARGER N first: the unexplored end of the
ladder is the point.
"""

# axis -> (priority, why)
PRIORITY = {
    "token": (1, "N scaling at fixed batch and 2 epochs: the headline claim"),
    "steps": (2, "step count at fixed batch, incl. epochs -- the ms-collapse axis"),
    "model": (2, "model scale (gpt2-medium and up) -- same question, bigger model"),
    "batch": (3, "batch size at fixed N: the second axis of the grid"),
}

# Cut per the 2026-08-25 ruling, plus the axes already cut by D16 (architecture)
# and D14 (preact_batchnorm). A cut row is never queued for new measurement;
# results already recorded stay in the CSV.
CUT_MARKERS = {
    "wd0.": "weight decay -- measured as a null (three points spanning 0.0004)",
    "clip": "gradient clipping",
    "scale0.": "logit scale",
    "ckptavg": "checkpoint averaging -- needs an EK-FAC feature that does not exist",
    "arch_control": "architecture axis, cut by D16",
    "preact": "architecture axis, cut by D16",
    "qk_norm": "architecture axis, cut by D16",
}


def is_cut(run_id: str):
    """Return the reason a row is cut, or None if it is live."""
    for marker, why in CUT_MARKERS.items():
        if marker in run_id:
            return why
    return None


def axis_of(run_id: str):
    """Return (axis, priority). Cut rows come back as ("cut", 99)."""
    if is_cut(run_id):
        return "cut", 99
    r = run_id
    if "gpt2-medium" in r or "gpt2-large" in r:
        return "model", PRIORITY["model"][0]
    # "_ep" alone matches eps1e17, which put every bs256 token row in "steps".
    if "_ep4" in r or "_ep8" in r:
        return "steps", PRIORITY["steps"][0]
    if "_bs32" in r and any(f"_{n}k_" in r for n in (32, 64, 128, 256, 512)):
        return "steps", PRIORITY["steps"][0]   # fixed-batch step ladder
    if "_bs256" in r:
        return "token", PRIORITY["token"][0]
    if "_bs" in r:
        return "batch", PRIORITY["batch"][0]
    return "token", PRIORITY["token"][0]


def sort_key(run_id: str, n_docs=0):
    _, p = axis_of(run_id)
    return (p, -int(float(n_docs or 0)), run_id)


if __name__ == "__main__":
    import csv
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/mnt/ssd-2/lucia/metasmoothness/experiments.csv"
    rows = sorted(csv.DictReader(open(path)),
                  key=lambda r: sort_key(r["run_id"], r.get("n_docs", 0)))
    cur = None
    for r in rows:
        a, p = axis_of(r["run_id"])
        if a != cur:
            cur = a
            why = "no further results wanted" if a == "cut" else PRIORITY[a][1]
            print(f"\n[{p}] {a}: {why}")
        n = int(float(r["n_docs"] or 0))
        extra = f"  ({is_cut(r['run_id'])})" if a == "cut" else ""
        print(f"    {r['run_id']:<34} N={n:>7,}{extra}")
