"""Emit the paper's lr-sweep appendix table, and gate on the tuning protocol.

The paper claims a specific selection procedure ("sweep three learning rates;
extend until the winner is interior"). This script is the thing that checks the
claim is true of the recorded data, rather than asking a reader to trust it.

    python scripts/lr_sweep_table.py            # audit only
    python scripts/lr_sweep_table.py --latex    # audit + emit the table body

Reads tuning.csv (gpt2 family) and qwen_tuning.csv (Qwen 2.5 1.5B). Exits
NON-ZERO if any group that feeds a paper row either
  (a) has a winner at an endpoint of its measured grid, by a margin over its
      interior neighbour that clears the tie threshold -- the protocol says add
      one more point outward and re-check before freezing; or
  (b) is frozen in experiments.csv at an lr whose held-out loss is worse than
      the winner's by more than the tie threshold.
TIE is 0.002 nats = 2x the measured seed noise (DECISIONS.md "Why steps of 2x
are the right spacing"). Within it the protocol deliberately keeps the group's
centre rather than the numerical winner, so an endpoint that only ties does NOT
require an extension and a centre that only ties is the correct selection.
Per CLAUDE.md: a script that only prints a number you should have reacted to is
a bug in the script, so these are failures, not warnings.

Groups still being measured (any empty lr cell) are reported as PENDING and do
not fail the build -- an endpoint win is not yet meaningful there. Single-lr
"control" groups inherit the anchor lr and are not sweeps; they are skipped.
"""

import argparse
import collections
import csv
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Two times the measured seed noise (~0.001 nats). Differences below this are
# not resolvable by the selection metric; the protocol's tie rule keeps the
# group's centre instead of chasing them.
TIE = 0.002


def load(path, model):
    rows = []
    for r in csv.DictReader(open(os.path.join(HERE, path))):
        r["_model"] = r.get("model") or model
        rows.append(r)
    return rows


def config_lr(target):
    """Qwen rows are not in experiments.csv; their frozen lr lives in the run
    config. Read it from there so those sweeps are gated too."""
    p = os.path.join(HERE, "configs", "experiments", target + ".yaml")
    if not os.path.exists(p):
        return None
    for line in open(p):
        if line.strip().startswith("lr:"):
            return {"lr": line.split(":", 1)[1].strip(), "status": "config"}
    return None


def groups(rows):
    g = collections.OrderedDict()
    for r in rows:
        g.setdefault(r["sweep_group"], []).append(r)
    for rs in g.values():
        rs.sort(key=lambda r: float(r["lr"]))
    return g


def analyse(rs):
    """-> (measured [(lr, ce)], winner lr|None, interior|None, pending, diverged)."""
    meas, pending, diverged = [], False, []
    for r in rs:
        h = (r.get("heldout_loss") or "").strip()
        lr = float(r["lr"])
        if r.get("status") == "diverged" or h == "nan":
            diverged.append(lr)
        elif h:
            meas.append((lr, float(h)))
        elif r.get("status") in ("cut", "blocked"):
            pass
        else:
            pending = True
    if not meas:
        return meas, None, None, pending, diverged
    win = min(meas, key=lambda t: t[1])[0]
    i = [lr for lr, _ in meas].index(win)
    # A diverged point outside the winner counts as a measured bound: it is
    # evidence the optimum is not further out, which is what interiority asks.
    hi = max([lr for lr, _ in meas] + diverged)
    lo = min([lr for lr, _ in meas] + diverged)
    interior = lo < win < hi
    return meas, win, interior, pending, diverged


def targets(rs):
    out = set()
    for r in rs:
        for t in (r.get("selects_lr_for") or "").replace(" ", ",").split(","):
            if t:
                out.add(t)
    return sorted(out)


def fmt(lr):
    return f"{lr:g}".replace("e-0", "e-").replace("e-", r"\mathrm{e}{-}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true", help="emit the table body")
    args = ap.parse_args()

    exp = {r["run_id"]: r for r in csv.DictReader(open(os.path.join(HERE, "experiments.csv")))}
    all_rows = load("tuning.csv", "gpt2") + load("qwen_tuning.csv", "qwen2.5-1.5b")

    fails, lines = [], []
    for name, rs in groups(all_rows).items():
        if len(rs) == 1:
            continue  # control group, inherits the anchor lr
        meas, win, interior, pending, div = analyse(rs)
        if not meas:
            print(f"  SKIP    {name:34s} (no measurements)")
            continue
        grid = ", ".join(f"{lr:g}" for lr, _ in meas) + (
            "".join(f", {lr:g}(div)" for lr in sorted(div)) if div else "")
        if pending:
            # A still-incomplete grid cannot decide interiority, but a target
            # already trained at an lr that is not a point of the grid at all
            # was never selected by this sweep. That is the silent failure.
            planned = {float(r["lr"]) for r in rs}
            for t in targets(rs):
                e = exp.get(t) or config_lr(t) or {}
                el = e.get("lr", "").strip()
                if el and not any(abs(float(el) - g) / g < 1e-6 for g in planned):
                    fails.append(f"{name}: {t} is frozen at lr={el}, which is not a "
                                 f"point of its grid [{grid}] (status={e.get('status')})")
            print(f"  PENDING {name:34s} best-so-far {win:g}  [{grid}]")
            continue
        ce = dict(meas)
        best = ce[win]
        ranked = sorted(meas, key=lambda t: t[1])
        margin = ranked[1][1] - best if len(ranked) > 1 else float("inf")
        tag = "OK   "
        if not interior:
            if margin >= TIE:
                fails.append(f"{name}: winner {win:g} is an endpoint of [{grid}] and beats "
                             f"the next point by {margin:.4f} >= {TIE} -- protocol requires "
                             f"one more point outward")
            else:
                tag = "TIE  "  # endpoint win inside seed noise; no extension owed
        for t in targets(rs):
            e = exp.get(t) or config_lr(t)
            if e is None:
                continue  # target lives in london.csv, audited there
            el = (e.get("lr") or "").strip()
            if not el:
                fails.append(f"{name}: {t} has no frozen lr")
                continue
            frozen = float(el)
            hit = next((c for lr, c in meas if abs(lr - frozen) / frozen < 1e-6), None)
            if hit is None:
                fails.append(f"{name}: {t} is frozen at lr={el}, which is not a measured "
                             f"point of [{grid}] (status={e.get('status')})")
            elif hit - best >= TIE:
                fails.append(f"{name}: {t} trained at lr={el} (heldout {hit:.4f}) but the "
                             f"winner {win:g} is {hit - best:.4f} better -- exceeds the "
                             f"{TIE} tie threshold (status={e.get('status')})")
            elif abs(frozen - win) / win > 1e-6:
                tag = "TIE  "  # centre kept over a numerical winner inside seed noise
        print(f"  {tag}   {name:34s} winner {win:g}  [{grid}]")
        lines.append((rs[0]["_model"], rs[0].get("optimizer", ""), name, meas, win, div))

    if args.latex:
        print("\n% --- generated by scripts/lr_sweep_table.py; do not hand-edit ---")
        for model, opt, name, meas, win, div in lines:
            grid = ", ".join(
                (r"\mathbf{%s}" % fmt(lr)) if lr == win else fmt(lr)
                for lr, _ in meas)
            grid += "".join(f", {fmt(lr)}^{{\\dagger}}" for lr in sorted(div))
            print(rf"{name.replace('_', chr(92) + '_')} & {opt} & ${grid}$ & "
                  rf"${fmt(win)}$ & {min(c for _, c in meas):.4f} \\")

    if fails:
        print("\nTUNING-PROTOCOL VIOLATIONS", file=sys.stderr)
        for f in fails:
            print("  " + f, file=sys.stderr)
        sys.exit(1)
    print("\nall frozen sweeps interior and consistent with experiments.csv")


if __name__ == "__main__":
    main()
