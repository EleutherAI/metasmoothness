#!/usr/bin/env python3
"""Parameter-update norms for each run: ||theta_final - theta_0|| in L1 and L2.

Fills the delta_l1 / delta_l2 columns of experiments.csv, which have been empty for the
whole paper grid since the legacy eps-root-damping family was excluded (2026-08-22) --
the only rows that ever carried them. Reads each run's own checkpoints/step_0.ckpt as the
baseline rather than stock gpt2, so the number is the true update applied by training
(correct also for gpt2-medium and for the logit-scale rows).

    python param_delta.py RUN_DIR [RUN_DIR ...]
    python param_delta.py --csv experiments.csv        # every row with a run_dir

Also reports rel_l2 = ||delta||_2 / ||theta_0||_2, the scale-free version, which is what
you want when comparing across model sizes. Stdlib + torch; loads on CPU, writes nothing.
"""
import argparse, csv, re, sys
from pathlib import Path

import torch
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys


def ckpt_steps(ckpt_dir: Path):
    """(init_ckpt, final_ckpt) by step number parsed from step_<N>.ckpt names."""
    found = {}
    for p in ckpt_dir.glob("step_*.ckpt"):
        m = re.fullmatch(r"step_(\d+)\.ckpt", p.name)
        if m:
            found[int(m.group(1))] = p
    if 0 not in found:
        raise FileNotFoundError(f"{ckpt_dir}: no step_0.ckpt (need the init as baseline)")
    if len(found) < 2:
        raise FileNotFoundError(f"{ckpt_dir}: only step_0 present, no trained checkpoint")
    return found[0], found[max(found)], max(found)


def load_params(ckpt: Path):
    """Model parameters only.

    Both step_0 and the final checkpoint carry optimizer state (D8:
    save_optimizer_state=last) under `opt_state/...` plus rng/batch bookkeeping under
    leading-dot keys. They are float tensors present in BOTH checkpoints, so a plain
    key intersection does not drop them and the norm silently picks up the optimizer
    moments -- roughly 2/3 of the keys are opt_state.
    """
    sd = _load_state_dict_from_keys(checkpoint_id=str(ckpt))
    return {k: v for k, v in sd.items()
            if isinstance(v, torch.Tensor) and v.is_floating_point()
            and not k.startswith(("opt_state/", "."))}


def delta_norms(run_dir: Path):
    init, final, step = ckpt_steps(run_dir / "checkpoints")
    a, b = load_params(init), load_params(final)
    keys = sorted(set(a) & set(b))
    if not keys:
        raise ValueError(f"{run_dir}: no shared float params between step_0 and step_{step}")
    skipped = (set(a) ^ set(b))
    l1 = l2sq = base_sq = 0.0
    for k in keys:
        d = (b[k].float() - a[k].float()).flatten()
        l1 += d.abs().sum().item()
        l2sq += d.pow(2).sum().item()
        base_sq += a[k].float().pow(2).sum().item()
    l2 = l2sq ** 0.5
    return dict(step=step, n_tensors=len(keys), skipped=sorted(skipped),
                delta_l1=l1, delta_l2=l2, rel_l2=l2 / base_sq ** 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="*", type=Path)
    ap.add_argument("--csv", type=Path, help="read run_dir from every row of this CSV")
    a = ap.parse_args()

    targets = list(a.run_dirs)
    labels = {}
    if a.csv:
        for r in csv.DictReader(open(a.csv, newline="")):
            d = r.get("run_dir", "").strip()
            if d and Path(d).is_dir():
                targets.append(Path(d))
                labels[str(Path(d))] = r["run_id"]
    if not targets:
        sys.exit("no run dirs to process")

    print(f"{'run':34s} {'step':>5s} {'delta_l1':>12s} {'delta_l2':>10s} {'rel_l2':>8s}")
    for d in targets:
        name = labels.get(str(d), d.name)
        try:
            n = delta_norms(d)
        except Exception as e:
            print(f"{name:34s} {'-':>5s} {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if n["skipped"]:
            print(f"  note {name}: keys in only one checkpoint: {n['skipped']}", file=sys.stderr)
        print(f"{name:34s} {n['step']:5d} {n['delta_l1']:12.2f} {n['delta_l2']:10.4f} {n['rel_l2']:8.5f}")


if __name__ == "__main__":
    main()
