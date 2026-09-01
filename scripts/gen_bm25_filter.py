#!/usr/bin/env python3
import argparse
import copy
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_config  # noqa: E402

EXP = [
    Path("/mnt/ssd-2/lucia/paper_runs/experiments"),
    Path("/mnt/ssd-1/lucia/paper_runs/experiments"),
]
MIRROR = Path("/mnt/ssd-2/lucia/datasets_local")


def root_for(run_id: str) -> Path:
    for base in EXP:
        p = base / run_id
        if p.is_dir():
            return p
    raise SystemExit(f"run dir not found: {run_id}")


def mirror(entry):
    if not isinstance(entry, dict) or not entry.get("dataset"):
        return
    local = MIRROR / Path(entry["dataset"]).name
    if local.is_dir():
        entry["dataset"] = str(local)


def template_validate(root: Path, prefix: str):
    candidates = [
        root / f"{prefix}_ekfac.yaml",
        root / f"{prefix}_magic.yaml",
        root / "filter_proponents_ekfac.yaml",
        root / "filter_proponents_magic.yaml",
        root / "filter_top40_ekfac.yaml",
    ]
    for path in candidates:
        if path.is_file():
            cfg = yaml.safe_load(open(path))
            return copy.deepcopy(cfg["steps"][0]["validate"])

    exp = run_config.load(root)
    magic = copy.deepcopy(next(s["magic"] for s in exp["steps"] if "magic" in s))
    for key in (
        "run_path",
        "cleanup_ckpts",
        "dist",
        "double_backward_batch_size",
        "save_optimizer_state",
        "skip_validation",
        "train_mode",
        "scores",
    ):
        magic.pop(key, None)
    return magic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--fraction", type=float, default=None)
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--prefix", default="filter_proponents")
    ap.add_argument("--method", default="filter-proponents")
    ap.add_argument("--scores-name", default="bm25_scores")
    ap.add_argument("--out-source", default="bm25")
    ap.add_argument(
        "--num-subsets",
        type=int,
        default=0,
        help="0 means no random retrains; recover summaries from existing controls",
    )
    args = ap.parse_args()

    root = root_for(args.run_id)
    scores = root / args.scores_name
    if not (scores / "info.json").is_file():
        raise SystemExit(f"missing BM25 scores at {scores}")

    cfg = template_validate(root, args.prefix)
    fraction = args.fraction if args.fraction is not None else cfg.get("subset_fraction")
    if not fraction:
        raise SystemExit("no subset_fraction; pass --fraction")

    cfg.update(
        run_path=str(root / f"{args.prefix}_{args.out_source}"),
        scores=str(scores),
        method=args.method,
        subset_fraction=float(fraction),
        num_subsets=int(args.num_subsets),
        save_mode="interval",
        save_interval=10**9,
    )
    cfg.pop("retrained_dir", None)
    cfg["distributed"] = dict(
        cfg.get("distributed") or {}, nproc_per_node=args.nproc, nnode=1
    )
    for key in ("data", "query"):
        mirror(cfg.get(key))

    out = root / f"{args.prefix}_{args.out_source}.yaml"
    yaml.safe_dump(
        {"steps": [{"validate": cfg}], "run_path": cfg["run_path"]},
        open(out, "w"),
        sort_keys=False,
    )
    print(
        f"wrote {out} fraction={fraction} scores={scores} "
        f"num_subsets={args.num_subsets}"
    )


if __name__ == "__main__":
    main()
