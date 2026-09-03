#!/usr/bin/env python3
"""Generate resumable EK-FAC mean-query score configs for OOD scaling runs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


RUNS = {
    4: "plan_adam_eps1e17_4k_bs256",
    8: "plan_adam_eps1e17_8k_bs256",
    16: "sm_adamw_eps1e17_16k_bs256",
    32: "plan_adam_eps1e17_32k_bs256",
    64: "plan_adam_eps1e17_64k_bs256",
    128: "plan_adam_eps1e17_128k_bs256",
    256: "plan_adam_eps1e17_256k_bs256",
    512: "plan_adam_eps1e17_512k_bs256",
}

MODEL_OVERRIDES = {
    64: "retrained.abandoned-bankbuild/base",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-name", required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--scales", type=int, nargs="+", required=True)
    parser.add_argument(
        "--experiments", type=Path, default=Path("/mnt/ssd-2/lucia/paper_runs/experiments")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("/mnt/ssd-2/lucia/ood_mean_scaling")
    )
    args = parser.parse_args()

    for scale in args.scales:
        source = args.experiments / RUNS[scale]
        source_config = source / "ekfac_scores" / "config.yaml"
        cfg = yaml.safe_load(source_config.read_text())
        ekfac = cfg["steps"][0]["ekfac"]

        run_dir = args.output_root / args.query_name / f"{scale}k"
        score_dir = run_dir / "ekfac_scores"
        model = source / MODEL_OVERRIDES.get(scale, "base/model")
        hessian = source / "ekfac_scores" / "hessian"
        if not model.exists():
            raise FileNotFoundError(f"missing model: {model}")
        if not (hessian / "kfac" / "total_processed.pt").exists():
            raise FileNotFoundError(f"incomplete Hessian: {hessian}")

        ekfac["index_cfg"]["run_path"] = str(score_dir)
        ekfac["index_cfg"]["model"] = str(model)
        ekfac["index_cfg"]["distributed"]["nnode"] = 1
        ekfac["index_cfg"]["distributed"]["nproc_per_node"] = 8
        ekfac["index_cfg"]["distributed"]["node_rank"] = None
        ekfac["hessian_pipeline_cfg"]["query"]["dataset"] = str(args.query)
        ekfac["hessian_pipeline_cfg"]["query"]["split"] = "train"
        ekfac["hessian_pipeline_cfg"]["query_aggregation"] = "mean"
        ekfac["hessian_pipeline_cfg"]["resume"] = True

        run_dir.mkdir(parents=True, exist_ok=True)
        score_dir.mkdir(parents=True, exist_ok=True)
        linked_hessian = score_dir / "hessian"
        if linked_hessian.is_symlink():
            if linked_hessian.resolve() != hessian.resolve():
                raise RuntimeError(f"wrong Hessian link: {linked_hessian}")
        elif linked_hessian.exists():
            raise FileExistsError(f"refusing to replace {linked_hessian}")
        else:
            os.symlink(hessian, linked_hessian)

        output = run_dir / "score.yaml"
        output.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"{scale}k: {output}")


if __name__ == "__main__":
    main()
