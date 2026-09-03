#!/usr/bin/env python3
"""Prepare a small two-shard versus eight-shard EK-FAC score comparison."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from datasets import load_from_disk


ROOT = Path("/mnt/ssd-2/lucia/ood_mean_scaling/reshard_validation/wikitext103_512k")
SOURCE_CONFIG = Path(
    "/mnt/ssd-2/lucia/ood_mean_scaling/wikitext103_fixed/512k/score.yaml"
)
SOURCE_DATA = Path("/mnt/ssd-2/lucia/datasets_local/train_512k.hf")
ORIGINAL_HESSIAN = Path(
    "/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_512k_bs256/ekfac_scores/hessian"
)
RESHARDED_HESSIAN = Path("/mnt/ssd-2/lucia/ood_mean_scaling/hessians_8/512k")


def main() -> None:
    data_path = ROOT / "train_first4096.hf"
    if not data_path.exists():
        load_from_disk(str(SOURCE_DATA)).select(range(4096)).save_to_disk(str(data_path))

    source = yaml.safe_load(SOURCE_CONFIG.read_text())
    for name, nproc, hessian in (
        ("two_shard", 2, ORIGINAL_HESSIAN),
        ("eight_shard", 8, RESHARDED_HESSIAN),
    ):
        cfg = yaml.safe_load(yaml.safe_dump(source))
        ekfac = cfg["steps"][0]["ekfac"]
        run_dir = ROOT / name / "ekfac_scores"
        ekfac["index_cfg"]["run_path"] = str(run_dir)
        ekfac["index_cfg"]["data"]["dataset"] = str(data_path)
        ekfac["index_cfg"]["distributed"]["nproc_per_node"] = nproc
        ekfac["hessian_pipeline_cfg"]["resume"] = True
        run_dir.mkdir(parents=True, exist_ok=True)
        link = run_dir / "hessian"
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            raise FileExistsError(link)
        os.symlink(hessian, link)
        output = ROOT / name / "score.yaml"
        output.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(output)


if __name__ == "__main__":
    main()
