#!/usr/bin/env python3
"""Generate reusable three-random control-bank configs for OOD filtering."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


RUNS = {
    4: "plan_adam_eps1e17_4k_bs256",
    8: "plan_adam_eps1e17_8k_bs256",
    16: "sm_adamw_eps1e17_16k_bs256",
    32: "plan_adam_eps1e17_32k_bs256",
    64: "plan_adam_eps1e17_64k_bs256",
    128: "plan_adam_eps1e17_128k_bs256",
}

MODEL_OVERRIDES = {
    64: "retrained.abandoned-bankbuild/base",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", type=int, nargs="+", required=True)
    parser.add_argument("--subset-start", type=int, default=0)
    parser.add_argument("--subset-stop", type=int, default=3)
    parser.add_argument("--config-suffix", default="bank")
    parser.add_argument(
        "--experiments",
        type=Path,
        default=Path("/mnt/ssd-2/lucia/paper_runs/experiments"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/ssd-2/lucia/ood_mean_scaling/random_banks"),
    )
    args = parser.parse_args()

    for scale in args.scales:
        source = args.experiments / RUNS[scale]
        source_config = source / "filter_proponents_ekfac" / "config.yaml"
        cfg = yaml.safe_load(source_config.read_text())
        validate = cfg["steps"][0]["validate"]

        run_dir = args.output_root / f"{scale}k"
        model = source / MODEL_OVERRIDES.get(scale, "base/model")
        scores = source / "ekfac_scores/scores"
        if not model.exists():
            raise FileNotFoundError(f"missing baseline model: {model}")
        if not scores.exists():
            raise FileNotFoundError(f"missing existing scores: {scores}")

        validate["run_path"] = str(run_dir)
        validate["distributed"]["nnode"] = 1
        validate["distributed"]["nproc_per_node"] = 8
        validate["distributed"]["node_rank"] = None
        # Keep the source query aligned with its existing per-query scores.
        # The saved leave-out models themselves are query-independent.
        validate["method"] = "lds"
        validate["scores"] = str(scores)
        validate["baseline_model"] = str(model)
        validate["retrained_dir"] = ""
        validate["num_subsets"] = 3
        validate["subset_start"] = args.subset_start
        validate["subset_stop"] = args.subset_stop
        validate["subset_fraction"] = 0.01
        validate["subsets"] = ""
        validate["save_models"] = True
        validate["save_mode"] = "interval"
        validate["save_interval"] = 1_000_000_000
        validate["resume"] = True
        validate["overwrite"] = False

        run_dir.mkdir(parents=True, exist_ok=True)
        output = run_dir / f"{args.config_suffix}.yaml"
        output.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"{scale}k: {output}")


if __name__ == "__main__":
    main()
