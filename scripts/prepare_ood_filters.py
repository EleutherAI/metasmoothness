#!/usr/bin/env python3
"""Generate top-1% mean-query filter configs using reusable random banks."""

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
        "--experiments",
        type=Path,
        default=Path("/mnt/ssd-2/lucia/paper_runs/experiments"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/ssd-2/lucia/ood_mean_scaling"),
    )
    args = parser.parse_args()

    for scale in args.scales:
        source = args.experiments / RUNS[scale]
        source_config = source / "filter_proponents_ekfac/config.yaml"
        if scale == 128 and not source_config.exists():
            source_config = Path(
                "/mnt/ssd-2/lucia/wikitext103_ood/128k_filter_top1pct_mean/config.yaml"
            )
        elif scale >= 256 and not source_config.exists():
            source_config = source / "filter_proponents_ekfac_q0_2/config.yaml"
        cfg = yaml.safe_load(source_config.read_text())
        validate = cfg["steps"][0]["validate"]

        run_dir = args.output_root / args.query_name / f"{scale}k/filter_top1pct"
        score_run = args.output_root / args.query_name / f"{scale}k"
        if args.query_name == "wikitext103":
            score_run = args.output_root / "wikitext103_fixed" / f"{scale}k"
        scores = score_run / "ekfac_scores/scores"
        model = source / MODEL_OVERRIDES.get(scale, "base/model")
        if scale >= 128:
            bank = source / "bank_from_filter"
        else:
            bank = args.output_root / "random_banks" / f"{scale}k"

        validate["run_path"] = str(run_dir)
        validate["distributed"]["nnode"] = 1
        validate["distributed"]["nproc_per_node"] = 8
        validate["distributed"]["node_rank"] = None
        validate["query"]["dataset"] = str(args.query)
        validate["query"]["split"] = "train"
        validate["query_method"] = "mean"
        validate["method"] = "filter-proponents"
        validate["scores"] = str(scores)
        validate["baseline_model"] = str(model)
        validate["retrained_dir"] = str(bank)
        validate["num_subsets"] = 0
        validate["subset_fraction"] = 0.01
        validate["save_models"] = False
        validate["save_mode"] = "interval"
        validate["save_interval"] = 1_000_000_000
        validate["resume"] = False
        # The generated filter.yaml itself creates run_dir before Bergson starts.
        validate["overwrite"] = True

        run_dir.mkdir(parents=True, exist_ok=True)
        output = run_dir / "filter.yaml"
        output.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"{scale}k: {output}")


if __name__ == "__main__":
    main()
