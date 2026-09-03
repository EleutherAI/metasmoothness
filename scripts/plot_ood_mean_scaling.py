#!/usr/bin/env python3
"""Plot completed WikiText-103 and BioForget mean-query filtering sweeps."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("/mnt/ssd-2/lucia/ood_mean_scaling")
SCALES = (4, 8, 16, 32, 64, 128, 256, 512)


def summary_path(dataset: str, scale: int) -> Path:
    if dataset == "wikitext103" and scale == 128:
        return Path(
            "/mnt/ssd-2/lucia/wikitext103_ood/128k_filter_top1pct_mean/filter_summary.csv"
        )
    return ROOT / dataset / f"{scale}k/filter_top1pct/filter_summary.csv"


def load_curve(dataset: str) -> list[dict[str, float]]:
    rows = []
    for scale in SCALES:
        path = summary_path(dataset, scale)
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            row = next(csv.DictReader(handle))
        rows.append(
            {
                "scale": scale,
                "filter": float(row["filter_change"]),
                "random": float(row["random_mean"]),
                "random_sd": float(row["random_sd"]),
                "random_n": int(row["random_n"]),
                "rank": int(row["rank"]),
            }
        )
    return rows


def main() -> None:
    output = ROOT / "ood_mean_scaling_diagnostic.png"
    curves = {
        dataset: load_curve(dataset) for dataset in ("wikitext103", "bioforget")
    }
    table = ROOT / "ood_mean_scaling_results.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "dataset",
                "scale_k",
                "filter_change",
                "random_mean",
                "random_sd",
                "random_n",
                "rank",
            ),
        )
        writer.writeheader()
        for dataset, rows in curves.items():
            for row in rows:
                writer.writerow(
                    {
                        "dataset": dataset,
                        "scale_k": row["scale"],
                        "filter_change": row["filter"],
                        "random_mean": row["random"],
                        "random_sd": row["random_sd"],
                        "random_n": row["random_n"],
                        "rank": row["rank"],
                    }
                )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, dataset, title in zip(
        axes,
        ("wikitext103", "bioforget"),
        ("WikiText-103 (50-doc mean)", "BioForget (50-doc mean)"),
        strict=True,
    ):
        rows = curves[dataset]
        x = [row["scale"] for row in rows]
        ax.plot(x, [row["filter"] for row in rows], marker="o", label="Top 1%")
        ax.errorbar(
            x,
            [row["random"] for row in rows],
            yerr=[row["random_sd"] for row in rows],
            marker="o",
            capsize=3,
            label="Random 1% (n=3)",
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(x, [f"{scale}k" for scale in x])
        ax.set_title(title)
        ax.set_xlabel("Training documents")
        ax.set_ylabel("Mean query loss change")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    print(table)
    print(output)


if __name__ == "__main__":
    main()
