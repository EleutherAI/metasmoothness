#!/usr/bin/env python3
"""Plot absolute query losses underlying the proponent-filter delta figures."""

import argparse
import csv
import os
import pathlib
import re
import statistics

import matplotlib.pyplot as plt


ROOT = pathlib.Path(__file__).resolve().parent.parent
RUN_ROOTS = (
    pathlib.Path("/mnt/ssd-2/lucia/paper_runs/experiments"),
    pathlib.Path("/mnt/ssd-1/lucia/paper_runs/experiments"),
)
NS = (4_000, 8_000, 16_000, 32_000, 64_000, 128_000, 256_000, 512_000)
RUNS = {
    4_000: "plan_adam_eps1e17_4k_bs256",
    8_000: "plan_adam_eps1e17_8k_bs256",
    16_000: "sm_adamw_eps1e17_16k_bs256",
    32_000: "plan_adam_eps1e17_32k_bs256",
    64_000: "plan_adam_eps1e17_64k_bs256",
    128_000: "plan_adam_eps1e17_128k_bs256",
    256_000: "plan_adam_eps1e17_256k_bs256",
    512_000: "plan_adam_eps1e17_512k_bs256",
}
PRETRAINED_GPT2_LOSS = 3.449889528751373
COLORS = {"ekfac": "#2a78d6", "magic": "#1baf7a", "bm25": "#5c5c5c"}


def run_path(run_id):
    return next((root / run_id for root in RUN_ROOTS if (root / run_id).exists()), None)


def read_filter_rows(run_id, subdir):
    """Read a canonical result, or reconstruct it from non-overlapping query shards."""
    root = run_path(run_id)
    if root is None:
        return []
    canonical = root / subdir / "filter_proponents.csv"
    if canonical.is_file():
        rows = list(csv.DictReader(canonical.open()))
        if len(rows) >= 20:
            return rows[:20]

    by_query = {}
    shard_re = re.compile(r"_q(\d+)_(\d+)(?:$|_)")
    for path in sorted(root.glob(f"{subdir}_q*/filter_proponents.csv")):
        match = shard_re.search(path.parent.name)
        if not match:
            continue
        start, end = map(int, match.groups())
        for row in csv.DictReader(path.open()):
            local_query = int(row["query"])
            global_query = start + local_query
            if start <= global_query < end:
                by_query[global_query] = row
    return [by_query[q] for q in sorted(by_query)]


def point(run_id, subdir):
    rows = read_filter_rows(run_id, subdir)
    if not rows:
        return None
    return {
        "n_queries": len(rows),
        "unfiltered": statistics.fmean(float(row["baseline_loss"]) for row in rows),
        "filtered": statistics.fmean(float(row["filtered_loss"]) for row in rows),
    }


def series(subdir):
    return [point(RUNS[n], subdir) for n in NS]


def style(ax, title):
    tokens = [2 * n * 512 for n in NS]
    ax.set_xscale("log", base=2)
    ax.set_xticks(tokens, [f"{x / 1e6:.0f}M" for x in tokens])
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.minorticks_off()
    ax.set_xlabel("Number of training tokens")
    ax.set_title(title, fontsize=10)
    ax.grid(color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.margins(x=0.08)


def draw(ax, points, color):
    xs = [2 * n * 512 for n, p in zip(NS, points) if p is not None]
    unfiltered = [p["unfiltered"] for p in points if p is not None]
    filtered = [p["filtered"] for p in points if p is not None]
    ax.axhline(
        PRETRAINED_GPT2_LOSS,
        color="#999999",
        linestyle=":",
        linewidth=2,
        label="Pretrained GPT-2",
    )
    ax.plot(
        xs,
        unfiltered,
        color="#222222",
        linestyle=":",
        marker="o",
        markersize=4,
        linewidth=2,
        label="Unfiltered trained",
    )
    ax.plot(
        xs,
        filtered,
        color=color,
        marker="o",
        markersize=5,
        linewidth=2,
        label="Filtered",
    )


def save(fig, outdir, name):
    fig.tight_layout()
    path = outdir / name
    fig.savefig(path)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "figures")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    top1 = series("filter_proponents_ekfac")
    top40 = series("filter_top40_ekfac")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=200, sharey=True)
    for ax, points, title in zip(
        axes,
        (top1, top40),
        ("Top 1% of documents removed", "Top 40 documents removed"),
    ):
        draw(ax, points, COLORS["ekfac"])
        style(ax, title)
    axes[0].set_ylabel("Mean query loss")
    axes[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), ncols=3)
    save(fig, args.outdir, "filter_scaling_absolute.png")

    methods = (
        ("EK-FAC", "filter_proponents_ekfac", COLORS["ekfac"]),
        ("MAGIC", "filter_proponents_magic", COLORS["magic"]),
        ("BM25", "filter_proponents_bm25", COLORS["bm25"]),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=200, sharey=True)
    rows = []
    for ax, (label, subdir, color) in zip(axes, methods):
        points = series(subdir)
        draw(ax, points, color)
        style(ax, label)
        for n, p in zip(NS, points):
            if p:
                rows.append({"method": label, "n_docs": n, **p})
    axes[0].set_ylabel("Mean query loss")
    axes[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), ncols=3)
    save(fig, args.outdir, "filter_method_absolute.png")

    table = args.outdir / "filter_absolute_losses.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("method", "n_docs", "n_queries", "unfiltered", "filtered")
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {table}")


if __name__ == "__main__":
    main()
