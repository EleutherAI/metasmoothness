#!/usr/bin/env python3
"""Companions to figures/filter_scaling.pdf, on the same rows and the same
two panels (top 1% / top 40):

  filter_scaling_absolute.pdf    the three query losses the QLD is a difference
                                 of -- unfiltered, after the random-control
                                 retrains (baseline + random_mean, see
                                 scripts/absolute_losses.py), and with the EK-FAC
                                 proponents removed -- each with the bootstrap
                                 95% CI over the same 20 queries, and the
                                 pretrained GPT-2 loss as a dotted reference.
  filter_scaling_relative.pdf    the filter's cost as a percent of the matched
                                 random control's loss, mean_q (filtered - random) /
                                 random, bootstrap CI over the same queries
                                 (absolute_losses.relative_point).

The absolute panels carry a much wider bar than the QLD figure does from the very
same data: they resample the raw losses, and queries differ from one another by
~1.4 nats where a filter moves one by ~0.1, so the interval is dominated by
between-query spread and is near-identical on all three series (they correlate
0.99 across queries). The QLD and relative figures difference that spread away.
"""

import argparse
import os
import pathlib

import matplotlib.pyplot as plt

import absolute_losses as absl


ROOT = pathlib.Path(__file__).resolve().parent.parent
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
BLUE = "#2a78d6"
PANELS = (("filter_proponents_ekfac", "(a) Top 1% of documents removed"),
          ("filter_top40_ekfac", "(b) Top 40 documents removed"))
tokens = lambda n: 2 * n * 512


def series(subdir):
    return [absl.point(RUNS[n], subdir, ci=absl.boot_ci) for n in NS]


def style(ax, title, full_x=False):
    ticks = [tokens(n) for n in NS]
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks, [f"{x / 1e6:.0f}M" for x in ticks])
    # The normalized panels drop rungs; hold the axis to the full ladder so a
    # missing 4M/8M point reads as absent rather than as a shorter x range.
    if full_x:
        ax.set_xlim(ticks[0] / 1.35, ticks[-1] * 1.35)
    ax.minorticks_off()
    ax.set_xlabel("Number of training tokens")
    ax.set_title(title, fontsize=10)
    ax.grid(color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.margins(x=0.09)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=pathlib.Path,
                        default=pathlib.Path(os.environ.get("FIGURES_DIR") or ROOT / "figures"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=200, sharey=True)
    for ax, (subdir, title) in zip(axes, PANELS):
        points = series(subdir)
        ax.axhline(PRETRAINED_GPT2_LOSS, color="#999999", linestyle=":", linewidth=2,
                   label="Pretrained GPT-2")
        absl.draw(ax, [tokens(n) for n in NS], points, BLUE, neutral=True, labels=True)
        style(ax, title)
        for n, p in zip(NS, points):
            if p:
                print(f"{title[:3]} N={n // 1000:>3}k n={p['n']:2d} unfiltered={p['unfiltered'][0]:.4f} "
                      f"random={p['random'][0]:.4f} filtered={p['filtered'][0]:.4f}")
            else:
                print(f"{title[:3]} N={n // 1000:>3}k no per-query losses ({RUNS[n]}/{subdir})")
    axes[0].set_ylabel(absl.label("Mean query loss"))
    absl.legend_above(fig, axes[0].get_legend_handles_labels()[0], 4)
    path = args.outdir / "filter_scaling_absolute.pdf"
    if absl.WRITE_ABSOLUTE:
        fig.savefig(path)
        print(f"wrote {path}")
    else:
        print(f"not written (absolute_losses.WRITE_ABSOLUTE is False): {path}")

    # Relative: the same effect as a percent of its random control's loss.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=200, sharey=True)
    for ax, (subdir, title) in zip(axes, PANELS):
        xs, ys, lo, hi = [], [], [], []
        for n in NS:
            p = absl.relative_point(RUNS[n], subdir)
            if p:
                xs.append(tokens(n)); ys.append(p[0]); lo.append(p[1]); hi.append(p[2])
                print(f"{title[:3]} N={n // 1000:>3}k relative={p[0]:5.2f}% [-{p[1]:.2f} +{p[2]:.2f}]")
            else:
                print(f"{title[:3]} N={n // 1000:>3}k no per-query losses ({RUNS[n]}/{subdir})")
        ax.errorbar(xs, ys, yerr=[lo, hi], color=BLUE, marker="o", markersize=5,
                    linewidth=2, capsize=3, capthick=1.2, label="EK-FAC proponents")
        style(ax, title)
    axes[0].set_ylabel(absl.label("Query loss increase (%)"))
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    path = args.outdir / "filter_scaling_relative.pdf"
    if absl.WRITE_RELATIVE:
        fig.savefig(path)
        print(f"wrote {path}")
    else:
        print(f"not written (absolute_losses.WRITE_RELATIVE is False): {path}")


if __name__ == "__main__":
    main()
