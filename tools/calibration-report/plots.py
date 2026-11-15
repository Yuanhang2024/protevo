"""Editable, low-saturation line figures with trajectory-level sample counts."""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np

PALETTE = ("#537A96", "#A57668", "#719582", "#9882A5", "#B09B63", "#6F969A")


def render(points, checkpoints, profiles, destination, formats=("svg", "pdf", "png"), dpi=600):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#9CA1A5", "axes.linewidth": .7,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    models = list(dict.fromkeys(row["model"] for row in profiles))
    panels = [models] if len(models) <= 1 or any(row.get("arm") for row in profiles) else [[model] for model in models]
    columns = min(2, max(1, len(panels)))
    rows = math.ceil(max(1, len(panels)) / columns)
    saved = []
    metrics = (("mean_fitness", "Mean fitness of queried variants (S1)"),
               ("max_fitness", "Best newly queried fitness (S1)"),
               ("cumulative_best", "Best known fitness including initial data (S1)"))
    for metric, ylabel in metrics:
        fig, axes = plt.subplots(rows, columns, figsize=(6.0 * columns, 3.5 * rows + .7), squeeze=False)
        for panel_index, panel in enumerate(panels):
            ax = axes.flat[panel_index]
            selected_profiles = [profile for profile in profiles if profile["model"] in panel]
            varied = len({tuple(profile[field] for field in ("candidatesPerRound", "topK", "rounds", "oversizePolicy")) for profile in selected_profiles}) > 1
            for index, profile in enumerate(selected_profiles):
                key, color = profile["profile"], PALETTE[index % len(PALETTE)]
                curve = sorted([row for row in points if row["profile"] == key], key=lambda row: row["queries"])
                x = np.array([row["queries"] for row in curve], dtype=float)
                y = np.array([np.nan if row.get(f"{metric}_estimate") is None else row[f"{metric}_estimate"] for row in curve])
                low = np.array([np.nan if row.get(f"{metric}_lower") is None else row[f"{metric}_lower"] for row in curve])
                high = np.array([np.nan if row.get(f"{metric}_upper") is None else row[f"{metric}_upper"] for row in curve])
                label = str(profile.get("arm") or profile["effort"])
                if varied or len(selected_profiles) > 6:
                    label += f' C{profile["candidatesPerRound"]}/k{profile["topK"]}/r{profile["rounds"]} {profile["oversizePolicy"]}'
                label += f' (n={profile["completed_trajectories"]}/{profile["planned_trajectories"]})'
                style = "--" if profile["oversizePolicy"] == "first_n" else "-"
                ax.plot(x, y, color=color, marker="o", markersize=4, linewidth=1.5, linestyle=style, label=label, zorder=3)
                ax.fill_between(x, low, high, color=color, alpha=.12, linewidth=0, zorder=1)
                observations = [row for row in checkpoints if row["profile"] == key and row["complete"] and row["checkpoint_available"]]
                ax.scatter([row["queries"] for row in observations], [row.get(metric, np.nan) for row in observations],
                           color=color, alpha=.22, s=9, edgecolors="none", zorder=2)
            ax.set(title=" / ".join(panel), xlabel="Measured variants queried", ylabel=ylabel)
            ax.set_xticks([10, 20, 30])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.grid(axis="y", alpha=.18, linewidth=.6)
            ax.legend(frameon=False, fontsize=7, loc="best")
        for ax in list(axes.flat)[len(panels):]:
            ax.set_visible(False)
        confidence = profiles[0]["mean_fitness"]["confidence_level"] * 100 if profiles else 95
        fig.text(.01, .015, f'Lines: equal-split mean; bands: {confidence:g}% hierarchical bootstrap; small points: independent trajectories.\n'
                 'n = completed/planned trajectories. Query checkpoints on one trajectory are dependent.', fontsize=8, color="#565B60")
        fig.tight_layout(rect=(0, .07, 1, 1))
        for suffix in formats:
            path = destination / f"quality-{metric}.{suffix}"
            fig.savefig(path, dpi=dpi, facecolor="white", metadata={"Creator": "GB1 calibration report"})
            saved.append(path)
        plt.close(fig)
    return saved
