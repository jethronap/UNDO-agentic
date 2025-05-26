from pathlib import Path
from typing import Dict, Any
from collections import Counter

import matplotlib.pyplot as plt


def private_public_pie(stats: Dict[str, Any], output_dir: Path) -> Path:
    """
    Build a pie / donut chart showing public vs private vs unknown cameras.
    :param stats: Summary-stats dict coming from `compute_statistics`
    :param output_dir: Directory to save the chart
    :return: A Path to the png file
    """

    total = stats["total"]
    public = stats["public_count"]
    private = stats["private_count"]
    unknown = total - public - private

    vals = [public, private, unknown]
    labels = ["Public", "Private", "Unknown"]

    fig, ax = plt.subplots(figsize=(6, 6))
    # Donut pie
    wedges, texts, autotexts = ax.pie(
        vals,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.4),
    )
    ax.set(aspect="equal", title="Camera Privacy Distribution")

    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / "privacy_distribution.png"
    fig.savefig(chart_path, bbox_inches="tight")
    plt.close(fig)

    return chart_path


def plot_zone_sensitivity(
    stats: Dict[str, Any],
    output_dir: Path,
    top_n: int = 10,
    filename: str = "zone_sensitivity.png",
) -> Path:
    """
    Build and save a stacked‐bar chart showing counts of sensitive vs non‐sensitive cameras per zone.
    :param stats: Summary-stats dict coming from `compute_statistics`
    :param output_dir: Directory to save the chart
    :param top_n: Number of zones to be plotted
    :param filename: The filename to save the chart with
    :return: Path to the png file
    """

    zone_totals: Counter = stats["zone_counts"]
    zone_sens: Counter = stats["zone_sensitivity_counts"]
    # build non‐sensitive = total − sensitive
    zone_nonsens = {z: zone_totals[z] - zone_sens.get(z, 0) for z in zone_totals}

    # pick top N zones by total cameras
    top = [z for z, _ in zone_totals.most_common(top_n)]
    other = set(zone_totals) - set(top)
    labels = top.copy()
    sens_vals = [zone_sens.get(z, 0) for z in top]
    nonsens_vals = [zone_nonsens[z] for z in top]

    if other:
        labels.append("other")
        sens_vals.append(sum(zone_sens[z] for z in other))
        nonsens_vals.append(sum(zone_nonsens[z] for z in other))

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.5)))
    y = list(range(len(labels)))
    ax.barh(y, nonsens_vals, label="Non-sensitive")
    ax.barh(y, sens_vals, left=nonsens_vals, label="Sensitive")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Camera count")
    ax.set_title("Sensitive vs non-sensitive cameras by zone")
    ax.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
