from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt


def private_public_pie(stats: Dict[str, Any], output_dir: Path) -> Path:
    """
    Build a pie / donut chart showing public vs private vs unknown cameras.
    :param stats: Summary-stats dict coming from `compute_statistics`.
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
