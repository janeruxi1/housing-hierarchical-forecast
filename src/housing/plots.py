"""Shared figure style and output helpers."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt

from housing.config import figures_dir, metrics_dir

COLORS = {"ours": "#1f5aa6", "zillow": "#d1495b", "grey": "#8c8c8c", "light": "#d9d9d9"}
REGIMES = [  # shaded on time-series figures
    ("2007-01-31", "2011-12-31", "Bust"),
    ("2020-06-30", "2022-06-30", "Pandemic boom"),
    ("2022-06-30", "2023-12-31", "Rate shock"),
]


def style() -> None:
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})


def shade_regimes(ax) -> None:
    import pandas as pd

    for a, b, label in REGIMES:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=COLORS["light"], alpha=0.5, lw=0)
        ax.text(pd.Timestamp(a), 1.0, f" {label}", transform=ax.get_xaxis_transform(), va="top", fontsize=8,
                color=COLORS["grey"])


def save(fig, name: str) -> None:
    d = figures_dir()
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.png", bbox_inches="tight")


def write_metrics(name: str, obj) -> None:
    d = metrics_dir()
    d.mkdir(parents=True, exist_ok=True)
    if hasattr(obj, "to_csv"):
        obj.to_csv(d / f"{name}.csv", index=False, float_format="%.6g")
    else:
        (d / f"{name}.json").write_text(json.dumps(obj, indent=2, default=float))
