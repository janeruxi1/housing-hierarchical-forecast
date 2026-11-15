"""Does summing counties reproduce Zillow's own metro, state and national ZHVI?

Our nodes are housing-unit-weighted averages of county ZHVI over the counties with a full
history. Zillow's are value-weighted over every home it has a Zestimate for. The two can differ
in level (different weights, coverage) while agreeing on growth, which is what we forecast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from housing.config import STATES
from housing.hierarchy import level_of

STATE_NAME_TO_ABBR = {name: abbr for abbr, name in STATES.values()}


def published_ids(pub: pd.DataFrame) -> pd.DataFrame:
    """Map Zillow's published regions to our node ids."""
    p = pub.copy()
    p["unique_id"] = np.select(
        [p["region_type"].eq("country"), p["region_type"].eq("msa"), p["region_type"].eq("state")],
        ["US", "MSA:" + p["region"], "ST:" + p["region"].map(STATE_NAME_TO_ABBR).fillna("?")],
        default="?",
    )
    return p[p["unique_id"] != "?"]


def coherence_gap(Y: pd.DataFrame, pub: pd.DataFrame, coverage: pd.DataFrame, nodes: pd.DataFrame,
                  min_coverage: float = 0.95, since: str = "2008-01-31") -> pd.DataFrame:
    """Level gap and growth agreement, per node with >= min_coverage of its housing units in the panel.

    The nation is always included, whatever its coverage, and `n_counties` flags single-county
    metros, where the rebuilt series is Zillow's own county series and agreement is automatic.
    """
    ours = Y[["unique_id", "ds", "typical_value"]].rename(columns={"ds": "date"})
    theirs = published_ids(pub)[["unique_id", "date", "zhvi"]]
    m = ours.merge(theirs, on=["unique_id", "date"]).merge(coverage, on="unique_id")
    m = m[((m["hu_coverage"] >= min_coverage) | (m["unique_id"] == "US")) & (m["date"] >= since)]
    m = m.sort_values(["unique_id", "date"])
    m["g_ours"] = m.groupby("unique_id")["typical_value"].transform(lambda s: np.log(s).diff())
    m["g_theirs"] = m.groupby("unique_id")["zhvi"].transform(lambda s: np.log(s).diff())
    m["g12_ours"] = m.groupby("unique_id")["typical_value"].transform(lambda s: np.log(s).diff(12))
    m["g12_theirs"] = m.groupby("unique_id")["zhvi"].transform(lambda s: np.log(s).diff(12))

    def per_node(g: pd.DataFrame) -> pd.Series:
        last = g.iloc[-1]
        d = g.dropna(subset=["g_ours", "g_theirs"])
        d12 = g.dropna(subset=["g12_ours", "g12_theirs"])
        return pd.Series({
            "hu_coverage": last["hu_coverage"],
            "level_gap_pct_last": 100 * (last["typical_value"] / last["zhvi"] - 1),
            "growth_corr_1m": d["g_ours"].corr(d["g_theirs"]),
            "growth_rmse_1m_pp": 100 * np.sqrt(np.mean((d["g_ours"] - d["g_theirs"]) ** 2)),
            "growth_rmse_12m_pp": 100 * np.sqrt(np.mean((d12["g12_ours"] - d12["g12_theirs"]) ** 2)),
        })

    out = m.groupby("unique_id").apply(per_node, include_groups=False).reset_index()
    out.insert(1, "level", level_of(out["unique_id"]))
    counts = pd.concat([nodes[lv].value_counts() for lv in ("State", "Metro")])
    out.insert(2, "n_counties", out["unique_id"].map(counts).fillna(len(nodes)).astype(int))
    return out
