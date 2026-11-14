"""The forecast hierarchy: county at the bottom, grouped two ways (by state and by metro).

ZHVI is a value-weighted *average*, so county ZHVIs do not add up to a metro ZHVI. What does add
up is the value of the housing stock: county ZHVI x county housing units. Every node in the
hierarchy is therefore a total ($), and its typical home value is total / housing units.
Housing units are held fixed at their latest Census estimate, so the totals move only with prices.

Node ids:  "US" | "ST:WA" | "MSA:Seattle, WA" | "MSA:Non-metro WA" | "CTY:53033"
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEVELS = ("National", "State", "Metro", "County")


def node_table(counties: pd.DataFrame) -> pd.DataFrame:
    """One row per county with the ids of its parents. Needs columns fips, state, metro."""
    c = counties[["fips", "state", "metro"]].drop_duplicates("fips").sort_values("fips").reset_index(drop=True)
    metro = np.where(c["metro"].fillna("").eq(""), "Non-metro " + c["state"], c["metro"])
    return pd.DataFrame({
        "County": "CTY:" + c["fips"],
        "State": "ST:" + c["state"],
        "Metro": "MSA:" + pd.Series(metro),
        "National": "US",
    })


def summing_matrix(nodes: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """S_df (unique_id + one 0/1 column per county) and tags, in hierarchicalforecast's format.

    The structure is *grouped*: metros cross state lines, so State and Metro are two separate
    groupings of the same counties rather than one tree.
    """
    bottom = nodes["County"].to_numpy()
    tags = {
        "National": np.array(["US"]),
        "State": np.sort(nodes["State"].unique()),
        "Metro": np.sort(nodes["Metro"].unique()),
        "County": bottom,
    }
    rows = []
    for level in LEVELS:
        for node in tags[level]:
            if level == "National":
                rows.append(np.ones(len(bottom)))
            else:
                rows.append((nodes[level].to_numpy() == node).astype(float))
    S = pd.DataFrame(np.vstack(rows), columns=bottom)
    S.insert(0, "unique_id", np.concatenate([tags[lv] for lv in LEVELS]))
    return S, tags


def aggregate(bottom_df: pd.DataFrame, S_df: pd.DataFrame, value: str = "y") -> pd.DataFrame:
    """Sum a bottom-level long table (unique_id, ds, value) up every node of S. Returns long."""
    wide = bottom_df.pivot(index="ds", columns="unique_id", values=value)
    bottom_ids = list(S_df.columns[1:])
    wide = wide[bottom_ids]
    agg = wide.to_numpy() @ S_df[bottom_ids].to_numpy().T
    out = pd.DataFrame(agg, index=wide.index, columns=S_df["unique_id"])
    return out.reset_index().melt(id_vars="ds", var_name="unique_id", value_name=value)[["unique_id", "ds", value]]


def level_of(unique_id: pd.Series) -> pd.Series:
    prefix = unique_id.str.split(":").str[0]
    return prefix.map({"US": "National", "ST": "State", "MSA": "Metro", "CTY": "County"})
