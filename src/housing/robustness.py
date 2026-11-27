"""Robustness check: the whole backtest again on Zillow's raw index (notebook 06).

The main analysis forecasts Zillow's smoothed, seasonally adjusted index, which is unusually easy
to forecast a month or two ahead. Here the same counties, housing-unit weights and hierarchy are
rebuilt from the raw index (neither smoothed nor seasonally adjusted), every model is refitted on
it, and forecasts are scored against raw outcomes.

Because the raw index has a seasonal cycle, a seasonal benchmark joins the simple rules:
*seasonal naive*, growth over the next h months = growth over the same months a year earlier.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from housing import ingest
from housing.config import MAX_GAP, START
from housing.hierarchy import aggregate, node_table, summing_matrix

RAW_FILE = "zhvi_county_raw.csv"


def build_raw_panel(raw: Path, smoothed: Path, out: Path) -> dict:
    """Write the raw-index versions of the processed tables to `out`, keeping the smoothed panel's
    counties and weights. Counties without a full raw history are dropped (and reported)."""
    out.mkdir(parents=True, exist_ok=True)
    base = pd.read_parquet(smoothed / "county_month.parquet")
    meta = base.drop_duplicates("fips")[["fips", "county", "state", "metro", "hu"]]
    rz = ingest.load_county_zhvi(raw, RAW_FILE)
    rz = rz[rz["fips"].isin(meta["fips"])]
    wide = np.log(rz.pivot(index="date", columns="fips", values="zhvi"))
    filled = wide.interpolate(limit=MAX_GAP, limit_area="inside")
    months = pd.DatetimeIndex(sorted(base["date"].unique()))
    filled = filled.reindex(months)
    full = filled.notna().all()
    keep = full[full].index
    audit = {
        "counties_in_smoothed_panel": int(meta["fips"].nunique()),
        "counties_with_full_raw_history": int(len(keep)),
        "months_interpolated": int(filled[keep].notna().sum().sum() - wide.reindex(months)[keep].notna().sum().sum()),
        "first_month": months[0].strftime("%Y-%m"), "last_month": months[-1].strftime("%Y-%m"),
    }
    long = np.exp(filled[keep]).stack().rename("zhvi").reset_index().rename(columns={"level_0": "date"})
    panel = long.merge(meta, on="fips")
    panel = panel[panel["date"] >= START]
    panel["value"] = panel["zhvi"] * panel["hu"]
    panel = panel[["fips", "county", "state", "metro", "date", "zhvi", "hu", "value"]].sort_values(["fips", "date"])
    panel.to_parquet(out / "county_month.parquet", index=False)

    nodes = node_table(panel)
    S, tags = summing_matrix(nodes)
    bottom = panel.assign(unique_id="CTY:" + panel["fips"], ds=panel["date"])
    Y = aggregate(bottom.rename(columns={"value": "y"}), S, "y")
    end = panel["date"].max()
    H = aggregate(bottom[bottom["date"] == end].assign(y=lambda d: d["hu"]), S, "y")[["unique_id", "y"]]
    Y = Y.merge(H.rename(columns={"y": "hu"}), on="unique_id")
    Y["typical_value"] = Y["y"] / Y["hu"]
    Y.to_parquet(out / "hierarchy_month.parquet", index=False)
    S.to_parquet(out / "S.parquet", index=False)
    nodes.to_parquet(out / "nodes.parquet", index=False)
    audit["nodes_by_level"] = {k: int(len(v)) for k, v in tags.items()}
    for name in ("macro_month.parquet", "state_unemp.parquet"):      # the drivers do not change
        shutil.copyfile(smoothed / name, out / name)
    (out / "raw_panel_audit.json").write_text(json.dumps(audit, indent=2))
    return audit


def seasonal_naive(L: pd.DataFrame, origins, horizons) -> pd.DataFrame:
    """Long table like backtest.run(): growth over the next h months = the same months' growth a year earlier."""
    X = L.to_numpy()
    ids = pd.Categorical(L.columns)
    rows = []
    for o in origins:
        i = L.index.get_loc(o)
        for h in horizons:
            if i + h >= len(L) or i - 12 < 0:
                continue
            rows.append(pd.DataFrame({"origin": o, "h": np.int8(h), "unique_id": ids,
                                      "pred": (X[i - 12 + h] - X[i - 12]).astype("float32"),
                                      "actual": (X[i + h] - X[i]).astype("float32")}))
    out = pd.concat(rows, ignore_index=True)
    out.insert(0, "method", pd.Categorical(["seasonal_naive"] * len(out)))
    return out
