"""The raw-index variant keeps the smoothed panel's counties and weights, and still adds up."""
import os

import numpy as np
import pandas as pd

from housing import backtest as bt
from housing import panel
from housing.config import data_dirs
from housing.robustness import build_raw_panel, seasonal_naive


def test_raw_panel_keeps_counties_and_weights_and_adds_up(built):
    raw, smoothed = data_dirs()
    os.environ["HOUSING_ZHVI"] = "raw"
    try:
        out = data_dirs()[1]
        assert out != smoothed and out.name == "processed_raw"
        audit = build_raw_panel(raw, smoothed, out)
        Y, S_df, tags, _ = panel.load_hierarchy()
    finally:
        os.environ.pop("HOUSING_ZHVI")
    assert audit["counties_with_full_raw_history"] == audit["counties_in_smoothed_panel"]
    base = pd.read_parquet(smoothed / "county_month.parquet").drop_duplicates("fips").set_index("fips")["hu"]
    rawc = pd.read_parquet(out / "county_month.parquet").drop_duplicates("fips").set_index("fips")["hu"]
    assert rawc.sort_index().equals(base.sort_index())                     # same counties, same weights
    wide = Y.pivot(index="ds", columns="unique_id", values="y")
    counties = [c for c in wide.columns if c.startswith("CTY:")]
    assert np.allclose(wide["US"], wide[counties].sum(axis=1))             # still adds up
    smooth_us = bt.log_wide(panel.load_hierarchy()[0])["US"].diff().std()
    assert bt.log_wide(Y)["US"].diff().std() > smooth_us                   # the raw index is noisier


def test_seasonal_naive_repeats_last_years_growth():
    idx = pd.date_range("2010-01-31", periods=40, freq="ME")
    season = 0.01 * np.sin(2 * np.pi * idx.month / 12)
    L = pd.DataFrame({"US": np.cumsum(0.002 + season)}, index=idx)
    out = seasonal_naive(L, [idx[20]], (1, 3))
    for h, row in zip((1, 3), out.itertuples()):
        assert np.isclose(row.pred, L["US"].iloc[20 - 12 + h] - L["US"].iloc[20 - 12], atol=1e-6)
    assert np.allclose(out["pred"], out["actual"], atol=1e-6)            # a pure seasonal pattern is predicted exactly
