"""The demo's extract (app/data, built by scripts/build_app_data.py from the real panel) reproduces the
results notebook 04 recorded, so the app can never show a number the memo does not."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
D, M = ROOT / "app" / "data", ROOT / "reports" / "metrics"


@pytest.fixture(scope="module")
def ex():
    fc = pd.read_parquet(D / "forecast.parquet")
    grid = pd.read_parquet(D / "scenarios.parquet")
    grid["version"] = grid["version"].astype(str)
    return {"g12": fc[fc["h"] == 12].set_index("unique_id")["growth_pct"].astype(float),
            "grid": grid, "places": pd.read_parquet(D / "places.parquet").set_index("unique_id"),
            "backtest": pd.read_parquet(D / "backtest.parquet")}


def scen(ex, version, rate, unemp):
    g = ex["grid"]
    return g[(g["version"] == version) & np.isclose(g["rate"], rate) & np.isclose(g["unemp"], unemp)].set_index(
        "unique_id")["growth_pct"].astype(float)


def lp(a, b):
    return 100 * np.log((1 + a / 100) / (1 + b / 100))


def test_baseline_matches_notebook(ex):
    rec = json.loads((M / "04_vs_zillow_forecast.json").read_text())
    assert ex["g12"]["US"] == pytest.approx(rec["us_ours_pct"], abs=1e-4)
    lv = pd.read_csv(M / "04_baseline_by_level.csv").set_index("level")
    log_g = 100 * np.log1p(ex["g12"] / 100)
    for level, med in log_g.groupby(ex["places"]["level"].reindex(log_g.index)).median().items():
        assert med == pytest.approx(lv.loc[level, "median (%)"], abs=1e-3), level


def test_every_scenario_row_matches(ex):
    rows = pd.read_csv(M / "04_scenarios.csv")
    deltas = {"Rates +1 pt": (1, 0), "Rates +2 pt (stress)": (2, 0), "Rates -1 pt": (-1, 0),
              "Recession (unemployment +3, rates -1)": (-1, 3)}
    us_value = ex["places"].loc["US", "value_bn"]
    for _, r in rows[rows["scenario"] != "Baseline"].iterrows():
        g = scen(ex, r["coefficients"], *deltas[r["scenario"]])["US"]
        assert g == pytest.approx(r["US 12-month growth (%)"], abs=1e-3)
        assert lp(g, ex["g12"]["US"]) == pytest.approx(r["vs baseline (pp)"], abs=1e-3)
        assert us_value * (g - ex["g12"]["US"]) / 100 == pytest.approx(r["value change vs baseline ($bn)"], rel=1e-4)


def test_metro_exposure_and_stress_count(ex):
    exp = pd.read_csv(M / "04_metro_exposure.csv").set_index("metro")
    ids = "MSA:" + exp.index
    hit1 = lp(scen(ex, "full sample", 1, 0).reindex(ids), ex["g12"].reindex(ids))
    np.testing.assert_allclose(hit1.to_numpy(), exp["rates +1: change (pp)"].to_numpy(), atol=1e-3)
    np.testing.assert_allclose(ex["places"].loc[ids, "beta"].to_numpy(), exp["beta"].to_numpy(), atol=1e-4)
    hit2 = lp(scen(ex, "full sample", 2, 0).reindex(ids), ex["g12"].reindex(ids)).to_numpy()
    stressed = 100 * ((1 + exp["Zillow 12m (%)"].to_numpy() / 100) * np.exp(hit2 / 100) - 1)
    rec = json.loads((M / "04_stress_falls.json").read_text())
    assert int((stressed < 0).sum()) == rec["metros_of_50_falling"]


def test_backtest_errors_match_notebook_03(ex):
    b = ex["backtest"].copy()
    b["level"] = b["unique_id"].astype(str).map(lambda u: "National" if u == "US" else
                                                {"ST": "State", "MSA": "Metro"}[u.split(":")[0]])
    b["e2"] = (100 * (b["pred"].astype(float) - b["actual"].astype(float))) ** 2
    rmse = np.sqrt(b.groupby(["method", "level", "h"], observed=True)["e2"].mean())
    rec = pd.read_csv(M / "03_final_vs_momentum.csv").set_index(["level", "h"])
    for (method, level, h), v in rmse.items():
        assert v == pytest.approx(rec.loc[(level, h), method], abs=1e-3), (method, level, h)


def test_no_county_index_levels_are_shipped(ex):
    """Zillow's county series are not redistributed: no dollar values or history for single counties."""
    county = ex["places"]["level"] == "County"
    assert ex["places"].loc[county, ["value_bn", "typical_value"]].isna().all().all()
    hist = pd.read_parquet(D / "history.parquet")
    assert not hist["unique_id"].astype(str).str.startswith("CTY:").any()
    assert not ex["backtest"]["unique_id"].astype(str).str.startswith("CTY:").any()
