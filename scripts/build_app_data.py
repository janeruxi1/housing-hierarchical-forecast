"""Build the small extract the Streamlit demo reads (app/data/), from the real processed panel.

    python scripts/build_app_data.py

The demo cannot download or rebuild the panel (the Zillow files are not redistributed), so this
script runs the chosen method from the last month, exactly as notebook 04 does, and saves only
derived results: growth rates, forecasts, scenario responses and backtest errors. No Zillow index
level for a single county is written. Dollar values appear only for the aggregates this project
builds (nation, states, metros). tests/test_app_data.py checks the extract against reports/metrics.

Scenarios are linear in the shock: county growth = baseline + beta x (coefficients x input change),
then counties are added up. The script evaluates that on a grid of rate and unemployment changes,
so the app can offer sliders without the county values.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from housing import backtest as bt  # noqa: E402
from housing import models, panel, scenarios  # noqa: E402
from housing.config import STATES, data_dirs, use_synthetic  # noqa: E402
from housing.hierarchy import level_of  # noqa: E402

OUT = ROOT / "app" / "data"
CHOSEN = "combo+bottom_up"
RATE_GRID = np.round(np.arange(-2.0, 3.01, 0.5), 2)     # change in the 30-year mortgage rate, points
UNEMP_GRID = np.arange(-1.0, 4.01, 1.0)                 # change in unemployment, points
STATE_NAME = {abbr: name for abbr, name in STATES.values()}


def labels(ids: pd.Index, counties: pd.DataFrame) -> pd.DataFrame:
    lv = level_of(pd.Series(ids)).to_numpy()
    cty = counties.drop_duplicates("fips").set_index("fips")
    names = []
    for u, level in zip(ids, lv):
        if level == "National":
            names.append("United States")
        elif level == "State":
            names.append(STATE_NAME.get(u[3:], u[3:]))
        elif level == "Metro":
            names.append(u[4:])
        else:
            f = u[4:]
            names.append(f"{cty.loc[f, 'county']}, {cty.loc[f, 'state']}" if f in cty.index else u)
    return pd.DataFrame({"unique_id": ids, "level": lv, "name": names})


def main() -> None:
    if use_synthetic():
        sys.exit("Run on the real data: the demo shows real results only.")
    proc = data_dirs()[1]
    OUT.mkdir(parents=True, exist_ok=True)

    Y, S_df, tags, nodes = panel.load_hierarchy()
    L = bt.log_wide(Y)[list(S_df["unique_id"])]
    F = models.features(L, panel.load_macro(), pd.read_parquet(proc / "state_unemp.parquet"), nodes)
    now = scenarios.forecast_now(L, F, tags)
    origin = now["origin"]
    y0 = np.exp(L.iloc[-1])
    hu = Y.drop_duplicates("unique_id").set_index("unique_id")["hu"].reindex(L.columns)
    counties = list(S_df.columns[1:])
    beta = scenarios.county_betas(L, counties, weights=hu)
    zero = pd.Series(0.0, index=scenarios.PATH_H)
    base_vals = scenarios.apply(now["combo"], S_df, y0, beta, zero)

    # 1. Places: name, level, parents, housing units, beta (counties and metros)
    county_meta = pd.read_parquet(proc / "county_month.parquet", columns=["fips", "county", "state"])
    place = labels(L.columns, county_meta)
    parent = nodes.set_index("County")
    place["state_id"] = place["unique_id"].map(lambda u: u if u.startswith("ST:") else None)
    # a metro's state is the one in its name ("New York, NY", "Non-metro WA"), even when it spans several
    is_metro = place["level"] == "Metro"
    place.loc[is_metro, "state_id"] = "ST:" + place.loc[is_metro, "name"].str[-2:]
    place.loc[place["level"] == "County", "state_id"] = place["unique_id"].map(parent["State"])
    place["metro_id"] = place["unique_id"].map(parent["Metro"])
    place["housing_units"] = hu.to_numpy()
    w = hu.reindex(counties)
    metro_of = parent["Metro"].reindex(counties)
    metro_beta = (beta * w).groupby(metro_of).sum() / w.groupby(metro_of).sum()
    place["beta"] = place["unique_id"].map(pd.concat([beta, metro_beta]))
    agg = place["level"] != "County"
    place["value_bn"] = np.where(agg, y0.to_numpy() / 1e9, np.nan)       # aggregates only
    place["typical_value"] = np.where(agg, (y0 / hu).to_numpy(), np.nan)
    place.to_parquet(OUT / "places.parquet", index=False)

    # 2. Baseline path, h = 0..12, cumulative growth from the forecast month (%)
    g = 100 * (base_vals.div(y0.reindex(base_vals.index), axis=0) - 1)
    g.insert(0, 0, 0.0)
    path = g.reset_index(names="unique_id").melt(id_vars="unique_id", var_name="h", value_name="growth_pct")
    path["h"] = path["h"].astype("int8")
    path["growth_pct"] = path["growth_pct"].astype("float32")
    path.to_parquet(OUT / "forecast.parquet", index=False)

    # 3. Scenario grid: 12-month growth (%) for every place, rate x unemployment x coefficient version
    rows = []
    for version, coefs in now["county_coefs"].items():
        for r in RATE_GRID:
            for u in UNEMP_GRID:
                shock = scenarios.shock_path(coefs, {"d_rate12": r, "rate_gap": r, "d_unemp12": u})
                vals = scenarios.apply(now["combo"][[12]], S_df, y0, beta, shock.loc[[12]])
                rows.append(pd.DataFrame({"unique_id": vals.index, "version": version, "rate": r, "unemp": u,
                                          "growth_pct": 100 * (vals[12].to_numpy() / y0.reindex(vals.index).to_numpy() - 1)}))
    grid = pd.concat(rows, ignore_index=True)
    grid["version"] = grid["version"].astype("category")
    grid[["rate", "unemp", "growth_pct"]] = grid[["rate", "unemp", "growth_pct"]].astype("float32")
    grid.to_parquet(OUT / "scenarios.parquet", index=False)

    # 4. History: 12-month growth of the typical home value (%), aggregates only
    agg_ids = place.loc[agg, "unique_id"]
    hist = Y[Y["unique_id"].isin(agg_ids)].pivot(index="ds", columns="unique_id", values="typical_value")
    h12 = (100 * (hist / hist.shift(12) - 1)).dropna(how="all").stack().rename("growth_pct").reset_index()
    h12["growth_pct"] = h12["growth_pct"].astype("float32")
    h12.to_parquet(OUT / "history.parquet", index=False)

    # 5. Backtest (aggregates): the chosen method and momentum, 3 and 12 months ahead, log growth
    b = pd.read_parquet(proc / "backtest_models.parquet",
                        filters=[("method", "in", [CHOSEN, "momentum", "naive"]), ("h", "in", [3, 12])])
    b = b[b["unique_id"].isin(agg_ids)].copy()
    b["method"] = b["method"].astype(str).astype("category")
    b[["pred", "actual"]] = b[["pred", "actual"]].astype("float32")
    b.to_parquet(OUT / "backtest.parquet", index=False)

    # 6. Error bands for the fan: 2.5/10/90/97.5th percentiles of the chosen method's past errors by level and h
    e = pd.read_parquet(proc / "backtest_models.parquet", filters=[("method", "==", CHOSEN)])
    e["level"] = level_of(e["unique_id"].astype(str)).to_numpy()
    e["err"] = e["pred"] - e["actual"]
    bands = e.groupby(["level", "h"])["err"].quantile([0.025, 0.1, 0.9, 0.975]).unstack().reset_index()
    bands.columns = ["level", "h", "q025", "q10", "q90", "q975"]
    bands.to_parquet(OUT / "bands.parquet", index=False)

    meta = {"origin": str(origin.date()), "chosen": CHOSEN, "rate_grid": RATE_GRID.tolist(),
            "unemp_grid": UNEMP_GRID.tolist(), "counties": len(counties),
            "places": {lv: int((place["level"] == lv).sum()) for lv in ["National", "State", "Metro", "County"]},
            "data": json.loads((proc.parent.parent / "reports" / "metrics" / "data_audit.json").read_text())["zhvi_last_month"]}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    for f in sorted(OUT.iterdir()):
        print(f"{f.name:20s} {f.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
