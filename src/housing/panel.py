"""Build the modelling tables from the raw downloads, and load them back."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from housing import ingest
from housing.config import FRED_NATIONAL, FRED_STATE_UNEMP, MAX_GAP, START, data_dirs, metrics_dir
from housing.hierarchy import aggregate, node_table, summing_matrix


def build(raw: Path, out: Path) -> dict:
    """raw/ -> processed/ parquet files. Returns the audit numbers (also written to data_audit.json)."""
    out.mkdir(parents=True, exist_ok=True)
    audit: dict = {}

    # 1. County ZHVI and its metro
    cz = ingest.load_county_zhvi(raw)
    cz["metro"] = cz["metro_full"].map(ingest.metro_short)
    cz["key"] = cz["state"] + "|" + cz["county"].map(ingest.norm_name)
    end = cz["date"].max()
    audit["zhvi_last_month"] = end.strftime("%Y-%m")
    audit["counties_in_zillow"] = int(cz["fips"].nunique())

    # 2. Housing-unit weights, joined by name (the Census table has no FIPS)
    hu = ingest.load_housing_units(raw)
    keys = cz.drop_duplicates("fips")[["fips", "key", "state"]]
    matched = keys.merge(hu[["key", "hu", "hu_2020"]], on="key", how="left")
    # Fallback: same name with the area type dropped, when that is unique within the state
    hu_short = hu.assign(skey=hu["key"].map(ingest.short_key))
    hu_short = hu_short[~hu_short["skey"].duplicated(keep=False)].set_index("skey")
    miss = matched["hu"].isna()
    sk = matched.loc[miss, "key"].map(ingest.short_key)
    for col in ("hu", "hu_2020"):
        matched.loc[miss, col] = sk.map(hu_short[col]).to_numpy()
    audit["counties_matched_by_fallback"] = int((miss & matched["hu"].notna()).sum())
    audit["counties_matched_to_housing_units"] = int(matched["hu"].notna().sum())
    audit["unmatched_counties_by_state"] = (
        matched[matched["hu"].isna()].groupby("state").size().sort_values(ascending=False).head(10).to_dict()
    )
    cz = cz.merge(matched[["fips", "hu", "hu_2020"]], on="fips", how="inner").dropna(subset=["hu"])

    # 3. Short gaps inside a county's history (Zillow leaves 1-5 month holes in some thin counties)
    #    are filled by log-linear interpolation; nothing is extrapolated before a series starts.
    wide = np.log(cz.pivot(index="date", columns="fips", values="zhvi"))
    filled = wide.interpolate(limit=MAX_GAP, limit_area="inside")
    audit["months_interpolated"] = int(filled.notna().sum().sum() - wide.notna().sum().sum())
    audit["counties_with_interpolation"] = int((filled.notna().sum() > wide.notna().sum()).sum())
    long = np.exp(filled).stack().rename("zhvi").reset_index()
    meta = cz.drop_duplicates("fips").drop(columns=["date", "zhvi"])
    cz = long.merge(meta, on="fips")

    # 4. Balanced panel: counties with a value every month from START to the last month
    cz = cz[cz["date"] >= START]
    n_months = cz["date"].nunique()
    full = cz.groupby("fips")["zhvi"].count().eq(n_months)
    keep = full[full].index
    audit["counties_full_history"] = int(len(keep))
    audit["months"] = int(n_months)
    audit["share_of_us_housing_units_kept"] = round(
        float(hu.loc[hu["key"].isin(cz.loc[cz["fips"].isin(keep), "key"]), "hu"].sum() / hu["hu"].sum()), 4)
    allc = cz.drop_duplicates("fips")[["fips", "state", "metro", "hu"]]
    allc["kept"] = allc["fips"].isin(keep)
    allc["metro"] = allc["metro"].where(allc["metro"].ne(""), "Non-metro " + allc["state"])
    cov = pd.concat([
        allc.assign(node="MSA:" + allc["metro"]), allc.assign(node="ST:" + allc["state"]), allc.assign(node="US")])
    cov = cov.groupby("node").apply(lambda g: g.loc[g["kept"], "hu"].sum() / g["hu"].sum(), include_groups=False)
    cov.rename("hu_coverage").rename_axis("unique_id").reset_index().to_parquet(out / "coverage.parquet", index=False)
    panel = cz[cz["fips"].isin(keep)].copy()
    panel["value"] = panel["zhvi"] * panel["hu"]  # $ value of the county's housing stock at this month's prices
    panel = panel[["fips", "county", "state", "metro", "date", "zhvi", "hu", "value"]].sort_values(["fips", "date"])
    panel.to_parquet(out / "county_month.parquet", index=False)

    # 5. Hierarchy and the aggregated series (y = total $ value; hu = housing units per node)
    nodes = node_table(panel)
    S, tags = summing_matrix(nodes)
    bottom = panel.assign(unique_id="CTY:" + panel["fips"], ds=panel["date"])
    Y = aggregate(bottom.rename(columns={"value": "y"}), S, "y")
    H = aggregate(bottom[bottom["date"] == end].assign(y=lambda d: d["hu"]), S, "y")[["unique_id", "y"]]
    Y = Y.merge(H.rename(columns={"y": "hu"}), on="unique_id")
    Y["typical_value"] = Y["y"] / Y["hu"]
    Y.to_parquet(out / "hierarchy_month.parquet", index=False)
    S.to_parquet(out / "S.parquet", index=False)
    nodes.to_parquet(out / "nodes.parquet", index=False)
    audit["nodes_by_level"] = {k: int(len(v)) for k, v in tags.items()}

    # 6. Zillow's own published series (for the coherence check) and its forecast (benchmark)
    pub = pd.concat([ingest.load_published(raw, "zhvi_metro.csv"), ingest.load_published(raw, "zhvi_state.csv")])
    pub.to_parquet(out / "zillow_published.parquet", index=False)
    if (raw / "zhvf_metro.csv").exists():
        ingest.load_zhvf(raw).to_parquet(out / "zhvf_metro.parquet", index=False)

    # 7. Drivers: national macro series and state unemployment, monthly
    macro = pd.concat([ingest.load_fred(raw, s) for s in FRED_NATIONAL], axis=1)
    macro.index.name = "date"
    macro.reset_index().to_parquet(out / "macro_month.parquet", index=False)
    su = []
    for s in FRED_STATE_UNEMP:
        if (raw / "fred" / f"{s}.csv").exists():
            su.append(ingest.load_fred(raw, s).rename("unrate").reset_index().assign(state=s[:-2]))
    if su:
        pd.concat(su).to_parquet(out / "state_unemp.parquet", index=False)
    if (raw / "population_county.csv").exists():
        ingest.load_population(raw).to_parquet(out / "population_county.parquet", index=False)

    mdir = metrics_dir()
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "data_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    return audit


# --- loaders used by notebooks and tests -------------------------------------------------------
def _p(name: str) -> pd.DataFrame:
    return pd.read_parquet(data_dirs()[1] / name)


def load_counties() -> pd.DataFrame:
    return _p("county_month.parquet")


def load_hierarchy() -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """(Y_df, S_df, tags, nodes)."""
    Y, S, nodes = _p("hierarchy_month.parquet"), _p("S.parquet"), _p("nodes.parquet")
    _, tags = summing_matrix(nodes)
    return Y, S, tags, nodes


def load_published() -> pd.DataFrame:
    return _p("zillow_published.parquet")


def load_macro() -> pd.DataFrame:
    return _p("macro_month.parquet")
