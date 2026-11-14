"""Synthetic stand-ins for the raw downloads, in the same file formats.

Used by the tests and CI so the whole pipeline runs without the (uncommitted) real data. The
numbers are made up: a national boom-bust-boom cycle, metros that load on it with different
betas, counties that wander around their metro, and a mortgage rate that drives part of it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from housing.config import FRED_NATIONAL, STATES

GEO = {  # state -> {metro full title (or "" for non-metro): n counties}
    "WA": {"Seattle-Tacoma-Bellevue, WA": 3, "Spokane-Spokane Valley, WA": 2, "": 2},
    "OR": {"Portland-Vancouver-Hillsboro, OR-WA": 2, "": 2},
    "AZ": {"Phoenix-Mesa-Chandler, AZ": 2, "Tucson, AZ": 1, "": 2},
    "OH": {"Cincinnati, OH-KY-IN": 2, "Columbus, OH": 2, "": 3},
}
STATE_FIPS = {abbr: f for f, (abbr, _) in STATES.items()}
STATE_NAME = {abbr: name for abbr, name in STATES.values()}


def _dates(start="2000-01-31", end="2025-09-30") -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="ME")


def _national_growth(dates: pd.DatetimeIndex, rate: np.ndarray, rng) -> np.ndarray:
    """Monthly log growth: a cycle shaped like 2000-2025 US housing, minus a rate effect, plus noise."""
    y = dates.year + (dates.month - 1) / 12
    cycle = np.select(
        [y < 2006.3, y < 2012, y < 2020.3, y < 2022.5, y >= 2022.5],
        [0.007, -0.0045, 0.004, 0.012, 0.001],
    )
    g = cycle - 0.0012 * (rate - rate.mean()) + rng.normal(0, 0.0015, len(dates))
    return pd.Series(g).ewm(alpha=0.35).mean().to_numpy()  # ZHVI is smoothed


def write_raw(raw: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "fred").mkdir(exist_ok=True)
    dates = _dates()
    n = len(dates)

    # Drivers
    rate = 6.5 - 3.5 * np.sin(np.linspace(0, 2.4, n)) + np.cumsum(rng.normal(0, 0.06, n))
    rate[-40:] += np.linspace(0, 3.0, 40)  # a 2022-style rate shock
    unrate = 5 + 2.5 * np.exp(-((np.arange(n) - 115) / 20) ** 2) + 8 * np.exp(-((np.arange(n) - 243) / 2) ** 2)
    nat = _national_growth(dates, rate, rng)

    weekly = pd.date_range(dates[0] - pd.offsets.MonthBegin(1), dates[-1], freq="W-THU")
    fred = {
        "MORTGAGE30US": pd.Series(np.interp(weekly.asi8, dates.asi8, rate), index=weekly),
        "UNRATE": pd.Series(unrate, index=dates - pd.offsets.MonthBegin(1)),
        "CPIAUCSL": pd.Series(170 * np.exp(np.cumsum(np.full(n, 0.002))), index=dates - pd.offsets.MonthBegin(1)),
    }
    for abbr in GEO:
        fred[f"{abbr}UR"] = pd.Series(unrate + rng.normal(0, 0.3), index=dates - pd.offsets.MonthBegin(1))
    for sid, s in fred.items():
        assert sid in FRED_NATIONAL or sid.endswith("UR")
        pd.DataFrame({"observation_date": s.index.strftime("%Y-%m-%d"), sid: s.round(3).values}).to_csv(
            raw / "fred" / f"{sid}.csv", index=False)

    # Counties
    counties, county_paths, hu_rows = [], {}, []
    k = 0
    for abbr, metros in GEO.items():
        for metro, n_c in metros.items():
            beta = rng.uniform(0.6, 1.5) if metro else rng.uniform(0.4, 0.9)
            metro_g = beta * nat + rng.normal(0, 0.0008, n)
            for j in range(n_c):
                k += 1
                fips_c = f"{k * 2 + 1:03d}"
                name = f"{['Pine', 'Cedar', 'Lake', 'River', 'Hill', 'Valley', 'Mesa'][k % 7]} {k} County"
                g = metro_g + pd.Series(rng.normal(0, 0.001, n)).ewm(alpha=0.2).mean().to_numpy()
                level = rng.uniform(1.2e5, 4.5e5) * np.exp(np.cumsum(g))
                start = 0 if k % 6 else 70  # a few counties only start in 2005-11, like thin rural ZHVI
                level[:start] = np.nan
                units = int(rng.uniform(8e3, 6e5) if metro else rng.uniform(3e3, 4e4))
                counties.append(dict(RegionID=str(1000 + k), SizeRank=k, RegionName=name, RegionType="county",
                                     StateName=abbr, State=abbr, Metro=metro or None,
                                     StateCodeFIPS=STATE_FIPS[abbr], MunicipalCodeFIPS=fips_c))
                county_paths[k] = level
                hu_rows.append((f".{name}, {STATE_NAME[abbr]}", units))

    def wide(meta: list[dict], paths: list[np.ndarray]) -> pd.DataFrame:
        vals = pd.DataFrame(np.vstack(paths), columns=dates.strftime("%Y-%m-%d"))
        return pd.concat([pd.DataFrame(meta), vals], axis=1)

    wide(counties, list(county_paths.values())).to_csv(raw / "zhvi_county.csv", index=False)

    # Zillow's published metro/state/national series: our unit-weighted average, plus a small
    # gap, because Zillow weights by value and covers homes our county set does not.
    cdf = pd.DataFrame(counties).assign(units=[u for _, u in hu_rows])
    paths = np.vstack(list(county_paths.values()))

    def published(mask: np.ndarray) -> np.ndarray:
        w = cdf["units"].to_numpy()[mask][:, None]
        p = paths[mask]
        avg = np.nansum(p * w, axis=0) / np.sum(np.where(np.isnan(p), 0, w), axis=0)
        return avg * (1 + rng.normal(0.01, 0.01)) * (1 + np.cumsum(rng.normal(0, 0.0004, n)))

    meta, ps = [dict(RegionID="102001", SizeRank=0, RegionName="United States", RegionType="country", StateName=None)], [
        published(np.ones(len(cdf), bool))]
    for metro in sorted({m for m in cdf["Metro"].dropna()}):
        city, st = metro.rsplit(",", 1)
        meta.append(dict(RegionID=str(len(meta)), SizeRank=len(meta), RegionName=f"{city.split('-')[0]},"
                         f" {st.strip().split('-')[0]}", RegionType="msa", StateName=st.strip().split("-")[0]))
        ps.append(published((cdf["Metro"] == metro).to_numpy()))
    wide(meta, ps).to_csv(raw / "zhvi_metro.csv", index=False)
    smeta, sps = [], []
    for abbr in GEO:
        smeta.append(dict(RegionID=STATE_FIPS[abbr], SizeRank=len(smeta), RegionName=STATE_NAME[abbr],
                          RegionType="state", StateName=STATE_NAME[abbr]))
        sps.append(published((cdf["State"] == abbr).to_numpy()))
    wide(smeta, sps).to_csv(raw / "zhvi_state.csv", index=False)

    # Zillow's forecast: one vintage, growth % at 1, 3 and 12 months
    base = dates[-1]
    zf = pd.DataFrame([{"RegionID": m["RegionID"], "SizeRank": m["SizeRank"], "RegionName": m["RegionName"],
                        "RegionType": m["RegionType"], "StateName": m["StateName"],
                        "BaseDate": base.strftime("%Y-%m-%d")} for m in meta])
    for h in (1, 3, 12):
        zf[(base + pd.offsets.MonthEnd(h)).strftime("%Y-%m-%d")] = np.round(rng.normal(0.2 * h, 0.3 * h ** 0.5, len(zf)), 1)
    zf.to_csv(raw / "zhvf_metro.csv", index=False)

    # Census housing units (xlsx with title rows, like CO-EST2025-HU)
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    rows = [["Annual Estimates of Housing Units for Counties (synthetic)"] + [None] * 7,
            ["Geographic Area", "April 1, 2020 Estimates Base"] + [None] * 6,
            [None, None] + years, ["United States", sum(u for _, u in hu_rows)] + [None] * 6]
    for name, units in hu_rows:
        rows.append([name, units] + [int(units * (1 + 0.008 * i)) for i in range(6)])
    rows.append(["Note: synthetic data", None] + [None] * 6)
    pd.DataFrame(rows).to_excel(raw / "housing_units_county.xlsx", header=False, index=False)

    # Census population and net domestic migration
    prow = []
    for c, (_, units) in zip(counties, hu_rows):
        r = {"SUMLEV": 50, "STATE": c["StateCodeFIPS"], "COUNTY": c["MunicipalCodeFIPS"], "CTYNAME": c["RegionName"]}
        for y in years:
            r[f"POPESTIMATE{y}"] = int(units * 2.4)
            r[f"DOMESTICMIG{y}"] = int(units * 2.4 * rng.normal(0, 0.008))
        prow.append(r)
    pd.DataFrame(prow).to_csv(raw / "population_county.csv", index=False)
