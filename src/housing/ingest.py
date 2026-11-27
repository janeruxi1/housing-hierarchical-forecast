"""Read the raw downloads into tidy (long) tables. No modelling choices here."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from housing.config import STATES

DATE_COL = re.compile(r"^\d{4}-\d{2}-\d{2}$")
STATE_NAME_TO_ABBR = {name: abbr for abbr, name in STATES.values()}


def read_zillow_wide(path: Path) -> pd.DataFrame:
    """Zillow CSVs are wide: id columns, then one column per month-end date. Return long."""
    wide = pd.read_csv(path, dtype={"StateCodeFIPS": str, "MunicipalCodeFIPS": str, "RegionID": str})
    dates = [c for c in wide.columns if DATE_COL.match(c)]
    ids = [c for c in wide.columns if c not in dates]
    long = wide.melt(id_vars=ids, value_vars=dates, var_name="date", value_name="value")
    long["date"] = pd.to_datetime(long["date"]) + pd.offsets.MonthEnd(0)
    return long.dropna(subset=["value"]).reset_index(drop=True)


def load_county_zhvi(raw: Path, name: str = "zhvi_county.csv") -> pd.DataFrame:
    df = read_zillow_wide(raw / name)
    st = df["StateCodeFIPS"].astype(str).str.zfill(2)
    co = df["MunicipalCodeFIPS"].astype(str).str.zfill(3)
    out = pd.DataFrame({
        "fips": st + co,
        "county": df["RegionName"],
        "state": df["State"],
        "metro_full": df["Metro"].fillna(""),
        "date": df["date"],
        "zhvi": df["value"].astype(float),
    })
    return out


def load_published(raw: Path, name: str) -> pd.DataFrame:
    """Zillow's own metro (and national) or state ZHVI: used to check our aggregation, not to model."""
    df = read_zillow_wide(raw / name)
    return pd.DataFrame({
        "region": df["RegionName"],
        "region_type": df["RegionType"].str.lower(),
        "date": df["date"],
        "zhvi": df["value"].astype(float),
    })


def load_zhvf(raw: Path) -> pd.DataFrame:
    """Zillow's metro forecast: % growth from BaseDate to each target month (a single vintage)."""
    wide = pd.read_csv(raw / "zhvf_metro.csv")
    dates = [c for c in wide.columns if DATE_COL.match(c)]
    long = wide.melt(id_vars=["RegionName", "RegionType", "BaseDate"], value_vars=dates,
                     var_name="target", value_name="growth_pct")
    long["BaseDate"] = pd.to_datetime(long["BaseDate"]) + pd.offsets.MonthEnd(0)
    long["target"] = pd.to_datetime(long["target"]) + pd.offsets.MonthEnd(0)
    long["horizon"] = ((long["target"].dt.year - long["BaseDate"].dt.year) * 12
                       + long["target"].dt.month - long["BaseDate"].dt.month)
    return long.rename(columns={"RegionName": "region", "RegionType": "region_type", "BaseDate": "base"})


def norm_name(s: str) -> str:
    """Normalise a county name for joining Zillow to Census.

    Handles the spellings that differ between the two sources: accents ('Doña Ana'), 'Saint' vs
    'St.', spacing ('De Kalb' vs 'DeKalb', 'La Salle' vs 'LaSalle') and apostrophes ("O'Brien").
    Key: lower case, no spaces or punctuation, e.g. 'Saint Louis County' -> 'stlouiscounty'.
    """
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\bsainte\b", "ste", s)
    s = re.sub(r"\bsaint\b", "st", s)
    return re.sub(r"[^a-z0-9]", "", s)


AREA_SUFFIXES = r"(city and borough|census area|municipality|borough|county|parish)$"


def short_key(key: str) -> str:
    """Key without the area type, for a fallback join ('Anchorage Borough' vs 'Anchorage Municipality')."""
    return re.sub(AREA_SUFFIXES.replace(" ", ""), "", key)


def load_housing_units(raw: Path) -> pd.DataFrame:
    """Census CO-EST2025-HU.xlsx: rows '.Autauga County, Alabama' with annual housing units.

    Returns county name, state abbreviation and the latest year's housing units (plus the
    April 2020 base). The table has no FIPS codes, so it is joined to Zillow by name.
    """
    x = pd.read_excel(raw / "housing_units_county.xlsx", header=None)
    rows = x[x[0].astype(str).str.startswith(".")]
    nums = rows.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    name = rows[0].str.lstrip(".")
    county = name.str.rsplit(",", n=1).str[0].str.strip()
    state_name = name.str.rsplit(",", n=1).str[1].str.strip()
    last = nums.columns[nums.notna().any()].max()
    out = pd.DataFrame({
        "county": county.values,
        "state": state_name.map(STATE_NAME_TO_ABBR).values,
        "hu_2020": nums.iloc[:, 0].values,
        "hu": nums[last].values,
    })
    out["key"] = out["state"] + "|" + out["county"].map(norm_name)
    return out.dropna(subset=["state", "hu"]).reset_index(drop=True)


def load_population(raw: Path) -> pd.DataFrame:
    """Census co-est2025-alldata: county population and net domestic migration, by year (July-June)."""
    df = pd.read_csv(raw / "population_county.csv", encoding="latin-1", dtype={"STATE": str, "COUNTY": str})
    df = df[df["SUMLEV"].astype(int) == 50]
    fips = df["STATE"].str.zfill(2) + df["COUNTY"].str.zfill(3)
    years = sorted(int(c[-4:]) for c in df.columns if re.fullmatch(r"POPESTIMATE\d{4}", c))
    rows = []
    for y in years:
        dom = df.get(f"DOMESTICMIG{y}")
        rows.append(pd.DataFrame({
            "fips": fips.values, "year": y,
            "pop": df[f"POPESTIMATE{y}"].values,
            "net_domestic_mig": dom.values if dom is not None else np.nan,
        }))
    out = pd.concat(rows, ignore_index=True)
    out["dom_mig_rate"] = out["net_domestic_mig"] / out["pop"] * 1000  # per 1,000 residents
    return out


def load_fred(raw: Path, series: str) -> pd.Series:
    """FRED CSV -> monthly series at month end (weekly series such as mortgage rates are averaged)."""
    df = pd.read_csv(raw / "fred" / f"{series}.csv")
    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    s = pd.to_numeric(df[series], errors="coerce")
    s.index = pd.to_datetime(df[date_col])
    return s.dropna().resample("ME").mean().rename(series).rename_axis("date")


def metro_short(full: str) -> str:
    """CBSA title -> Zillow's short metro name: 'Seattle-Tacoma-Bellevue, WA' -> 'Seattle, WA'.

    Zillow's county file carries the metro each county belongs to; its metro file uses the short
    form (first principal city, first state). Names already in short form pass through unchanged.
    """
    if not isinstance(full, str) or "," not in full:
        return full if isinstance(full, str) else ""
    city, states = full.rsplit(",", 1)
    first = city.split("-")[0].split("/")[0].strip()   # 'Louisville/Jefferson County' -> 'Louisville'
    return f"{first}, {states.strip().split('-')[0]}"
