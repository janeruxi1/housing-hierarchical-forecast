"""Paths, source URLs and fixed choices, in one place."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SYNTHETIC = ROOT / "data" / "synthetic"
FIGURES = ROOT / "reports" / "figures"
METRICS = ROOT / "reports" / "metrics"


def use_synthetic() -> bool:
    """CI and tests set HOUSING_SYNTHETIC=1 so nothing reads the (uncommitted) real data."""
    return os.environ.get("HOUSING_SYNTHETIC", "") not in ("", "0")


def data_dirs() -> tuple[Path, Path]:
    """(raw, processed) folders for the current mode."""
    if use_synthetic():
        return SYNTHETIC / "raw", SYNTHETIC / "processed"
    return RAW, PROCESSED


# --- Zillow Research public CSVs (https://www.zillow.com/research/data/) -------------------
# Middle-tier (35th-65th percentile), all homes, smoothed and seasonally adjusted: the cut Zillow
# itself forecasts (ZHVF). Zillow notes that download paths change occasionally, so the
# downloader falls back to searching the data page for the file name.
ZHVI_CUT = "uc_sfrcondo_tier_0.33_0.67_sm_sa_month"
ZILLOW_BASE = "https://files.zillowstatic.com/research/public_csvs"
ZILLOW_FILES = {
    "zhvi_county.csv": f"{ZILLOW_BASE}/zhvi/County_zhvi_{ZHVI_CUT}.csv",
    "zhvi_metro.csv": f"{ZILLOW_BASE}/zhvi/Metro_zhvi_{ZHVI_CUT}.csv",  # includes the "United States" row
    "zhvi_state.csv": f"{ZILLOW_BASE}/zhvi/State_zhvi_{ZHVI_CUT}.csv",
    "zhvf_metro.csv": f"{ZILLOW_BASE}/zhvf_growth/Metro_zhvf_growth_{ZHVI_CUT}.csv",
}
ZILLOW_DATA_PAGE = "https://www.zillow.com/research/data/"

# --- Census Population Estimates, Vintage 2025 ----------------------------------------------
CENSUS_FILES = {
    # Annual housing units by county, April 2020 base to July 2025 (xlsx only)
    "housing_units_county.xlsx": (
        "https://www2.census.gov/programs-surveys/popest/tables/2020-2025/housing/totals/CO-EST2025-HU.xlsx"
    ),
    # County population and components of change (incl. net domestic migration), 2020-2025
    "population_county.csv": (
        "https://www2.census.gov/programs-surveys/popest/datasets/2020-2025/counties/totals/co-est2025-alldata.csv"
    ),
}

# --- FRED (no API key needed for the CSV export) ---------------------------------------------
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
FRED_NATIONAL = {
    "MORTGAGE30US": "30-year fixed mortgage rate, weekly (Freddie Mac PMMS)",
    "UNRATE": "US unemployment rate, monthly, SA",
    "CPIAUCSL": "CPI, all urban consumers, SA (to state real values)",
}
STATES = {
    "01": ("AL", "Alabama"), "02": ("AK", "Alaska"), "04": ("AZ", "Arizona"), "05": ("AR", "Arkansas"),
    "06": ("CA", "California"), "08": ("CO", "Colorado"), "09": ("CT", "Connecticut"), "10": ("DE", "Delaware"),
    "11": ("DC", "District of Columbia"), "12": ("FL", "Florida"), "13": ("GA", "Georgia"), "15": ("HI", "Hawaii"),
    "16": ("ID", "Idaho"), "17": ("IL", "Illinois"), "18": ("IN", "Indiana"), "19": ("IA", "Iowa"),
    "20": ("KS", "Kansas"), "21": ("KY", "Kentucky"), "22": ("LA", "Louisiana"), "23": ("ME", "Maine"),
    "24": ("MD", "Maryland"), "25": ("MA", "Massachusetts"), "26": ("MI", "Michigan"), "27": ("MN", "Minnesota"),
    "28": ("MS", "Mississippi"), "29": ("MO", "Missouri"), "30": ("MT", "Montana"), "31": ("NE", "Nebraska"),
    "32": ("NV", "Nevada"), "33": ("NH", "New Hampshire"), "34": ("NJ", "New Jersey"), "35": ("NM", "New Mexico"),
    "36": ("NY", "New York"), "37": ("NC", "North Carolina"), "38": ("ND", "North Dakota"), "39": ("OH", "Ohio"),
    "40": ("OK", "Oklahoma"), "41": ("OR", "Oregon"), "42": ("PA", "Pennsylvania"), "44": ("RI", "Rhode Island"),
    "45": ("SC", "South Carolina"), "46": ("SD", "South Dakota"), "47": ("TN", "Tennessee"), "48": ("TX", "Texas"),
    "49": ("UT", "Utah"), "50": ("VT", "Vermont"), "51": ("VA", "Virginia"), "53": ("WA", "Washington"),
    "54": ("WV", "West Virginia"), "55": ("WI", "Wisconsin"), "56": ("WY", "Wyoming"),
}
# State unemployment rate, SA, e.g. WAUR
FRED_STATE_UNEMP = {f"{abbr}UR": name for abbr, name in STATES.values()}

# --- Modelling choices ------------------------------------------------------------------------
START = "2005-01-31"          # first month of the balanced county panel (keeps the 2006-11 bust in sample)
HORIZONS = (1, 3, 6, 12)      # months ahead, as Zillow's ZHVF (1, 3 and 12)


def metrics_dir() -> Path:
    """Recorded results go to reports/metrics; synthetic runs write beside the synthetic data instead."""
    return SYNTHETIC / "metrics" if use_synthetic() else METRICS


def figures_dir() -> Path:
    return SYNTHETIC / "figures" if use_synthetic() else FIGURES
