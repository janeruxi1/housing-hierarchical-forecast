"""Rolling-origin backtest: forecast from every month-end origin, score against what happened.

Everything works on log housing-stock value, one column per node, so a forecast error at horizon h
is directly an error in cumulative h-month growth (in log points, ~ percentage points / 100).

Baselines (each is a rule a planner could apply with no model):
* naive     - no change: the value stays where it is.
* drift     - the average monthly growth since the start of the panel continues.
* momentum  - the last 12 months' average monthly growth continues. ZHVI is smoothed, so recent
              growth persists; this is the baseline to beat in calm periods.
* long_run  - the median monthly growth over the last 4 years continues (the long-run anchor in
              Zillow's published method).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from housing.config import HORIZONS
from housing.hierarchy import level_of

FIRST_ORIGIN = "2012-01-31"
REGIMES = [  # by target month
    ("2012-01-31", "2020-05-31", "Recovery 2012-20"),
    ("2020-06-30", "2022-06-30", "Pandemic boom"),
    ("2022-07-31", "2023-12-31", "Rate shock"),
    ("2024-01-31", "2099-12-31", "Flat 2024+"),
]


def log_wide(Y: pd.DataFrame) -> pd.DataFrame:
    """ds x unique_id table of log total value."""
    return np.log(Y.pivot(index="ds", columns="unique_id", values="y")).sort_index()


# A baseline maps (log history up to and including the origin, horizon) -> log forecast per node.
Baseline = Callable[[np.ndarray, int], np.ndarray]


def naive(hist: np.ndarray, h: int) -> np.ndarray:
    return hist[-1]


def drift(hist: np.ndarray, h: int) -> np.ndarray:
    return hist[-1] + h * (hist[-1] - hist[0]) / (len(hist) - 1)


def momentum(hist: np.ndarray, h: int, window: int = 12) -> np.ndarray:
    return hist[-1] + h * (hist[-1] - hist[-1 - window]) / window


def long_run(hist: np.ndarray, h: int, window: int = 48) -> np.ndarray:
    return hist[-1] + h * np.median(np.diff(hist[-window - 1:], axis=0), axis=0)


BASELINES: dict[str, Baseline] = {"naive": naive, "drift": drift, "momentum": momentum, "long_run": long_run}


def origins(L: pd.DataFrame, first: str = FIRST_ORIGIN, max_h: int = max(HORIZONS)) -> pd.DatetimeIndex:
    """Every month-end from `first` that still has the longest horizon's outcome observed."""
    return L.index[(L.index >= first) & (L.index <= L.index[-1 - max_h])]


def run(L: pd.DataFrame, methods: dict[str, Baseline], horizons=HORIZONS, first: str = FIRST_ORIGIN) -> pd.DataFrame:
    """Long table: method, origin, h, unique_id, pred (forecast growth), actual (realised growth).

    Growth is cumulative log growth from the origin to origin + h. Errors = pred - actual.
    """
    X = L.to_numpy()
    idx = {d: i for i, d in enumerate(L.index)}
    out = []
    for o in origins(L, first, max(horizons)):
        i = idx[o]
        hist = X[: i + 1]
        for h in horizons:
            actual = X[i + h] - X[i]
            for name, f in methods.items():
                out.append(pd.DataFrame({"method": name, "origin": o, "h": h, "unique_id": L.columns,
                                         "pred": (f(hist, h) - X[i]).astype("float32"),
                                         "actual": actual.astype("float32")}))
    res = pd.concat(out, ignore_index=True)
    res["method"] = res["method"].astype("category")
    res["unique_id"] = res["unique_id"].astype("category")
    return res


def mase_scale(L: pd.DataFrame, horizons=HORIZONS, before: str = FIRST_ORIGIN) -> pd.DataFrame:
    """Per node and horizon: mean |h-month change| before the first origin (the in-sample naive error)."""
    train = L[L.index < before]
    return pd.DataFrame({h: (train.diff(h).abs().mean()) for h in horizons}).rename_axis("unique_id")


def add_context(res: pd.DataFrame, scale: pd.DataFrame) -> pd.DataFrame:
    res = res.copy()
    res["err"] = res["pred"] - res["actual"]
    res["level"] = level_of(res["unique_id"].astype(str)).astype("category")
    res["target"] = res["origin"].values.astype("datetime64[M]") + res["h"].values.astype("timedelta64[M]")
    res["target"] = pd.to_datetime(res["target"]) + pd.offsets.MonthEnd(0)
    res["regime"] = pd.cut(res["target"], bins=[pd.Timestamp(a) - pd.Timedelta(days=1) for a, _, _ in REGIMES]
                           + [pd.Timestamp(REGIMES[-1][1])], labels=[r for _, _, r in REGIMES])
    s = scale.stack().rename("scale").reset_index().rename(columns={"level_1": "h"})
    s["unique_id"] = s["unique_id"].astype(str)
    res = res.merge(s, on=["unique_id", "h"], how="left") if len(res) < 2e6 else _merge_scale(res, scale)
    return res


def _merge_scale(res: pd.DataFrame, scale: pd.DataFrame) -> pd.DataFrame:
    """Memory-friendly lookup for big result tables."""
    res["scale"] = np.nan
    uid = res["unique_id"].astype(str)
    for h in scale.columns:
        m = res["h"].to_numpy() == h
        res.loc[m, "scale"] = uid[m].map(scale[h]).to_numpy()
    return res


def score(res: pd.DataFrame, by=("method", "level", "h")) -> pd.DataFrame:
    """RMSE and MAE of growth (pp), MASE, bias (pp) and direction hit rate, per group.

    MASE averages |error| / in-sample naive error per node, so a value below 1 means better than
    the naive rule would have been historically. Direction hit rate counts forecasts whose sign of
    growth matches the outcome (undefined for naive, which always predicts 0).
    """
    r = res.assign(abs_err=res["err"].abs(), sq=res["err"] ** 2, ase=res["err"].abs() / res["scale"],
                   hit=(np.sign(res["pred"]) == np.sign(res["actual"])).astype(float))
    g = r.groupby(list(by), observed=True)
    out = pd.DataFrame({
        "rmse_pp": 100 * np.sqrt(g["sq"].mean()),
        "mae_pp": 100 * g["abs_err"].mean(),
        "bias_pp": 100 * g["err"].mean(),
        "mase": g["ase"].mean(),
        "direction_hit": g["hit"].mean(),
        "n": g.size(),
    }).reset_index()
    if "method" in out:
        out.loc[out["method"] == "naive", "direction_hit"] = np.nan
    return out


def skill_vs(scores: pd.DataFrame, reference: str = "naive", metric: str = "rmse_pp") -> pd.DataFrame:
    """1 - metric / reference metric, in the same grouping (positive = better than the reference)."""
    keys = [c for c in ("level", "h", "regime") if c in scores]
    ref = scores[scores["method"] == reference].set_index(keys)[metric]
    s = scores.set_index(keys)
    s["skill_vs_" + reference] = 1 - s[metric] / ref.reindex(s.index).to_numpy()
    return s.reset_index()


# Zillow's published accuracy for its metro forecast: top 100 metros, forecasts made Jan 2019 -
# Nov 2021, RMSE of growth (as a fraction) at each horizon. Source: Zillow Research, "Zillow's Open
# Market Home Price Appreciation Forecasting Methodology".
ZILLOW_PUBLISHED_RMSE = {1: 0.0105, 3: 0.0213, 6: 0.0388}
ZILLOW_WINDOW = ("2019-01-31", "2021-11-30")


def zillow_comparison(res: pd.DataFrame, Y: pd.DataFrame, top: int = 100) -> pd.DataFrame:
    """Our RMSE for the largest `top` metros (by housing units) over Zillow's window, next to theirs."""
    hu = Y[Y["unique_id"].str.startswith("MSA:") & ~Y["unique_id"].str.contains("Non-metro")]
    biggest = hu.drop_duplicates("unique_id").nlargest(top, "hu")["unique_id"]
    a, b = (pd.Timestamp(x) for x in ZILLOW_WINDOW)
    r = res[res["unique_id"].isin(biggest) & res["origin"].between(a, b) & res["h"].isin(ZILLOW_PUBLISHED_RMSE)]
    out = r.groupby(["method", "h"], observed=True)["err"].apply(lambda e: np.sqrt(np.mean(e ** 2))).unstack("h")
    out.loc["Zillow (published)"] = pd.Series(ZILLOW_PUBLISHED_RMSE)
    return out
