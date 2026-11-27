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
    """Add error, level, target month, regime and MASE scale. Works in place (8M-row tables fit in 3 GB).

    Lookups go through the categorical codes of unique_id, so no per-row strings are created.
    """
    if not isinstance(res["unique_id"].dtype, pd.CategoricalDtype):
        res["unique_id"] = res["unique_id"].astype("category")
    cats = res["unique_id"].cat.categories.astype(str)
    codes = res["unique_id"].cat.codes.to_numpy()
    res["err"] = (res["pred"] - res["actual"]).astype("float32")
    lv = level_of(pd.Series(cats))
    res["level"] = pd.Categorical(lv.to_numpy()[codes], categories=["National", "State", "Metro", "County"])
    h = res["h"].to_numpy().astype(int)
    month = res["origin"].to_numpy().astype("datetime64[M]") + h.astype("timedelta64[M]")
    res["target"] = pd.to_datetime(month) + pd.offsets.MonthEnd(0)
    starts = np.array([np.datetime64(a, "M") for a, _, _ in REGIMES])
    k = np.searchsorted(starts, month, side="right") - 1
    labels = [r for _, _, r in REGIMES]
    res["regime"] = pd.Categorical.from_codes(np.where(k >= 0, k, -1), categories=labels)
    sc = scale.reindex(cats)
    res["scale"] = np.full(len(res), np.nan, dtype="float32")
    for hh in sc.columns:
        m = h == hh
        res.loc[m, "scale"] = sc[hh].to_numpy().astype("float32")[codes[m]]
    return res


def score(res: pd.DataFrame, by=("method", "level", "h")) -> pd.DataFrame:
    """RMSE and MAE of growth (pp), MASE, bias (pp) and direction hit rate, per group.

    MASE averages |error| / in-sample naive error per node, so a value below 1 means better than
    the naive rule would have been historically. Direction hit rate counts forecasts whose sign of
    growth matches the outcome (undefined for naive, which always predicts 0).
    """
    r = res[list(by) + ["err"]].assign(
        abs_err=res["err"].abs(), sq=res["err"].astype("float64") ** 2, ase=res["err"].abs() / res["scale"],
        hit=(np.sign(res["pred"]) == np.sign(res["actual"])).astype("float32"))
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


def conformal_coverage(res: pd.DataFrame, levels=(0.8, 0.95), min_past: int = 8) -> pd.DataFrame:
    """Time-safe prediction intervals for one method, and how often they cover the outcome.

    At each origin, for each node: interval = forecast +/- q x sd, where sd is the spread of the
    node's own past errors (origins whose outcome was known by then) and q is the quantile of
    |past error| / sd pooled over the nodes of the same level. No future error is ever used.
    Returns coverage and median width (pp) by level, horizon and nominal level.
    """
    rows = []
    for h, d in res.groupby("h"):
        E = d.pivot(index="origin", columns="unique_id", values="err").sort_index()
        lv = level_of(E.columns.to_series().astype(str)).to_numpy()
        origins_ = E.index
        target_end = origins_.values.astype("datetime64[M]") + np.timedelta64(int(h), "M")
        Ev = E.to_numpy()
        for k, o in enumerate(origins_):
            past = target_end < np.datetime64(o, "M") + np.timedelta64(1, "M")
            if past.sum() < min_past:
                continue
            P = Ev[past]
            sd = P.std(0, ddof=1)
            z = np.abs(P) / sd
            for level in np.unique(lv):
                m = lv == level
                for nominal in levels:
                    q = np.nanquantile(z[:, m], nominal)
                    half = q * sd[m]
                    rows.append({"h": h, "level": level, "nominal": nominal, "origin": o,
                                 "covered": np.nanmean(np.abs(Ev[k, m]) <= half),
                                 "width_pp": 200 * np.nanmedian(half)})
    out = pd.DataFrame(rows)
    return out.groupby(["level", "h", "nominal"]).agg(coverage=("covered", "mean"), median_width_pp=("width_pp", "median"),
                                                       origins=("origin", "nunique")).reset_index()


# --- how sure is a difference in accuracy? ----------------------------------------------------
def per_date_mse(res: pd.DataFrame, method: str, level: str, h: int) -> pd.Series:
    """Mean squared growth error across a level's nodes, one value per forecast date.

    Comparing methods date by date (not forecast by forecast) respects that all nodes at one date
    share the same shock: the independent unit is the date, not the node.
    """
    r = res[(res["method"] == method) & (res["level"] == level) & (res["h"] == h)]
    e = r["pred"].astype("float64") - r["actual"].astype("float64")
    return (e ** 2).groupby(r["origin"], observed=True).mean()


def dm_test(loss_a: pd.Series, loss_b: pd.Series, lag: int) -> tuple[float, float]:
    """Diebold-Mariano test of equal accuracy, with the Harvey-Leybourne-Newbold correction.

    The loss differential's variance uses Bartlett (Newey-West) weights up to `lag`: forecasts
    whose outcome windows overlap (12-month outcomes from quarterly dates overlap 3 dates) are not
    independent. Returns (statistic, two-sided p-value from a t distribution with n - 1 df).
    """
    from scipy import stats

    d = (loss_a - loss_b).dropna().to_numpy()
    n = len(d)
    dc = d - d.mean()
    gamma = [dc @ dc / n] + [dc[k:] @ dc[:-k] / n for k in range(1, lag + 1)]
    var = gamma[0] + 2 * sum((1 - k / (lag + 1)) * gamma[k] for k in range(1, lag + 1))
    if var <= 0:
        return float("nan"), float("nan")
    hln = np.sqrt((n + 1 - 2 * (lag + 1) + (lag + 1) * lag / n) / n)
    stat = hln * d.mean() / np.sqrt(var / n)
    return float(stat), float(2 * stats.t.sf(abs(stat), n - 1))


def block_bootstrap_skill(loss_a: pd.Series, loss_b: pd.Series, block: int, reps: int = 4000,
                          seed: int = 0) -> tuple[float, float]:
    """95% interval for skill = 1 - RMSE_a / RMSE_b, resampling runs of `block` consecutive dates."""
    a, b = loss_a.to_numpy(), loss_b.to_numpy()
    n = len(a)
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(reps, k))
    idx = (starts[..., None] + np.arange(block)).reshape(reps, -1)[:, :n]
    skill = 1 - np.sqrt(a[idx].mean(1) / b[idx].mean(1))
    lo, hi = np.percentile(skill, [2.5, 97.5])
    return float(lo), float(hi)


def overlap_lag(h: int, step_months: int = 3) -> int:
    """How many later forecast dates share part of an h-month outcome window."""
    return max(0, int(np.ceil(h / step_months)) - 1)
