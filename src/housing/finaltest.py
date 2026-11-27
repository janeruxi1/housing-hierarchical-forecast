"""Final test: choose on early forecast dates, score once on later ones (docs/final_test_plan.md)."""
from __future__ import annotations

import numpy as np
import pandas as pd

SELECTION_END = "2021-10-31"
TEST_START = "2023-01-31"
PASS_MIN_WINS = 7          # of the 11 test dates, metro 12-month RMSE
N_TOP_METROS = 50


def period(origins: pd.Series) -> pd.Series:
    o = pd.to_datetime(origins)
    return pd.Series(np.select([o <= SELECTION_END, o >= TEST_START], ["selection", "test"], "gap"),
                     index=origins.index)


def choose(scores_selection: pd.DataFrame) -> tuple[str, pd.Series]:
    """Notebook 03's rule on the selection period: lowest RMSE averaged over levels x horizons."""
    coherent = scores_selection[scores_selection["method"].astype(str).str.contains(r"\+")]
    avg = coherent.groupby("method", observed=True)["rmse_pp"].mean().sort_values()
    return str(avg.index[0]), avg


def rmse_by_origin(res: pd.DataFrame, methods: list[str], level: str = "Metro", h: int = 12) -> pd.DataFrame:
    """origin x method table of RMSE of growth (pp) across the nodes of one level at one horizon."""
    r = res[res["method"].isin(methods) & (res["level"] == level) & (res["h"] == h)]
    t = rmse_table(r, by=("origin", "method"))
    return t.unstack("method")[methods]


def verdict(by_origin: pd.DataFrame, chosen: str, reference: str = "momentum") -> dict:
    """Apply the pass rule. by_origin holds the test dates only."""
    wins = int((by_origin[chosen] < by_origin[reference]).sum())
    pooled = {}  # RMSE over all test forecasts = sqrt(mean of squared per-date RMSE), equal nodes per date
    for m in (chosen, reference):
        pooled[m] = float(np.sqrt(np.mean(by_origin[m] ** 2)))
    return {
        "chosen": chosen, "reference": reference, "test_dates": int(len(by_origin)),
        "rmse_chosen_pp": pooled[chosen], "rmse_reference_pp": pooled[reference],
        "wins": wins, "rule_rmse_lower": pooled[chosen] < pooled[reference],
        "rule_min_wins": PASS_MIN_WINS, "rule_wins_met": wins >= PASS_MIN_WINS,
        "passes": bool(pooled[chosen] < pooled[reference] and wins >= PASS_MIN_WINS),
    }


def challenger_series(Y: pd.DataFrame, tags: dict, n_metros: int = N_TOP_METROS) -> list[str]:
    """The nation, every state and the largest metros by housing units (non-metro groups excluded)."""
    hu = Y.drop_duplicates("unique_id").set_index("unique_id")["hu"]
    metros = [m for m in tags["Metro"] if "Non-metro" not in m]
    return ["US", *sorted(tags["State"]), *hu.reindex(metros).nlargest(n_metros).index]


def tuning_origins(eval_origins: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Every fourth selection date (8): where the Prophet settings are compared."""
    return eval_origins[eval_origins <= SELECTION_END][::4]


def challenger_origins(eval_origins: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """All test dates plus every other selection date."""
    sel = eval_origins[eval_origins <= SELECTION_END][::2]
    test = eval_origins[eval_origins >= TEST_START]
    return sel.append(test)


def forecast_challengers(L: pd.DataFrame, series: list[str], origins, model: str, horizons=(1, 3, 6, 12),
                         n_jobs: int = 2) -> pd.DataFrame:
    """Long table (method, origin, h, unique_id, pred, actual) for one challenger model.

    Each model sees the log value of one series up to the origin, forecasts 12 months, and the
    growth to each horizon is scored against what happened, as in the rest of the backtest.
    """
    X = L[series]
    rows = []
    for o in origins:
        i = L.index.get_loc(o)
        hist = X.iloc[: i + 1]
        if model.startswith("prophet"):
            f = _prophet(hist, max(horizons), n_jobs, **PROPHET_SETTINGS[model])
        else:
            f = _statsforecast(hist, max(horizons), model, n_jobs)
        for h in horizons:
            if i + h >= len(L):
                continue
            rows.append(pd.DataFrame({
                "method": model, "origin": o, "h": np.int8(h), "unique_id": series,
                "pred": (f[h - 1] - hist.iloc[-1].to_numpy()).astype("float32"),
                "actual": (X.iloc[i + h].to_numpy() - hist.iloc[-1].to_numpy()).astype("float32"),
            }))
    return pd.concat(rows, ignore_index=True)


def _statsforecast(hist: pd.DataFrame, H: int, model: str, n_jobs: int) -> np.ndarray:
    import warnings

    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS

    m = {"autoets": AutoETS(season_length=1),
         "autoarima": AutoARIMA(season_length=1, approximation=True, max_p=2, max_q=2, max_P=0, max_Q=0,
                                nmodels=20)}[model]
    df = hist.reset_index().melt(id_vars=hist.index.name or "ds", var_name="unique_id", value_name="y")
    df = df.rename(columns={hist.index.name or "ds": "ds"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = StatsForecast(models=[m], freq="ME", n_jobs=n_jobs).forecast(df=df, h=H)
    col = [c for c in fc.columns if c not in ("unique_id", "ds")][0]
    wide = fc.pivot(index="ds", columns="unique_id", values=col)[list(hist.columns)]
    return wide.to_numpy()                                   # H x series


PROPHET_SETTINGS = {  # default, then the two tuned on selection dates only (see docs/final_test_plan.md)
    "prophet": {},
    "prophet_cr95": {"changepoint_range": 0.95},
    "prophet_cr95_flex": {"changepoint_range": 0.95, "changepoint_prior_scale": 0.5},
}


def _prophet_one(y: pd.Series, H: int, **kw) -> np.ndarray:
    import logging

    logging.getLogger("prophet.plot").disabled = True       # "plotly not installed" notice
    from prophet import Prophet

    logging.getLogger("cmdstanpy").disabled = True           # per-fit "Chain start/done" lines

    p = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
                uncertainty_samples=0, **kw)          # point forecasts only; bands are not scored
    p.fit(pd.DataFrame({"ds": y.index, "y": y.to_numpy()}))
    fut = pd.DataFrame({"ds": pd.date_range(y.index[-1], periods=H + 1, freq="ME")[1:]})
    return p.predict(fut)["yhat"].to_numpy()


def _prophet(hist: pd.DataFrame, H: int, n_jobs: int, **kw) -> np.ndarray:
    from joblib import Parallel, delayed

    out = Parallel(n_jobs=n_jobs)(delayed(_prophet_one)(hist[c], H, **kw) for c in hist.columns)
    return np.column_stack(out)


def win_counts(by_origin: pd.DataFrame, reference: str = "momentum") -> pd.Series:
    """How many dates each method beat the reference (lower RMSE)."""
    return pd.Series({m: int((by_origin[m] < by_origin[reference]).sum()) for m in by_origin if m != reference})



def level_of_id(ids: pd.Series) -> np.ndarray:
    s = ids.astype(str)
    return np.select([s == "US", s.str.startswith("ST:"), s.str.startswith("MSA:")], ["National", "State", "Metro"],
                     "County")


def rmse_table(res: pd.DataFrame, by=("method", "level", "h")) -> pd.Series:
    """RMSE of growth (pp) per group."""
    e = res["pred"].astype("float64") - res["actual"].astype("float64")
    return 100 * np.sqrt((e ** 2).groupby([res[b] for b in by], observed=True).mean())


def score_forward(frozen: pd.DataFrame, L: pd.DataFrame) -> pd.DataFrame:
    """Score frozen forecasts against whatever months have arrived since (empty until then)."""
    o = pd.Timestamp(frozen["origin"].iloc[0])
    out = []
    for h, g in frozen.groupby("h"):
        t = o + pd.offsets.MonthEnd(int(h))
        if t not in L.index or o not in L.index:
            continue
        act = 100 * (L.loc[t] - L.loc[o])
        out.append(g.assign(actual_pct=g["unique_id"].map(act).to_numpy()))
    return pd.concat(out, ignore_index=True) if out else frozen.iloc[:0].assign(actual_pct=[])
