"""Forecast from the latest month, and what-if scenarios on top of it.

The baseline is the chosen method from notebook 03 (combo = mean of ar2 and panel, made coherent
bottom-up), run from the last month of data for 1-12 months ahead.

A scenario changes the inputs the panel model reads at the forecast date (mortgage rates,
unemployment) and asks how its forecast moves. The shift is computed from the *county* panel
regression, the one fitted on the most data (1,371 counties x ~200 months), and then scaled per
county by its sensitivity to the national cycle (beta), so that places that swing harder with the
nation also take a harder hit. Counties are then added up, so every level stays coherent.

ar2 has no view on rates, so the scenario shift comes entirely from the panel model, applied in
full on top of the combined baseline. This is the assumption to state when quoting a scenario.

Like a cat model: the shock (rates up) x the exposure (each county's beta) = the loss (value change).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from housing import models

PATH_H = tuple(range(1, 13))
PANDEMIC = ("2020-03-31", "2021-12-31")

# Changes to the panel model's inputs at the forecast date. A sudden 1-point rise in mortgage rates
# raises both the 12-month change and the gap to the 3-year average by 1 point.
SCENARIOS = {
    "Rates +1 pt": {"d_rate12": 1.0, "rate_gap": 1.0},
    "Rates +2 pt (stress)": {"d_rate12": 2.0, "rate_gap": 2.0},
    "Rates -1 pt": {"d_rate12": -1.0, "rate_gap": -1.0},
    "Recession (unemployment +3, rates -1)": {"d_unemp12": 3.0, "d_rate12": -1.0, "rate_gap": -1.0},
}


def forecast_now(L: pd.DataFrame, F: dict, tags: dict, horizons=PATH_H) -> dict:
    """Combo forecasts for every node from the last month, plus county panel coefficients
    (full training sample and with the pandemic years left out)."""
    o = pd.DatetimeIndex([L.index[-1]])
    levels = {lv: list(ids) for lv, ids in tags.items()}
    ar = models.ar2(L, o, horizons)
    pn, coefs = models.panel(L, F, levels, o, horizons)
    _, coefs_ex = models.panel(L, F, {"County": levels["County"]}, o, horizons, exclude=PANDEMIC)
    combo = {h: 0.5 * (ar[(o[0], h)] + pn[(o[0], h)]) for h in horizons}
    county_coefs = {"full sample": pd.DataFrame({h: coefs[("County", h)] for h in horizons}),
                    "pandemic left out": pd.DataFrame({h: coefs_ex[("County", h)] for h in horizons})}
    return {"origin": o[0], "combo": pd.DataFrame(combo, index=L.columns), "county_coefs": county_coefs}


def county_betas(L: pd.DataFrame, counties: list[str], start: str = "2009-01-31", shrink_k: float = 36.0,
                 weights: pd.Series | None = None) -> pd.Series:
    """Sensitivity of each county's 12-month growth to national 12-month growth, shrunk towards 1,
    then rescaled so the housing-unit-weighted average is exactly 1 (the national shock is unchanged)."""
    g = L.diff(12).loc[start:]
    x = g["US"]
    y = g[counties]
    xm, ym = x.mean(), y.mean()
    beta = ((y - ym).mul(x - xm, axis=0)).sum() / ((x - xm) ** 2).sum()
    n = y.notna().sum()
    beta = (n * beta + shrink_k) / (n + shrink_k)
    w = weights.reindex(counties) if weights is not None else pd.Series(1.0, index=counties)
    return beta / np.average(beta, weights=w)


def shock_path(coefs: pd.DataFrame, deltas: dict) -> pd.Series:
    """Change in h-month log growth (by h) for the average county: coefficients x input changes."""
    return sum(coefs.loc[f] * d for f, d in deltas.items())


def apply(base: pd.DataFrame, S_df: pd.DataFrame, y0: pd.Series, beta: pd.Series, shock: pd.Series) -> pd.DataFrame:
    """Coherent scenario values ($) for every node and horizon.

    base: combo log-growth forecasts (node x h); y0: values at the forecast date; shock: by h.
    County growth = base + beta x shock; every other node is the sum of its counties.
    """
    counties = list(S_df.columns[1:])
    g = base.loc[counties] + np.outer(beta.reindex(counties), shock.reindex(base.columns))
    vals = np.exp(g).mul(y0.reindex(counties), axis=0)
    total = S_df.iloc[:, 1:].to_numpy() @ vals.to_numpy()
    return pd.DataFrame(total, index=S_df["unique_id"], columns=base.columns)
