"""Run every model and reconciliation through the rolling backtest; save the long result tables.

Base forecasts come from every month-end from 2012 (cheap, and they supply the past errors the
reconciliation weights need). Reconciliation and the head-to-head comparison use quarterly origins
from 2014 (48 forecast dates), when every model has at least four years of training data.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from housing import backtest as bt
from housing import models, panel
from housing.config import HORIZONS, data_dirs
from housing.reconcile import Hierarchy, shrinkage_lambda, standardise

EVAL_FIRST = "2014-01-31"
BASES = ["momentum", "ar2", "panel", "zillow_style", "combo"]
RECONCILERS = ["bottom_up", "wls_struct", "wls_var", "mint_shrink"]
ERROR_WINDOW = 60          # past origins (months) used to estimate reconciliation weights
MIN_PAST = 12              # fewer past errors than this: wls_var / mint fall back to wls_struct


def eval_origins(L: pd.DataFrame) -> pd.DatetimeIndex:
    o = bt.origins(L, EVAL_FIRST)
    return o[(o.month - 1) % 3 == 0]


def base_forecasts(L, F, tags, horizons=HORIZONS):
    origins = bt.origins(L, bt.FIRST_ORIGIN)
    X = L.to_numpy()
    preds = {"naive": {}, "momentum": {}}
    for o in origins:
        i = L.index.get_loc(o)
        for h in horizons:
            preds["naive"][(o, h)] = np.zeros(L.shape[1])
            preds["momentum"][(o, h)] = bt.momentum(X[: i + 1], h) - X[i]
    preds["ar2"] = models.ar2(L, origins, horizons)
    levels = {lv: list(ids) for lv, ids in tags.items()}
    preds["panel"], coefs = models.panel(L, F, levels, origins, horizons)
    us = L.columns.get_loc("US")
    national = {k: v[us] for k, v in preds["panel"].items()}
    preds["zillow_style"] = models.zillow_style(L, national, origins, horizons)
    # Equal-weight average of the two best single models: ar2 reads each place's own rhythm,
    # panel reads rates and the cycle; their errors are only partly correlated.
    preds["combo"] = {k: 0.5 * (preds["ar2"][k] + preds["panel"][k]) for k in preds["ar2"]}
    return preds, coefs


def reconcile_all(L, S_df, preds, horizons=HORIZONS):
    """{base: {(origin, h): forecast}} -> {f"{base}+{rec}": {(origin, h): forecast}} on eval origins."""
    order = list(S_df["unique_id"])
    assert order == list(L.columns)
    n_bottom = S_df.shape[1] - 1
    H = Hierarchy(S_df.iloc[:, 1:].to_numpy(float), n_agg=len(order) - n_bottom)
    X = L.to_numpy()
    all_origins = sorted({o for o, _ in preds["momentum"]})
    out, fallback = {}, 0
    for base in BASES:
        P = preds[base]
        for rec in RECONCILERS:
            out[f"{base}+{rec}"] = {}
        for o in eval_origins(L):
            i = L.index.get_loc(o)
            y0 = np.exp(X[i])
            for h in horizons:
                yhat = y0 * np.exp(P[(o, h)])
                past = [p for p in all_origins if p < o and L.index.get_loc(p) + h <= i][-ERROR_WINDOW:]
                E = np.array([P[(p, h)] - (X[L.index.get_loc(p) + h] - X[L.index.get_loc(p)]) for p in past])
                res = {"bottom_up": H.bottom_up(yhat), "wls_struct": H.wls(yhat, H.struct)}
                if len(past) >= MIN_PAST and np.isfinite(E).all():
                    Z, sd = standardise(E)
                    d = sd * yhat
                    res["wls_var"] = H.wls(yhat, d ** 2)
                    res["mint_shrink"] = H.mint_lowrank(yhat, d, Z, shrinkage_lambda(Z))
                else:
                    fallback += 1
                    res["wls_var"] = res["mint_shrink"] = res["wls_struct"]
                for rec, ytil in res.items():
                    with np.errstate(invalid="ignore", divide="ignore"):
                        out[f"{base}+{rec}"][(o, h)] = np.log(np.where(ytil > 0, ytil, np.nan) / y0)
    return out, fallback


def run_all(save: bool = True) -> dict:
    """Returns the scored-date results; long tables use categorical ids and float32 to stay small."""
    t0 = time.time()
    Y, S_df, tags, nodes = panel.load_hierarchy()
    L = bt.log_wide(Y)[list(S_df["unique_id"])]
    macro = panel.load_macro()
    su = pd.read_parquet(data_dirs()[1] / "state_unemp.parquet")
    F = models.features(L, macro, su, nodes)
    preds, coefs = base_forecasts(L, F, tags)
    t1 = time.time()
    rec, fallback = reconcile_all(L, S_df, preds)
    t2 = time.time()
    evo = set(eval_origins(L))   # only the scored forecast dates are kept
    parts = [models.to_long(p, L, m, evo) for m, p in {**preds, **rec}.items()]
    del preds, rec
    long = pd.concat(parts, ignore_index=True)
    del parts
    long["method"] = long["method"].astype("category")
    long["unique_id"] = long["unique_id"].astype("category")
    info = {"base_seconds": round(t1 - t0, 1), "reconcile_seconds": round(t2 - t1, 1),
            "reconcile_fallbacks": fallback, "rows": len(long)}
    coef_df = pd.DataFrame(coefs).T.rename_axis(["level", "h"]).reset_index()
    if save:
        out = data_dirs()[1]
        long.to_parquet(out / "backtest_models.parquet", index=False)
        coef_df.to_parquet(out / "panel_coefficients.parquet", index=False)
    return {"long": long, "coefs": coef_df, "info": info, "L": L, "Y": Y}
