"""Base forecasts for every node at once, in log growth, from every origin.

All three models are fitted with closed-form least squares on numpy arrays, so a full rolling
backtest (every month from 2012, 2,000+ nodes, four horizons) runs in seconds rather than hours.

* ar2        - ARIMA(2,1,0) with drift per node: monthly growth regressed on its last two
               months over a 5-year rolling window, iterated forward. Classic single-series
               time-series model.
* panel      - direct multi-horizon panel regression, pooled across the nodes of each level:
               h-month growth on recent growth (1, 3, 12 months and the 12 months before that),
               national 12-month growth, the change in mortgage rates over the last year, rates
               against their 3-year average, and the 12-month change in unemployment (state where
               the node has one). "Direct" means one regression per horizon, so no future
               mortgage rate or unemployment is ever needed: only what is known at the origin.
* zillow_style - Zillow's published structure: a national forecast (here the panel model's) plus,
               for each node, its sensitivity (beta) to national growth, shrunk towards 1, and a
               mean-reverting local residual.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from housing.config import HORIZONS

FEATURES = ["g1", "g3", "g12", "g12_prev", "us12", "d_rate12", "rate_gap", "d_unemp12"]
TRAIN_START = "2010-01-31"   # first month with every feature defined (needs 24 months of history)


# --- features ---------------------------------------------------------------------------------
def node_state(unique_ids: pd.Index, nodes: pd.DataFrame) -> pd.Series:
    """The state each node belongs to ('' for the nation; a metro's first-named state)."""
    cty = dict(zip(nodes["County"], nodes["State"].str[3:]))
    out = {}
    for u in unique_ids:
        if u == "US":
            out[u] = ""
        elif u.startswith("ST:"):
            out[u] = u[3:]
        elif u.startswith("CTY:"):
            out[u] = cty[u]
        else:  # "MSA:Seattle, WA" or "MSA:Non-metro WA"
            out[u] = u[-2:]
    return pd.Series(out)


def features(L: pd.DataFrame, macro: pd.DataFrame, state_unemp: pd.DataFrame, nodes: pd.DataFrame) -> dict:
    """Wide (month x node) feature tables, all known at the end of each month."""
    idx, cols = L.index, L.columns

    def bcast(s: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(np.repeat(s.reindex(idx).to_numpy()[:, None], len(cols), 1), index=idx, columns=cols)

    m = macro.set_index("date")
    rate = m["MORTGAGE30US"].reindex(idx)
    us_u = m["UNRATE"].reindex(idx)
    su = state_unemp.pivot(index="date", columns="state", values="unrate").reindex(idx)
    st = node_state(cols, nodes)
    d_unemp = pd.DataFrame({c: (su[st[c]] if st[c] in su else us_u).diff(12) for c in cols})
    return {
        "g1": L.diff(1), "g3": L.diff(3), "g12": L.diff(12), "g12_prev": L.diff(12).shift(12),
        "us12": bcast(L["US"].diff(12)),
        "d_rate12": bcast(rate.diff(12)),
        "rate_gap": bcast(rate - rate.rolling(36).mean()),
        "d_unemp12": d_unemp,
    }


# --- AR(2) per node ---------------------------------------------------------------------------
def ar2(L: pd.DataFrame, origins: pd.DatetimeIndex, horizons=HORIZONS, window: int = 60) -> dict:
    """{(origin, h): log-growth forecast per node}. Persistence is capped at 0.98 so no path explodes."""
    g = L.diff().to_numpy()
    out = {}
    for o in origins:
        i = L.index.get_loc(o)
        w = min(window, i - 2)                  # expanding until 5 years are available (row 0 is NaN)
        y = g[i - w + 1: i + 1]
        x1, x2 = g[i - w: i], g[i - w - 1: i - 1]
        X = np.stack([np.ones_like(y), x1, x2], axis=-1)            # window x N x 3
        XtX = np.einsum("tnk,tnl->nkl", X, X) + 1e-10 * np.eye(3)
        Xty = np.einsum("tnk,tn->nk", X, y)
        b = np.linalg.solve(XtX, Xty[..., None])[..., 0]            # N x 3
        persist = b[:, 1] + b[:, 2]
        shrink = np.ones_like(persist)
        np.divide(0.98, persist, out=shrink, where=persist > 0.98)
        b[:, 1] *= shrink
        b[:, 2] *= shrink
        last, prev = g[i], g[i - 1]
        cum = np.zeros(len(last))
        path = {}
        for step in range(1, max(horizons) + 1):
            nxt = b[:, 0] + b[:, 1] * last + b[:, 2] * prev
            cum = cum + nxt
            prev, last = last, nxt
            path[step] = cum.copy()
        for h in horizons:
            out[(o, h)] = path[h]
    return out


# --- direct panel regression -------------------------------------------------------------------
def _cum_normal_equations(X: np.ndarray, y: np.ndarray, valid: np.ndarray):
    """Running sums over time of X'X and X'y (rows = months, pooled over nodes)."""
    Xv = np.where(valid[..., None], X, 0.0)
    yv = np.where(valid, y, 0.0)
    XtX = np.cumsum(np.einsum("tnk,tnl->tkl", Xv, Xv), axis=0)
    Xty = np.cumsum(np.einsum("tnk,tn->tk", Xv, yv), axis=0)
    return XtX, Xty


def panel(L: pd.DataFrame, F: dict, levels: dict, origins: pd.DatetimeIndex, horizons=HORIZONS,
          ridge: float = 1e-4, exclude: tuple[str, str] | None = None) -> tuple[dict, dict]:
    """Direct h-step pooled OLS per level. Returns ({(origin, h): forecast per node}, coefficients).

    `exclude` drops training rows whose *target window* overlaps a period (e.g. the 2020-21
    pandemic, when unemployment spiked while prices boomed), to test how much a coefficient
    depends on it.

    Training rows for origin o and horizon h are months t with t + h <= o, so every target used is
    fully observed at the origin. Features are standardised with the training-period scale.
    """
    T = len(L.index)
    start = L.index.get_loc(L.index[L.index >= TRAIN_START][0])
    preds = {(o, h): np.full(L.shape[1], np.nan) for o in origins for h in horizons}
    coefs = {}
    for level, ids in levels.items():
        cols = [L.columns.get_loc(c) for c in ids]
        Xraw = np.stack([F[f].to_numpy()[:, cols] for f in FEATURES], axis=-1)     # T x n x k
        scale = np.nanstd(Xraw[start:], axis=(0, 1))
        scale[scale == 0] = 1
        X = np.concatenate([np.ones(Xraw.shape[:2] + (1,)), Xraw / scale], axis=-1)
        Lv = L.to_numpy()[:, cols]
        for h in horizons:
            y = np.full_like(Lv, np.nan)
            y[:-h] = Lv[h:] - Lv[:-h]
            valid = np.isfinite(X).all(-1) & np.isfinite(y)
            valid[:start] = False
            if exclude is not None:
                a, b = (pd.Timestamp(x) for x in exclude)
                t = L.index
                overlap = (t + pd.offsets.MonthEnd(h) >= a) & (t <= b)
                valid[overlap] = False
            XtX, Xty = _cum_normal_equations(np.nan_to_num(X), np.nan_to_num(y), valid)
            k = X.shape[-1]
            pen = ridge * np.eye(k)
            pen[0, 0] = 0
            for o in origins:
                i = L.index.get_loc(o)
                j = i - h                                   # last training month: target ends at o
                if j < start or j >= T:
                    continue
                n_rows = valid[: j + 1].sum()
                b = np.linalg.solve(XtX[j] + pen * max(n_rows, 1), Xty[j])
                pred = X[i] @ b
                preds[(o, h)][cols] = pred
                if o == origins[-1]:
                    coefs[(level, h)] = pd.Series(b, index=["const"] + FEATURES) / np.r_[1, scale]
    return preds, coefs


# --- Zillow-style: national forecast x shrunk beta + local residual ------------------------------
def zillow_style(L: pd.DataFrame, national: dict, origins: pd.DatetimeIndex, horizons=HORIZONS,
                 window: int = 60, shrink_k: float = 36.0) -> dict:
    """national: {(origin, h): forecast for 'US'} (e.g. from the panel model).

    For each node over the last `window` months: monthly growth = a + beta x national growth + r.
    beta is shrunk towards 1 (k pseudo-months at beta = 1); the residual r decays at a pooled AR(1)
    rate phi, so a node running hot relative to the nation cools back towards its usual gap.
    """
    g = L.diff().to_numpy()
    us = g[:, L.columns.get_loc("US")]
    out = {}
    for o in origins:
        i = L.index.get_loc(o)
        w = min(window, i)
        y, x = g[i - w + 1: i + 1], us[i - w + 1: i + 1][:, None]
        xm, ym = x.mean(0), y.mean(0)
        beta = ((x - xm) * (y - ym)).sum(0) / ((x - xm) ** 2).sum(0)
        beta = (w * beta + shrink_k * 1.0) / (w + shrink_k)
        alpha = ym - beta * xm
        r = y - alpha - beta * x
        phi = np.clip((r[1:] * r[:-1]).sum() / (r[:-1] ** 2).sum(), 0, 0.98)
        for h in horizons:
            decay = phi * (1 - phi ** h) / (1 - phi) if phi > 0 else 0.0
            f = alpha * h + beta * national[(o, h)] + r[-1] * decay
            f[L.columns.get_loc("US")] = national[(o, h)]
            out[(o, h)] = f
    return out


def to_long(preds: dict, L: pd.DataFrame, method: str, origins=None) -> pd.DataFrame:
    """{(origin, h): array} -> compact long table with realised growth, like backtest.run().

    Categorical ids and float32 values keep the 8 million-row backtest within a laptop's memory.
    """
    X = L.to_numpy()
    ids = pd.Categorical(L.columns)
    rows = []
    for (o, h), p in preds.items():
        i = L.index.get_loc(o)
        if i + h >= len(L) or (origins is not None and o not in origins):
            continue
        rows.append(pd.DataFrame({"origin": o, "h": np.int8(h), "unique_id": ids,
                                  "pred": p.astype("float32"), "actual": (X[i + h] - X[i]).astype("float32")}))
    out = pd.concat(rows, ignore_index=True)
    out.insert(0, "method", pd.Categorical([method] * len(out)))
    return out
