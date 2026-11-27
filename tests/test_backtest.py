import numpy as np
import pandas as pd

from housing import backtest as bt


def _L(n=80, k=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2008-01-31", periods=n, freq="ME")
    return pd.DataFrame(np.cumsum(rng.normal(0.003, 0.01, (n, k)), axis=0), index=idx,
                        columns=["US", "ST:WA", "CTY:53033"])


def test_baseline_formulas():
    hist = np.log(np.array([[100.0], [101.0], [102.01]]))
    assert np.allclose(bt.naive(hist, 3), hist[-1])
    assert np.allclose(bt.drift(hist, 2) - hist[-1], 2 * np.log(1.01))
    assert np.allclose(bt.momentum(hist, 4, window=2) - hist[-1], 4 * np.log(1.01))


def test_no_peeking_at_the_future():
    """Changing values after an origin must not change forecasts made at that origin."""
    L = _L()
    a = bt.run(L, bt.BASELINES, horizons=(1, 3), first="2012-01-31")
    L2 = L.copy()
    L2.iloc[60:] += 5.0
    b = bt.run(L2, bt.BASELINES, horizons=(1, 3), first="2012-01-31")
    early = a["origin"] < L.index[60]
    assert np.allclose(a.loc[early, "pred"], b.loc[early, "pred"])


def test_actual_is_realised_growth_and_scores_are_sane():
    L = _L()
    res = bt.add_context(bt.run(L, bt.BASELINES, horizons=(1, 12)), bt.mase_scale(L, horizons=(1, 12)))
    r = res[(res["unique_id"] == "US") & (res["h"] == 12) & (res["method"] == "naive")].iloc[0]
    i = L.index.get_loc(r["origin"])
    assert np.isclose(r["actual"], L["US"].iloc[i + 12] - L["US"].iloc[i], atol=1e-6)
    assert np.isclose(r["err"], -r["actual"], atol=1e-6)   # naive predicts zero growth
    s = bt.score(res)
    assert (s["rmse_pp"] >= s["mae_pp"]).all()
    assert s.loc[s["method"] == "naive", "direction_hit"].isna().all()
    assert set(res["regime"].dropna().unique()) <= {r for _, _, r in bt.REGIMES}


def test_dm_test_and_bootstrap_interval():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2014-01-31", periods=48, freq="3ME")
    base = pd.Series(rng.gamma(2.0, 1.0, 48), index=idx)
    same = base * rng.uniform(0.95, 1.05, 48)
    _, p_same = bt.dm_test(same, base, lag=3)
    _, p_better = bt.dm_test(base * 0.5, base, lag=3)
    assert p_same > 0.2 and p_better < 0.01
    lo, hi = bt.block_bootstrap_skill(base * 0.5, base, block=4)
    point = 1 - np.sqrt(0.5)
    assert lo <= point + 1e-9 and point - 1e-9 <= hi
    assert bt.overlap_lag(12) == 3 and bt.overlap_lag(3) == 0 and bt.overlap_lag(1) == 0
