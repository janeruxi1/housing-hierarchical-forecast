import numpy as np
import pandas as pd

from housing import backtest as bt
from housing import models, panel, scenarios
from housing.config import data_dirs


def _setup():
    Y, S_df, tags, nodes = panel.load_hierarchy()
    L = bt.log_wide(Y)[list(S_df["unique_id"])]
    F = models.features(L, panel.load_macro(), pd.read_parquet(data_dirs()[1] / "state_unemp.parquet"), nodes)
    hu = Y.drop_duplicates("unique_id").set_index("unique_id")["hu"].reindex(L.columns)
    return Y, S_df, tags, L, F, hu


def test_scenarios_are_coherent_and_zero_shock_is_baseline(built):
    Y, S_df, tags, L, F, hu = _setup()
    now = scenarios.forecast_now(L, F, tags)
    counties = list(S_df.columns[1:])
    beta = scenarios.county_betas(L, counties, weights=hu)
    assert np.isclose(np.average(beta, weights=hu[counties]), 1.0)
    y0 = np.exp(L.iloc[-1])
    zero = scenarios.apply(now["combo"], S_df, y0, beta, pd.Series(0.0, index=scenarios.PATH_H))
    shock = scenarios.shock_path(now["county_coefs"]["full sample"], scenarios.SCENARIOS["Rates +1 pt"])
    up = scenarios.apply(now["combo"], S_df, y0, beta, shock)
    for vals in (zero, up):                       # nation = sum of states = sum of counties
        assert np.isclose(vals.loc["US", 12], vals.loc[tags["State"], 12].sum())
        assert np.isclose(vals.loc["US", 12], vals.loc[counties, 12].sum())
    # with no shock, counties are exactly the combo forecast
    assert np.allclose(np.log(zero.loc[counties, 12] / y0[counties]), now["combo"].loc[counties, 12])


def test_shock_is_linear_in_the_inputs(built):
    *_, L, F, hu = _setup()
    coefs = pd.DataFrame({12: {"d_rate12": -0.004, "rate_gap": -0.016, "d_unemp12": 0.005}})
    one = scenarios.shock_path(coefs, scenarios.SCENARIOS["Rates +1 pt"])
    two = scenarios.shock_path(coefs, scenarios.SCENARIOS["Rates +2 pt (stress)"])
    assert np.isclose(two[12], 2 * one[12]) and np.isclose(one[12], -0.02)
