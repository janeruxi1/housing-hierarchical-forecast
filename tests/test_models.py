"""The models use only what is known at each forecast date: changing the future leaves past forecasts unchanged."""
import numpy as np
import pandas as pd
import pytest

from housing import backtest as bt
from housing import models, panel
from housing.config import data_dirs


@pytest.fixture(scope="module")
def inputs(built):
    Y, S_df, tags, nodes = panel.load_hierarchy()
    L = bt.log_wide(Y)[list(S_df["unique_id"])]
    macro = panel.load_macro()
    su = pd.read_parquet(data_dirs()[1] / "state_unemp.parquet")
    levels = {lv: list(ids) for lv, ids in tags.items()}
    return L, macro, su, nodes, levels


def _forecasts(L, macro, su, nodes, levels, origins):
    F = models.features(L, macro, su, nodes)
    p, _ = models.panel(L, F, levels, origins)
    a = models.ar2(L, origins)
    us = L.columns.get_loc("US")
    z = models.zillow_style(L, {k: v[us] for k, v in p.items()}, origins)
    return {"panel": p, "ar2": a, "zillow_style": z}


def test_no_model_peeks_at_the_future(inputs):
    L, macro, su, nodes, levels = inputs
    origins = bt.origins(L)[::6]
    cut = origins[len(origins) // 2]
    L2, macro2, su2 = L.copy(), macro.copy(), su.copy()
    L2.loc[L2.index > cut] += 0.5                                     # home values jump after the cut
    num = macro2.columns.drop("date")
    macro2.loc[macro2["date"] > cut, num] += 3.0                     # rates and unemployment too
    su2.loc[su2["date"] > cut, "unrate"] += 3.0
    a = _forecasts(L, macro, su, nodes, levels, origins)
    b = _forecasts(L2, macro2, su2, nodes, levels, origins)
    checked = 0
    for m in a:
        for (o, h), v in a[m].items():
            if o <= cut:
                assert np.allclose(v, b[m][(o, h)], equal_nan=True, rtol=0, atol=1e-12), (m, o, h)
                checked += 1
    assert checked > 0
