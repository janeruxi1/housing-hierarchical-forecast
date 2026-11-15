import numpy as np
import pandas as pd

from housing import panel
from housing.hierarchy import level_of


def test_every_level_adds_up_to_the_nation(built):
    Y, S, tags, _ = panel.load_hierarchy()
    last = Y[Y["ds"] == Y["ds"].max()].set_index("unique_id")["y"]
    us = last["US"]
    for level in ("State", "Metro", "County"):
        assert np.isclose(last[tags[level]].sum(), us, rtol=1e-9), level


def test_each_county_has_exactly_one_parent_per_grouping(built):
    _, S, tags, _ = panel.load_hierarchy()
    S = S.set_index("unique_id")
    for level in ("State", "Metro"):
        assert (S.loc[tags[level]].sum(axis=0) == 1).all(), level


def test_typical_value_is_unit_weighted_mean_of_county_zhvi(built):
    Y, _, _, nodes = panel.load_hierarchy()
    c = panel.load_counties()
    last = c[c["date"] == c["date"].max()].assign(County="CTY:" + c["fips"]).merge(nodes, on="County")
    for st, g in last.groupby("State"):
        expected = np.average(g["zhvi"], weights=g["hu"])
        got = Y.loc[(Y["unique_id"] == st) & (Y["ds"] == c["date"].max()), "typical_value"].item()
        assert np.isclose(got, expected), st


def test_panel_is_balanced_from_start(built):
    c = panel.load_counties()
    assert c.groupby("fips")["date"].count().nunique() == 1
    assert built["counties_full_history"] < built["counties_in_zillow"]   # late starters dropped
    assert built["months_interpolated"] > 0                                # short holes filled


def test_level_of():
    ids = pd.Series(["US", "ST:WA", "MSA:Seattle, WA", "CTY:53033"])
    assert level_of(ids).tolist() == ["National", "State", "Metro", "County"]
