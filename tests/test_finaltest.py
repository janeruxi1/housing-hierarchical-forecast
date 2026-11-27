"""The final-test rules (docs/final_test_plan.md): periods, the choice, the pass rule, frozen forecasts."""
import numpy as np
import pandas as pd

from housing import finaltest as ft

Q = pd.date_range("2014-01-31", "2025-09-30", freq="3ME")          # the 47 quarterly forecast dates


def test_periods_split_32_gap_4_test_11():
    p = ft.period(pd.Series(Q)).value_counts()
    assert (p["selection"], p["gap"], p["test"]) == (32, 4, 11)


def test_challenger_and_tuning_dates():
    co = ft.challenger_origins(Q)
    assert len(co) == 27 and (co >= ft.TEST_START).sum() == 11
    tu = ft.tuning_origins(Q)
    assert len(tu) == 8 and set(tu) <= set(co) and (tu <= ft.SELECTION_END).all()


def test_choose_ignores_uncoherent_methods():
    s = pd.DataFrame({"method": ["naive", "a+bottom_up", "b+wls_var"], "rmse_pp": [0.1, 2.0, 1.0]})
    chosen, avg = ft.choose(s)
    assert chosen == "b+wls_var" and "naive" not in avg.index


def _table(chosen_err, ref_err):
    return pd.DataFrame({"m": chosen_err, "momentum": ref_err}, index=Q[-11:])


def test_verdict_needs_both_lower_rmse_and_seven_wins():
    assert ft.verdict(_table([1.0] * 11, [2.0] * 11), "m")["passes"]
    lower_but_few_wins = ft.verdict(_table([0.1] * 3 + [2.1] * 8, [2.0] * 11), "m")
    assert lower_but_few_wins["rule_rmse_lower"] and not lower_but_few_wins["passes"]
    many_wins_but_higher = ft.verdict(_table([1.9] * 8 + [9.0] * 3, [2.0] * 11), "m")
    assert many_wins_but_higher["wins"] == 8 and not many_wins_but_higher["passes"]


def test_rmse_table_in_percentage_points():
    r = pd.DataFrame({"method": ["a", "a"], "level": ["Metro"] * 2, "h": [12, 12], "pred": [0.01, -0.01],
                      "actual": [0.0, 0.0]})
    assert np.isclose(ft.rmse_table(r).iloc[0], 1.0)


def test_level_of_id():
    assert list(ft.level_of_id(pd.Series(["US", "ST:WA", "MSA:Seattle, WA", "CTY:53033"]))) == [
        "National", "State", "Metro", "County"]


def test_score_forward_waits_for_outcomes():
    L = pd.DataFrame({"US": np.log([100.0, 101.0])}, index=pd.to_datetime(["2026-08-31", "2026-09-30"]))
    frozen = pd.DataFrame({"method": "naive", "h": [1, 12], "unique_id": "US", "pred_pct": 0.0,
                           "origin": "2026-09-30"})
    assert ft.score_forward(frozen, L).empty
    L.loc[pd.Timestamp("2026-10-31"), "US"] = np.log(102.01)
    got = ft.score_forward(frozen, L)
    assert list(got["h"]) == [1] and np.isclose(got["actual_pct"].iloc[0], 100 * np.log(1.01))
