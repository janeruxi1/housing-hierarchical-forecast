"""Every headline number in the README, decision memo, model card and notebook answers matches
reports/metrics, which the notebooks write on the real data.

Each check formats a recorded value the way the documents quote it and looks for that text, so
a document that drifts from the results (or results that change without the documents) fails CI.
Minus signs are compared as plain hyphens.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "metrics"
def norm(text: str) -> str:
    """Hyphen for minus, single spaces for any run of whitespace (line wraps do not matter)."""
    return " ".join(text.replace("\u2212", "-").split())


DOCS = {name: norm((ROOT / path).read_text(encoding="utf-8"))
        for name, path in [("readme", "README.md"), ("results", "docs/results.md"), ("memo", "reports/decision_memo.md"),
                           ("card", "MODEL_CARD.md"), ("nb03", "notebooks/03_models_reconciliation.py"),
                           ("nb04", "notebooks/04_scenarios.py"), ("phases", "docs/PHASES.md"),
                           ("nb05", "notebooks/05_final_test.py"), ("nb06", "notebooks/06_raw_index.py")]}


def j(name):
    return json.loads((METRICS / f"{name}.json").read_text())


def c(name):
    return pd.read_csv(METRICS / f"{name}.csv")


def f1(x):
    return f"{x:.1f}"


audit = j("data_audit")
final = c("03_final_vs_momentum").set_index(["level", "h"])
chosen = j("03_selection")["chosen"]
regime = c("03_final_by_regime").query("h == 12").pivot_table(index="regime", columns="method", values="rmse_pp")
scores = c("03_scores_all").set_index(["method", "level", "h"])
cov = c("03_interval_coverage").set_index(["level", "h", "nominal"])
coh = c("01_coherence_summary").set_index("group")
lead02 = c("02_leaderboard").set_index(["method", "level", "h"])
reg02 = c("02_metro_by_regime").query("h == 12").set_index(["method", "regime"])
scen = c("04_scenarios").set_index(["scenario", "coefficients"])
zf = j("04_vs_zillow_forecast")
expo = c("04_metro_exposure").set_index("metro")      # sorted: most exposed first
falls = j("04_stress_falls")
zc = c("03_zillow_comparison").set_index("method")

# Notebook 05: the final test (choose on 2014-2021, score once on 2023-2025) and the challengers
sel05, v05 = j("05_selection"), j("05_verdict")
all05 = c("05_all_methods_metro12").set_index("method")
ch05 = c("05_challengers").query("period == 'test' and level == 'Metro' and h == 12").iloc[0]
wins05 = c("05_challenger_wins").set_index("method")["test_dates_better_than_momentum"]
prophet_pick = c("05_prophet_tuning").iloc[0]["method"]
T_CH, T_MOM = f1(v05["rmse_chosen_pp"]), f1(v05["rmse_reference_pp"])
T_NAIVE, T_COMBO = f1(all05.loc["naive", "test"]), f1(all05.loc["combo+bottom_up", "test"])
T_COMBO_WINS = int(all05.loc["combo+bottom_up", "test dates better than momentum"])
ETS, ARIMA, PROPHET = f1(ch05["autoets"]), f1(ch05["autoarima"]), f1(ch05[prophet_pick])
CH_ROWS = [f"| {label} | {f1(ch05[m])} pp | {'-' if m == 'momentum' else int(wins05[m])} |" for label, m in [
    ("AutoETS (damped trend for almost every series)", "autoets"), ("No change", "naive"), ("Momentum", "momentum"),
    ("combo+bottom_up (this project)", "combo+bottom_up"), ("AutoARIMA", "autoarima"),
    ("Prophet, tuned on 2014-2021", prophet_pick), ("Prophet, default settings", "prophet")]]

# Notebook 05: how sure (date-by-date Diebold-Mariano and block-bootstrap intervals, metros)
sig = c("05_significance").set_index(["period", "level", "h"])
s1, s3, s12 = (sig.loc[("all 47 dates", "Metro", h)] for h in (1, 3, 12))
t1, t3 = (sig.loc[("test, 2023-25", "Metro", h)] for h in (1, 3))


def pc(x):
    return f"{round(100 * x)}"


SIG_SHORT = (f"{pc(s1['skill'])}% below momentum at 1 month and {pc(s3['skill'])}% at 3 months (95% intervals "
             f"{pc(s1['skill_lo95'])} to {pc(s1['skill_hi95'])}% and {pc(s3['skill_lo95'])} to {pc(s3['skill_hi95'])}%)")
S12 = f"{100 * s12['skill_lo95']:+.0f}% to {100 * s12['skill_hi95']:+.0f}%"
SIG_LONG = (f"{pc(s1['skill'])}% below momentum at 1 month (95% interval {pc(s1['skill_lo95'])} to "
            f"{pc(s1['skill_hi95'])}%), {pc(s3['skill'])}% at 3 months ({pc(s3['skill_lo95'])} to {pc(s3['skill_hi95'])}%), "
            f"{pc(s12['skill'])}% at 12 months ({S12}")

# Notebook 06: the raw index (metros, against the best simple rule, all 47 dates)
raw_sig = c("06_raw_significance").query("period == 'all 47 dates' and level == 'Metro'").set_index("h")
r1, r3, r12 = raw_sig.loc[1], raw_sig.loc[3], raw_sig.loc[12]
raw_audit, swing = j("06_raw_audit"), j("06_raw_vs_smoothed")["seasonal swing, raw (pp, highest minus lowest month)"]
R1, R1LO, R1HI, R3, R12 = pc(r1["skill"]), pc(r1["skill_lo95"]), pc(r1["skill_hi95"]), pc(r3["skill"]), pc(r12["skill"])
S3 = f"{100 * r3['skill_lo95']:+.0f}% to {100 * r3['skill_hi95']:+.0f}%"

LEVELS = [("National", "Nation"), ("State", "States"), ("Metro", "Metros"), ("County", "Counties")]
ACCURACY_ROWS = [f"| {label} | {f1(final.loc[(lv, 12), chosen])} pp | {f1(final.loc[(lv, 12), 'momentum'])} pp | "
                 f"{f1(final.loc[(lv, 12), 'naive'])} pp |" for lv, label in LEVELS]

rs_mom, rs_ch = regime.loc["Rate shock", "momentum"], regime.loc["Rate shock", chosen]
boom_mom, boom_ch = regime.loc["Pandemic boom", "momentum"], regime.loc["Pandemic boom", chosen]
rs_cut = f"{f1(rs_mom)} to {f1(rs_ch)} pp, {round(100 * (1 - rs_ch / rs_mom))}% lower"
metro_skill = f"{round(100 * final.loc[('Metro', 12), 'skill_vs_momentum'])}%"
cov80 = f"{round(100 * cov.loc[('Metro', 12, 0.8), 'coverage'])}%"
cov95 = f"{round(100 * cov.loc[('Metro', 12, 0.95), 'coverage'])}%"
width95 = f"{f1(cov.loc[('Metro', 12, 0.95), 'median_width_pp'])} pp"

r1_full = scen.loc[("Rates +1 pt", "full sample")]
r1_ex = scen.loc[("Rates +1 pt", "pandemic left out")]
r2_full = scen.loc[("Rates +2 pt (stress)", "full sample")]
r2_ex = scen.loc[("Rates +2 pt (stress)", "pandemic left out")]
shock_range = f"{f1(-r1_ex['vs baseline (pp)'])} to {f1(-r1_full['vs baseline (pp)'])} pp"
shock2_range = f"{f1(-r2_ex['vs baseline (pp)'])} to {f1(-r2_full['vs baseline (pp)'])} pp"
dollars = (f"$0.{round(-r1_ex['value change vs baseline ($bn)'] / 10):02d} to "
           f"${-r1_full['value change vs baseline ($bn)'] / 1000:.1f} trillion")

top10 = expo.head(10)
bottom5 = expo.tail(5)
hit = expo["rates +1: change (pp)"]


def by_metro(name):
    return f"{name.split(',')[0]} {f1(hit[name])} pp"


CLAIMS = {
    "readme": [
        *ACCURACY_ROWS,
        f"{audit['counties_full_history']:,} counties", f"{round(100 * audit['share_of_us_housing_units_kept'])}% of US homes",
        f"{audit['nodes_by_level']['State']} states and {audit['nodes_by_level']['Metro']} metro groups",
        f"RMSE {coh.loc['Metro (2+ counties)', 'median_rmse_12m_pp']:.2f} pp",
        f"RMSE {f1(lead02.loc[('momentum', 'Metro', 12), 'rmse_pp'])} pp vs {f1(lead02.loc[('naive', 'Metro', 12), 'rmse_pp'])} pp",
        f"{f1(reg02.loc[('momentum', 'Rate shock'), 'rmse_pp'])} pp in the 2022-23 rate shock",
        rs_cut, f"({f1(boom_ch)} vs {f1(boom_mom)} pp)", f"{metro_skill} below momentum",
        f"cover {cov80} of metro", shock_range, dollars, by_metro("Las Vegas, NV"), by_metro("Phoenix, AZ"),
        by_metro("Pittsburgh, PA"), f"({f1(zf['us_ours_pct'])}%)", f"Zillow's {f1(zf['us_zillow_pct'])}%",
        f"correlation {zf['correlation']:.2f} across {zf['metros_compared']} metros",
        f"median {f1(zf['median_gap_pp'])} pp higher",
        f"from {f1(r1_full['vs baseline (pp)'])} to {f1(r1_ex['vs baseline (pp)'])} pp",
        f"scores {T_CH} pp against momentum's {T_MOM} pp", f"no-change forecast ({T_NAIVE} pp)",
        f"lowest 12-month error ({ETS} pp, better than momentum on {wins05['autoets']} of 11 dates)",
        f"AutoARIMA scores {ARIMA} pp", f"Prophet {PROPHET} pp even after tuning",
        f"{T_CH} pp against momentum's {T_MOM} pp, better on only {v05['wins']} of 11 dates",
        f"ties momentum ({T_COMBO} pp, better on {T_COMBO_WINS} of 11)", *CH_ROWS,
        SIG_SHORT, f"interval runs from {S12}", f"{SIG_LONG}, Diebold-Mariano p = {s12['dm_p']:.2f})",
        f"gains hold ({pc(t1['skill'])}% and {pc(t3['skill'])}%)",
        f"error is still {R1}% below the best simple rule at 1 month (95% interval {R1LO} to {R1HI}%)",
        f"only {R3}% below a seasonal rule ({S3})",
        f"metro error is {R1}% below the best simple rule at 1 month (95% interval {R1LO} to {R1HI}%), against "
        f"{pc(s1['skill'])}% on the smoothed index", f"{R3}% below a seasonal-naive rule ({S3})",
        f"at 12 months it is {R12}%", f"swings about {swing:.1f} pp",
    ],
    "memo": [
        *ACCURACY_ROWS,
        f"{audit['counties_full_history']:,} counties holding {round(100 * audit['share_of_us_housing_units_kept'])}% of US homes",
        shock_range, dollars, shock2_range,
        f"{top10['beta'].min():.1f} to {top10['beta'].max():.1f} times",
        f"{f1(-top10['rates +1: change (pp)'].max())} to {f1(-top10['rates +1: change (pp)'].min())} pp off",
        f"{f1(-bottom5['rates +1: change (pp)'].max())} to {f1(-bottom5['rates +1: change (pp)'].min())} pp in",
        f"expects {f1(zf['us_ours_pct'])}% growth", f"Zillow's {f1(zf['us_zillow_pct'])}%",
        f"Across {zf['metros_compared']} metros", f"(correlation {zf['correlation']:.2f})",
        f"median {f1(zf['median_gap_pp'])} pp higher",
        f"{falls['metros_of_50_falling']} of the {falls['metros_with_zillow_forecast']} largest metros",
        *[f"{m.split(',')[0]} ({f1(v)}%)" for m, v in list(falls["falling"].items())[:3]],
        f"covered only {cov80}", f"({cov95} coverage; a median width of {width95}",
        f"was {f1(rs_mom)} pp for momentum", f"{f1(regime.loc['Rate shock', 'naive'])} pp for \"no change\"",
        f"against {f1(rs_ch)} pp here", f"({f1(boom_ch)} vs {f1(boom_mom)} pp)",
        f"misses by {f1(scores.loc[(chosen, 'National', 12), 'rmse_pp'])} pp, against "
        f"{f1(scores.loc[('combo', 'National', 12), 'rmse_pp'])} pp",
        f"({T_NAIVE} pp for metros)", f"only ties momentum ({T_COMBO} pp)",
        f"interval runs from {S12}", f"({pc(s1['skill'])}% and {pc(s3['skill'])}% below momentum for metros)",
        f"only the 1-month gain survives ({R1}% below the best simple rule)",
    ],
    "card": [
        *ACCURACY_ROWS,
        f"{audit['nodes_by_level']['State']} states, {audit['nodes_by_level']['Metro']} metro",
        f"{audit['counties_full_history']:,} counties", f"{round(100 * audit['share_of_us_housing_units_kept'])}% of US housing units",
        f"from {scores.index.get_level_values('method').nunique()} candidates",
        f"rate shock: {f1(rs_ch)} pp, against {f1(rs_mom)} pp", f"boom: {f1(boom_ch)} pp, against {f1(boom_mom)} pp",
        f"cover {cov80} of metro outcomes", f"95% ranges cover {cov95}",
        f"{zc.loc[chosen, '1']:.4f}", f"{zc.loc[chosen, '6']:.4f}",
        f"{zc.loc['Zillow (published)', '1']:.4f}", f"{zc.loc['Zillow (published)', '6']:.4f}",
        f"expects {f1(zf['us_ours_pct'])}% national growth", f"Zillow's {f1(zf['us_zillow_pct'])}%",
        f"(correlation {zf['correlation']:.2f})", f"median {f1(zf['median_gap_pp'])} pp higher",
        f"from {f1(r1_full['vs baseline (pp)'])} to {f1(r1_ex['vs baseline (pp)'])} pp",
        f"scores {T_CH} pp at metro 12 months against momentum's {T_MOM} pp", f"\"no change\" ({T_NAIVE} pp)",
        f"ties momentum ({T_COMBO} pp)", f"AutoETS {ETS} pp", f"AutoARIMA {ARIMA} pp", f"Prophet {PROPHET} pp",
        f"{SIG_LONG}, not distinguishable from zero)",
        f"metro error {R1}% below it at 1 month (95% interval {R1LO} to {R1HI}%), {R3}% at 3 months ({S3}), "
        f"{R12}% at 12 months",
    ],
    "phases": [
        f"{audit['counties_full_history']:,} counties", f"{round(100 * audit['share_of_us_housing_units_kept'])}%",
        f"{audit['counties_with_interpolation']} counties had gaps", f"{audit['months_interpolated']:,} months in all",
        f"median 12-month growth RMSE of {coh.loc['Metro (2+ counties)', 'median_rmse_12m_pp']:.2f} pp",
        f"States track it at {coh.loc['State', 'median_rmse_12m_pp']:.2f} pp",
        f"{f1(lead02.loc[('momentum', 'Metro', 12), 'rmse_pp'])} pp against {f1(lead02.loc[('naive', 'Metro', 12), 'rmse_pp'])} pp",
        f"{f1(reg02.loc[('momentum', 'Rate shock'), 'rmse_pp'])} pp in the 2022-23 rate shock",
        f"{f1(final.loc[('Metro', 12), chosen])} pp, {metro_skill} below momentum",
        f"falls from {f1(rs_mom)} to {f1(rs_ch)} pp", f"({f1(boom_ch)} vs {f1(boom_mom)} pp)",
        f"misses by {f1(scores.loc[(chosen, 'National', 12), 'rmse_pp'])} pp, against "
        f"{f1(scores.loc[('combo', 'National', 12), 'rmse_pp'])} pp",
        f"cover {cov80} of metro outcomes", f"95% ranges cover {cov95}",
        f"({f1(zf['us_ours_pct'])}%), against Zillow's own {f1(zf['us_zillow_pct'])}%",
        f"Across {zf['metros_compared']} metros", f"(correlation {zf['correlation']:.2f})",
        f"median {f1(zf['median_gap_pp'])} pp higher", shock_range, dollars, f"Las Vegas {f1(hit['Las Vegas, NV'])} pp",
        f"{falls['metros_of_50_falling']} of 50 metros falling",
        f"fails: {T_CH} pp against momentum's {T_MOM} pp, better on {v05['wins']} of 11 dates",
        f"\"no change\" is best ({T_NAIVE} pp)",
        f"({ETS} pp on the 50 largest metros, better than momentum on {wins05['autoets']} of 11)",
        f"even after tuning ({PROPHET} pp)",
        f"clear at 1 month ({pc(s1['skill'])}%, 95% interval {pc(s1['skill_lo95'])} to {pc(s1['skill_hi95'])}%) and 3 "
        f"months ({pc(s3['skill'])}%, {pc(s3['skill_lo95'])} to {pc(s3['skill_hi95'])}%), but not at 12 months "
        f"({pc(s12['skill'])}%, {S12})",
        f"All {raw_audit['counties_raw']:,} counties have a full raw history", f"swings about {swing:.1f} pp",
        f"metro error {R1}% below the best simple rule (95% interval {R1LO} to {R1HI}%), against {pc(s1['skill'])}% "
        "on the smoothed index", f"{R3}% below seasonal naive ({S3}). At 12 months, {R12}%",
    ],
    "nb05": [f"{T_CH} pp metro 12-month error against {T_MOM} pp for momentum, better on only {v05['wins']} of "
             f"{v05['test_dates']}", f"no-change forecast ({T_NAIVE} pp)", f"({T_COMBO} pp, {T_COMBO_WINS} of 11 dates)",
             f"({ETS} pp on the 50 largest metros, better than momentum on {wins05['autoets']} of 11 dates)",
             f"tuning on 2014-2021 ({PROPHET} pp)", f"(95% interval {S12})",
             f"({pc(s1['skill'])}% and {pc(s3['skill'])}% for metros)"],
    "nb06": [f"metro error {R1}% below the best simple rule (95% interval {R1LO}% to {R1HI}%), down from "
             f"{pc(s1['skill'])}% on the smoothed index", f"{R3}% below seasonal naive, with an interval from {S3}",
             f"so most of the smoothed index's {pc(s3['skill'])}% came from the smoothing", f"small and uncertain ({R12}%)",
             f"about {swing:.1f} pp of monthly growth"],
    "nb03": [rs_cut.replace(" to ", " pp to ", 1), f"{metro_skill} lower", f"about {cov80} of metro"],
    "nb04": [shock_range.replace(" to ", "-"), by_metro("Las Vegas, NV"), by_metro("Phoenix, AZ"), by_metro("Pittsburgh, PA"),
             f"correlation {zf['correlation']:.2f}"],
}


# The README's detail lives in docs/results.md: README claims may appear in either.
DOCS["readme"] = DOCS["readme"] + " " + DOCS.pop("results")

README_HEADLINES = [
    f"{metro_skill} below momentum", rs_cut, shock_range, dollars, by_metro("Las Vegas, NV"),
    by_metro("Pittsburgh, PA"), f"expects {f1(zf['us_ours_pct'])}% national growth against Zillow's {f1(zf['us_zillow_pct'])}%",
    f"(correlation {zf['correlation']:.2f} across {zf['metros_compared']} metros)",
    f"{falls['metros_of_50_falling']} of the 50 largest metros lose value", f"covered only {cov80} of metro outcomes",
    f"from {f1(r1_full['vs baseline (pp)'])} to {f1(r1_ex['vs baseline (pp)'])} pp",
]


@pytest.mark.parametrize("doc,claim", [(d, s) for d, claims in CLAIMS.items() for s in claims])
def test_document_quotes_recorded_number(doc, claim):
    assert claim in DOCS[doc], f"{doc}: expected to find {claim!r}"


@pytest.mark.parametrize("claim", README_HEADLINES)
def test_readme_headline_quotes_recorded_number(claim):
    """The front page itself, not only docs/results.md, carries the headline numbers."""
    assert claim in norm((ROOT / "README.md").read_text(encoding="utf-8")), claim


def test_memo_lists_the_ten_most_exposed_metros_in_order():
    names = [m.split(",")[0] for m in top10.index]
    listed = ", ".join(names[:-1]) + " and " + names[-1]
    assert listed in DOCS["memo"]


def test_qualitative_claims_hold():
    ratio = hit["Las Vegas, NV"] / hit["Pittsburgh, PA"]
    assert 3.5 <= ratio <= 4.5                                   # "about four times"
    neg = (top10["Zillow 12m (%)"] < 0).sum()
    assert "Seven of the ten" in DOCS["memo"] and neg == 7
    assert chosen == "combo+bottom_up"
    # "at metro and county level, bottom-up is as accurate as the best MinTrace variant" (within 1%)
    for lv in ("Metro", "County"):
        b = scores.loc[("combo+bottom_up", lv, 12), "rmse_pp"]
        best = min(scores.loc[(f"combo+{r}", lv, 12), "rmse_pp"] for r in ("wls_var", "mint_shrink", "wls_struct"))
        assert b <= best * 1.01, lv


def test_final_test_verdict_matches_the_documents():
    """The documents say the final test fails and name the method picked on 2014-2021."""
    assert sel05["chosen"] == "zillow_style+mint_shrink" and not v05["passes"]
    assert v05["test_dates"] == 11 and sel05["selection_dates"] == 32
    for doc in ("readme", "card", "phases", "nb05"):
        assert "zillow_style+mint_shrink" in DOCS[doc] or doc == "readme", doc
        assert "fail" in DOCS[doc], doc
    assert all05["test"].idxmin() == "naive"                       # "no candidate beats no change"
    assert ch05[["autoets", "autoarima", prophet_pick, "naive", "momentum", "combo+bottom_up"]].idxmin() == "autoets"
    assert prophet_pick == "prophet_cr95_flex" and ch05[prophet_pick] < ch05["prophet"]


def test_forward_forecasts_are_frozen():
    """Section 5 of notebook 05 saves today's forecasts once: every method, series and horizon."""
    f = pd.read_csv(METRICS / "05_forward_forecasts_2026-09.csv")
    assert set(f["method"]) == {"combo+bottom_up", "momentum", "naive", "autoets", "autoarima", "prophet_cr95_flex"}
    assert f.groupby("method").size().eq(400).all() and set(f["h"]) == {1, 3, 6, 12}
    assert (f["origin"] == "2026-09-30").all()



def test_significance_claims_hold():
    """'Clear at 1-3 months, could be chance at 12' holds in the recorded tests."""
    for r in (s1, s3, t1, t3):
        assert r["skill_lo95"] > 0 and r["dm_p"] < 0.05
    assert s12["skill_lo95"] < 0 < s12["skill_hi95"] and s12["dm_p"] > 0.05


def test_raw_index_claims_hold():
    """'Only the 1-month edge survives on the raw index', on the same counties."""
    assert raw_audit["counties_raw"] == raw_audit["counties_smoothed"] == audit["counties_full_history"]
    assert r1["skill_lo95"] > 0 and r1["dm_p"] < 0.01
    assert r3["skill_lo95"] < 0 < r3["skill_hi95"] and r3["rule"] == "seasonal_naive"
    assert r12["skill_lo95"] < 0 < r12["skill_hi95"]
