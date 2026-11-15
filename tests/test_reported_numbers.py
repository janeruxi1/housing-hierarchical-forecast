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
        for name, path in [("readme", "README.md"), ("memo", "reports/decision_memo.md"),
                           ("card", "MODEL_CARD.md"), ("nb03", "notebooks/03_models_reconciliation.py"),
                           ("nb04", "notebooks/04_scenarios.py"), ("phases", "docs/PHASES.md")]}


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
    ],
    "nb03": [rs_cut.replace(" to ", " pp to ", 1), f"{metro_skill} lower", f"about {cov80} of metro"],
    "nb04": [shock_range.replace(" to ", "-"), by_metro("Las Vegas, NV"), by_metro("Phoenix, AZ"), by_metro("Pittsburgh, PA"),
             f"correlation {zf['correlation']:.2f}"],
}


@pytest.mark.parametrize("doc,claim", [(d, s) for d, claims in CLAIMS.items() for s in claims])
def test_document_quotes_recorded_number(doc, claim):
    assert claim in DOCS[doc], f"{doc}: expected to find {claim!r}"


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
