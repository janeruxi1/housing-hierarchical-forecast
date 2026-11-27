"""Where are US home values heading, and what would a rate shock do? Interactive companion app.

Tabs:
    1. The answer          -- the headline, and how the chosen forecast did in the backtest
    2. Explore a place     -- any state, metro or county: its forecast, its history and its past misses
    3. Rate shock          -- move mortgage rates and unemployment; see the nation and the 50 largest metros
    4. How it works        -- the hierarchy, the models, the reconciliation and the limits

Reads only the derived extract in app/data (built by scripts/build_app_data.py from the real
panel; no Zillow index levels for single counties) and the results in reports/metrics.
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "app" / "data"
METRICS = ROOT / "reports" / "metrics"
REPO_URL = "https://github.com/janeruxi1/housing-hierarchical-forecast"
MEMO_URL = f"{REPO_URL}/blob/main/reports/decision_memo.md"
BLUE, RED, GREY, INK = "#1F5AA6", "#D1495B", "#8C8C8C", "#111827"
METHOD_LABEL = {"combo+bottom_up": "Chosen forecast", "momentum": "Momentum", "naive": "No change"}
VERSIONS = {"full sample": "As backtested (full sample)", "pandemic left out": "Refitted without 2020-21"}
LEVELS = {"Metro": "Metro areas", "State": "States", "County": "Counties"}

st.set_page_config(page_title="US Home Value Forecast", page_icon="🏠", layout="wide")


def pct(x: float, digits: int = 1) -> str:
    return f"{x:+.{digits}f}%"


def pp(x: float, digits: int = 1) -> str:
    return f"{x:+.{digits}f} pp"


def usd_bn(x: float) -> str:
    sign = "-" if x < 0 else "+"
    return f"{sign}${abs(x) / 1e3:.2f} tn" if abs(x) >= 1000 else f"{sign}${abs(x):,.0f} bn"


def log_pp(g: float, g0: float) -> float:
    """Change between two growth rates (%), in log points x 100, as the notebooks report it."""
    return 100 * np.log((1 + g / 100) / (1 + g0 / 100))


@st.cache_data(show_spinner=False)
def load() -> dict:
    meta = json.loads((DATA / "meta.json").read_text())
    places = pd.read_parquet(DATA / "places.parquet")
    fc = pd.read_parquet(DATA / "forecast.parquet")
    grid = pd.read_parquet(DATA / "scenarios.parquet")
    grid["version"] = grid["version"].astype(str)
    grid[["rate", "unemp"]] = grid[["rate", "unemp"]].astype(float).round(2)
    bt = pd.read_parquet(DATA / "backtest.parquet")
    bt["method"] = bt["method"].astype(str)
    bt["unique_id"] = bt["unique_id"].astype(str)
    return {"meta": meta, "places": places.set_index("unique_id"),
            "g12": fc[fc["h"] == 12].set_index("unique_id")["growth_pct"].astype(float),
            "path": fc, "grid": grid.set_index(["version", "rate", "unemp", "unique_id"])["growth_pct"],
            "history": pd.read_parquet(DATA / "history.parquet"), "backtest": bt,
            "bands": pd.read_parquet(DATA / "bands.parquet").set_index(["level", "h"])}


def metrics_json(name: str) -> dict:
    return json.loads((METRICS / f"{name}.json").read_text())


def metrics_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(METRICS / f"{name}.csv")


def scenario(version: str, rate: float, unemp: float) -> pd.Series:
    return D["grid"].loc[(version, round(rate, 2), round(unemp, 2))].astype(float)


def band_table(level: str) -> pd.DataFrame:
    """Past-error quantiles by month ahead (0-12), interpolated between the scored horizons (1, 3, 6, 12)."""
    q = D["bands"].loc[level].reindex(range(13))
    q.loc[0] = 0.0
    return q.interpolate()


def band(level: str, h: int, g: float, lo: str, hi: str) -> tuple[float, float]:
    """Range around a forecast of g% from the chosen method's past errors (pred - actual) at this level."""
    q = band_table(level).loc[h]
    lg = np.log1p(g / 100)
    return 100 * np.expm1(lg - q[hi]), 100 * np.expm1(lg - q[lo])


D = load()
P = D["places"]
origin = pd.Timestamp(D["meta"]["origin"])
target = origin + pd.offsets.MonthEnd(12)
vs_z = metrics_json("04_vs_zillow_forecast")
zil_metros = metrics_csv("04_vs_zillow_forecast_metros").set_index("unique_id")["growth_pct"]
final = metrics_csv("03_final_vs_momentum")
exposure = metrics_csv("04_metro_exposure")
stress = metrics_json("04_stress_falls")
us0 = float(D["g12"]["US"])
rate1 = {v: log_pp(float(scenario(v, 1.0, 0.0)["US"]), us0) for v in VERSIONS}
metro12 = final[(final["level"] == "Metro") & (final["h"] == 12)].iloc[0]
sig = metrics_csv("05_significance").set_index(["period", "level", "h"])
sig3, sig12 = sig.loc[("all 47 dates", "Metro", 3)], sig.loc[("all 47 dates", "Metro", 12)]
verdict = metrics_json("05_verdict")

# --- Header -------------------------------------------------------------------
st.title("🏠 Where Are US Home Values Heading, and What Would a Rate Shock Do?")
st.markdown(
    f"A 12-month forecast of home values from **{origin:%B %Y}** for the nation, 49 states, "
    f"{D['meta']['places']['Metro']} metro groups and {D['meta']['counties']:,} counties (84% of US homes), "
    "built from the counties up so every level adds up. Tested on 47 past forecast dates from 2014 to 2025, "
    "and used to stress-test the effect of a rise in mortgage rates."
)
st.markdown(f"[Code and notebooks]({REPO_URL}) · [Decision memo]({MEMO_URL}) · "
            f"[Model card]({REPO_URL}/blob/main/MODEL_CARD.md)")
with st.expander("ℹ️ How to use this demo"):
    st.markdown(
        """
1. **🏁 The answer**: the forecast, and how it did against simple rules of thumb on past dates.
2. **🔎 Explore a place**: pick any state, metro or county and see its forecast, its history, how
   much a rate rise would hurt it, and how the method's past forecasts for it turned out.
3. **📉 Rate shock**: move mortgage rates and unemployment and watch the nation and the 50 largest
   metros respond. This is the stress test the memo is built on.
4. **🧭 How it works**: how the counties add up, which models are combined, and what not to use it for.

**Growth** = change in the typical home value over the next 12 months. **pp** = percentage points.
**Momentum** = "the last 12 months' growth continues", the rule of thumb to beat.
**Beta** = how hard a place swings with the national housing cycle (1 = like the nation).
"""
    )

tab_answer, tab_place, tab_shock, tab_how = st.tabs(
    ["🏁 The answer", "🔎 Explore a place", "📉 Rate shock", "🧭 How it works"])

# =====================================================================
# TAB 1 -- THE ANSWER
# =====================================================================
with tab_answer:
    st.success(
        f"**The forecast is most useful for ranking places and sizing rate risk, not for the level.** It expects "
        f"**{pct(us0)}** growth for the nation over 12 months, above Zillow's own **{pct(vs_z['us_zillow_pct'])}**, "
        f"but the two rank metros alike (correlation {vs_z['correlation']:.2f}). A 1-point rise in mortgage rates "
        f"takes **{abs(rate1['pandemic left out']):.1f} to {abs(rate1['full sample']):.1f} pp** off national growth, "
        "and up to four times as much in high-beta Sun Belt metros."
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("US growth, next 12 months", pct(us0), border=True,
              help=f"Typical home value, {origin:%b %Y} to {target:%b %Y}, for the panel's 1,371 counties.")
    k2.metric("Zillow's forecast, same month", pct(vs_z["us_zillow_pct"]), border=True,
              help=f"Zillow's published forecast (ZHVF). Across {vs_z['metros_compared']} metros this model sits a "
                   f"median {vs_z['median_gap_pp']:.1f} pp higher, so the memo budgets on Zillow's level.")
    k3.metric("Metro error vs momentum, 3 months ahead", f"{-sig3['skill']:.0%}", border=True,
              help=f"Over 47 forecast dates; 95% interval {-sig3['skill_hi95']:.0%} to {-sig3['skill_lo95']:.0%}. "
                   f"At 12 months the gain is {metro12['skill_vs_momentum']:.0%} ({metro12['combo+bottom_up']:.1f} pp "
                   f"against {metro12['momentum']:.1f} pp), but its interval ({sig12['skill_lo95']:+.0%} to "
                   f"{sig12['skill_hi95']:+.0%}) includes zero, and on 2023-25 alone it only ties momentum.")
    k4.metric("Large metros losing value under stress", f"{stress['metros_of_50_falling']} of 50", border=True,
              help="Zillow's own 12-month forecast plus this model's +2-point rate shock, for the 50 largest metros.")

    left, right = st.columns(2)
    with left:
        lvl = st.radio("Forecast error by months ahead (lower is better), at the level of",
                       ["National", "State", "Metro", "County"], index=2, horizontal=True, key="acc_level")
        f = final[final["level"] == lvl].melt(id_vars="h", value_vars=list(METHOD_LABEL),
                                              var_name="method", value_name="rmse")
        f["method"] = f["method"].map(METHOD_LABEL)
        chart = alt.Chart(f).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("h:O", title="Months ahead"),
            y=alt.Y("rmse:Q", title="Error in growth (RMSE, pp)"),
            color=alt.Color("method:N", title=None, legend=alt.Legend(orient="bottom"),
                            scale=alt.Scale(domain=list(METHOD_LABEL.values()), range=[BLUE, GREY, "#CBD5E1"])),
            tooltip=["method", "h", alt.Tooltip("rmse:Q", format=".2f")])
        st.altair_chart(chart.properties(height=330), width="stretch")
    with right:
        st.markdown("**Where it earns its keep: metros, 12 months ahead, by period**")
        st.caption("Momentum = the last 12 months' growth continues; no change = values stay put.")
        r = metrics_csv("03_final_by_regime")
        r = r[(r["h"] == 12) & r["method"].isin(METHOD_LABEL)].copy()
        r["method"] = r["method"].map(METHOD_LABEL)
        order = ["Recovery 2012-20", "Pandemic boom", "Rate shock", "Flat 2024+"]
        bars = alt.Chart(r).mark_bar().encode(
            x=alt.X("regime:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("method:N", sort=list(METHOD_LABEL.values())),
            y=alt.Y("rmse_pp:Q", title="Error in growth (RMSE, pp)"),
            color=alt.Color("method:N", title=None, legend=alt.Legend(orient="bottom"),
                            scale=alt.Scale(domain=list(METHOD_LABEL.values()), range=[BLUE, GREY, "#CBD5E1"])),
            tooltip=["regime", "method", alt.Tooltip("rmse_pp:Q", format=".1f", title="RMSE (pp)"), "n"])
        st.altair_chart(bars.properties(height=370), width="stretch")
    rs = r.set_index(["method", "regime"])["rmse_pp"]
    st.caption(
        f"The gain is in the 2022-23 rate shock: {rs[(METHOD_LABEL['combo+bottom_up'], 'Rate shock')]:.1f} pp against "
        f"{rs[(METHOD_LABEL['momentum'], 'Rate shock')]:.1f} pp for momentum, because the panel model reads mortgage "
        f"rates. In the 2020-22 boom momentum was slightly better ({rs[(METHOD_LABEL['momentum'], 'Pandemic boom')]:.1f} "
        f"vs {rs[(METHOD_LABEL['combo+bottom_up'], 'Pandemic boom')]:.1f} pp): no model saw that coming. "
        "Each period holds only a few forecast dates, so read these as indicative: compared date by date, the "
        "12-month gain over momentum is not distinguishable from chance."
    )
    st.info(f"**For a risk committee:** budget on the more conservative level (Zillow's), and use this model for the "
            f"ranking and the rate stress, which is what it was built and backtested to do. "
            f"[Decision memo]({MEMO_URL})")

# =====================================================================
# TAB 2 -- EXPLORE A PLACE
# =====================================================================
with tab_place:
    c1, c2 = st.columns([1, 2])
    level = c1.radio("Level", list(LEVELS), format_func=LEVELS.get, horizontal=True, key="level")
    opts = P[P["level"] == level].sort_values("housing_units", ascending=False)
    labels = opts["name"].tolist()
    default = labels.index("Seattle, WA") if level == "Metro" and "Seattle, WA" in labels else 0
    name = c2.selectbox(f"{LEVELS[level][:-1] if level != 'County' else 'County'} (largest first)", labels,
                        index=default, key="place")
    uid = opts.index[labels.index(name)]
    row = P.loc[uid]
    g = float(D["g12"][uid])
    lo80, hi80 = band(level, 12, g, "q10", "q90")
    lo95, hi95 = band(level, 12, g, "q025", "q975")
    hit = {v: log_pp(float(scenario(v, 1.0, 0.0)[uid]), g) for v in VERSIONS}

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Forecast, next 12 months", pct(g), border=True,
              help=f"{origin:%b %Y} to {target:%b %Y}. Range of past errors: 80% {lo80:+.1f}% to {hi80:+.1f}%; "
                   f"95% {lo95:+.1f}% to {hi95:+.1f}%. In the backtest such ranges ran narrow at turning points.")
    if uid in zil_metros.index:
        m2.metric("Zillow's forecast", pct(float(zil_metros[uid])), border=True,
                  help="Zillow's published 12-month forecast for the same month.")
    else:
        m2.metric("Zillow's forecast", "not compared", border=True,
                  help="Compared only for metros where the panel holds at least 95% of the homes (Zillow publishes "
                       "metro and national forecasts only).")
    m3.metric("If mortgage rates rise 1 point", pp(hit["full sample"]), border=True,
              help=f"Change in 12-month growth. Refitted without 2020-21: {pp(hit['pandemic left out'])}.")
    if level == "State":
        m4.metric("Homes in the panel", f"{row['housing_units'] / 1e6:.2f}M", border=True)
    else:
        m4.metric("Beta (national cycle = 1)", f"{row['beta']:.2f}", border=True,
                  help="How hard this place's 12-month growth has moved with the nation's since 2009 "
                       "(shrunk towards 1 for short or noisy histories).")
    extra = [] if level == "County" else [f"{row['housing_units'] / 1e3:,.0f}k homes",
                                          f"typical value ${row['typical_value'] / 1e3:,.0f}k"]
    if level == "County":
        extra = [f"{row['housing_units'] / 1e3:,.0f}k homes", f"metro: {P.loc[row['metro_id'], 'name']}"]
    st.caption(" · ".join(extra))

    if level == "County":
        path = D["path"][D["path"]["unique_id"] == uid].copy()
        q = band_table("County")
        lg = np.log1p(path["growth_pct"].astype(float) / 100)
        path["lo"] = 100 * np.expm1(lg - path["h"].map(q["q90"]))
        path["hi"] = 100 * np.expm1(lg - path["h"].map(q["q10"]))
        path["month"] = [origin + pd.offsets.MonthEnd(int(h)) if h else origin for h in path["h"]]
        area = alt.Chart(path).mark_area(opacity=0.2, color=BLUE).encode(
            x=alt.X("month:T", title=None, axis=alt.Axis(format="%b %Y", values=[
                origin + pd.offsets.MonthEnd(k) for k in range(0, 13, 2)])), y=alt.Y("lo:Q", title=f"Growth since {origin:%b %Y} (%)"), y2="hi:Q")
        line = alt.Chart(path).mark_line(color=BLUE, strokeWidth=2).encode(
            x="month:T", y="growth_pct:Q", tooltip=[alt.Tooltip("month:T", format="%b %Y"),
                                                    alt.Tooltip("growth_pct:Q", format="+.2f", title="growth (%)")])
        st.markdown("**The forecast path, with the 10th-90th percentile of the method's past county errors**")
        st.altair_chart((area + line).properties(height=280), width="stretch")
        st.caption("County histories are not shown: they are Zillow's own series, which this project does not "
                   "redistribute. States and metros are this project's own aggregates of its counties.")
    else:
        hist = D["history"][(D["history"]["unique_id"] == uid) & (D["history"]["ds"] >= "2009-01-01")]
        compare_us = st.toggle("Compare with the nation", value=uid != "US", key="compare_us",
                               disabled=uid == "US")
        lines = hist.assign(series=name)
        if compare_us and uid != "US":
            us = D["history"][(D["history"]["unique_id"] == "US") & (D["history"]["ds"] >= "2009-01-01")]
            lines = pd.concat([lines, us.assign(series="United States")])
        dom = [name] + (["United States"] if compare_us and uid != "US" else [])
        hist_chart = alt.Chart(lines).mark_line(strokeWidth=2).encode(
            x=alt.X("ds:T", title=None),
            y=alt.Y("growth_pct:Q", title="Growth over the previous 12 months (%)"),
            color=alt.Color("series:N", title=None, scale=alt.Scale(domain=dom, range=[INK, "#94A3B8"]),
                            legend=alt.Legend(orient="bottom")),
            tooltip=["series", alt.Tooltip("ds:T", format="%b %Y"), alt.Tooltip("growth_pct:Q", format="+.1f")])
        pts = pd.DataFrame({"ds": [target], "g": [g], "lo80": [lo80], "hi80": [hi80], "lo95": [lo95],
                            "hi95": [hi95], "who": ["This forecast"]})
        if uid in zil_metros.index:
            pts = pd.concat([pts, pd.DataFrame({"ds": [target + pd.Timedelta(days=45)], "g": [float(zil_metros[uid])],
                                                "who": ["Zillow's forecast"]})])
        fc_color = alt.Color("who:N", title=None, scale=alt.Scale(domain=["This forecast", "Zillow's forecast"],
                                                                  range=[BLUE, RED]), legend=alt.Legend(orient="bottom"))
        r95 = alt.Chart(pts.dropna(subset=["lo95"])).mark_rule(color=BLUE, opacity=0.35, strokeWidth=3).encode(
            x="ds:T", y="lo95:Q", y2="hi95:Q")
        r80 = alt.Chart(pts.dropna(subset=["lo80"])).mark_rule(color=BLUE, strokeWidth=6).encode(
            x="ds:T", y="lo80:Q", y2="hi80:Q")
        dots = alt.Chart(pts).mark_circle(size=110, opacity=1).encode(
            x="ds:T", y="g:Q", color=fc_color,
            tooltip=["who", alt.Tooltip("g:Q", format="+.1f", title="12-month growth (%)")])
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#CBD5E1").encode(y="y:Q")
        st.markdown(f"**{name}: the last 12 months' growth over time, and the forecast for {target:%B %Y}**")
        st.altair_chart((zero + hist_chart + r95 + r80 + dots).resolve_scale(color="independent")
                        .properties(height=320), width="stretch")
        st.caption("Thick bar: 80% range of the method's past errors at this level; thin bar: 95%. In the "
                   "backtest the 80% range covered only 73% of metro outcomes, so plan on the wider one.")

        st.markdown("**How the method's past 12-month forecasts for this place turned out**")
        b = D["backtest"]
        b = b[(b["unique_id"] == uid) & (b["h"] == 12) & b["method"].isin(["combo+bottom_up", "momentum"])].copy()
        b["target"] = b["origin"] + pd.offsets.MonthEnd(12)
        actual = b.drop_duplicates("origin")[["target", "actual"]].assign(series="What happened")
        actual["value"] = 100 * np.expm1(actual["actual"].astype(float))
        preds = b.assign(series=b["method"].map(METHOD_LABEL), value=100 * np.expm1(b["pred"].astype(float)))
        both = pd.concat([actual[["target", "series", "value"]], preds[["target", "series", "value"]]])
        dom2 = ["What happened", METHOD_LABEL["combo+bottom_up"], METHOD_LABEL["momentum"]]
        bt_chart = alt.Chart(both).mark_line(point=True).encode(
            x=alt.X("target:T", title="Month forecast for", axis=alt.Axis(format="%Y", tickCount="year")),
            y=alt.Y("value:Q", title="12-month growth (%)"),
            color=alt.Color("series:N", title=None, sort=dom2, scale=alt.Scale(domain=dom2, range=[INK, BLUE, GREY]),
                            legend=alt.Legend(orient="bottom")),
            strokeDash=alt.condition(alt.datum.series == "What happened", alt.value([1, 0]), alt.value([4, 3])),
            tooltip=["series", alt.Tooltip("target:T", format="%b %Y"), alt.Tooltip("value:Q", format="+.1f")])
        st.altair_chart(bt_chart.properties(height=280), width="stretch")
        err = b.assign(e=100 * (b["pred"].astype(float) - b["actual"].astype(float))).groupby("method")["e"]
        rmse = np.sqrt(err.apply(lambda x: (x ** 2).mean()))
        st.caption(f"Over {b['origin'].nunique()} forecast dates (one a quarter, 2014-2025), the error for "
                   f"{name} was **{rmse['combo+bottom_up']:.1f} pp** for the chosen forecast and "
                   f"**{rmse['momentum']:.1f} pp** for momentum.")

# =====================================================================
# TAB 3 -- RATE SHOCK
# =====================================================================
with tab_shock:
    st.markdown(
        "A stress test in the catastrophe-model sense: **shock × exposure = loss**. The **shock** is how the county "
        "panel regression's 12-month forecast moves when its mortgage-rate and unemployment inputs change. The "
        "**exposure** is each county's beta. Counties are then added up, so the nation, states and metros agree."
    )
    s1, s2, s3 = st.columns([1.2, 1.2, 1])
    rate = s1.select_slider("Sudden change in the 30-year mortgage rate (points)", D["meta"]["rate_grid"],
                            value=1.0, key="rate", format_func=lambda v: f"{v:+.1f}")
    unemp = s2.select_slider("Change in unemployment (points)", D["meta"]["unemp_grid"], value=0.0, key="unemp",
                             format_func=lambda v: f"{v:+.0f}")
    version = s3.radio("Model", list(VERSIONS), format_func=VERSIONS.get, key="version",
                       help="The full-sample model is the one backtested. Refitted without 2020-21, when unemployment "
                            "spiked while prices boomed, the rate effect is smaller. The memo quotes the range.")
    sc = scenario(version, rate, unemp)
    us_s = float(sc["US"])
    value_change = P.loc["US", "value_bn"] * (us_s - us0) / 100

    big = P[(P["level"] == "Metro") & ~P["name"].str.startswith("Non-metro")].nlargest(50, "housing_units").copy()
    big["baseline"] = D["g12"].reindex(big.index).to_numpy()
    big["scenario"] = sc.reindex(big.index).to_numpy()
    big["change"] = log_pp(big["scenario"], big["baseline"])
    big["zillow"] = big["name"].map(exposure.set_index("metro")["Zillow 12m (%)"])
    big["zillow_stressed"] = 100 * ((1 + big["zillow"] / 100) * np.exp(big["change"] / 100) - 1)
    falling = int((big["zillow_stressed"] < 0).sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("US growth under this scenario", pct(us_s), border=True,
              help=f"Baseline: {pct(us0)}.")
    k2.metric("Change vs baseline", pp(log_pp(us_s, us0)), border=True)
    k3.metric("Housing value vs baseline", usd_bn(value_change), border=True,
              help="Value of the panel's homes 12 months ahead, scenario minus baseline.")
    k4.metric("Large metros losing value", f"{falling} of 50", border=True,
              help="Conservative view: Zillow's own 12-month forecast as the starting level, plus this scenario's "
                   "change for the metro.")

    big = big.sort_values("change")
    bars = alt.Chart(big).mark_bar().encode(
        y=alt.Y("name:N", sort=big["name"].tolist(), title=None, axis=alt.Axis(labelLimit=200, labelOverlap=False)),
        x=alt.X("change:Q", title="Change in 12-month growth vs baseline (pp)"),
        color=alt.Color("beta:Q", title="Beta", scale=alt.Scale(scheme="blues")),
        tooltip=["name", alt.Tooltip("beta:Q", format=".2f"), alt.Tooltip("baseline:Q", format="+.1f", title="baseline (%)"),
                 alt.Tooltip("scenario:Q", format="+.1f", title="scenario (%)"),
                 alt.Tooltip("change:Q", format="+.2f", title="change (pp)"),
                 alt.Tooltip("zillow_stressed:Q", format="+.1f", title="Zillow level + shock (%)")])
    left, right = st.columns([1, 1.1])
    with left:
        st.markdown("**The 50 largest metros, most exposed first**")
        st.altair_chart(bars.properties(height=1000), width="stretch")
    with right:
        st.markdown("**Conservative stress view**: Zillow's level plus this shock")
        table = big.sort_values("zillow_stressed")[["name", "zillow_stressed", "zillow", "change", "beta"]]
        st.dataframe(table, hide_index=True, width="stretch", height=900, column_config={
            "name": "Metro",
            "zillow_stressed": st.column_config.NumberColumn("Zillow + shock (%)", format="%+.1f",
                                                             help="12-month growth under stress, conservative level"),
            "zillow": st.column_config.NumberColumn("Zillow (%)", format="%+.1f", help="Zillow's 12-month forecast"),
            "change": st.column_config.NumberColumn("Shock (pp)", format="%+.2f",
                                                    help="This model's change in 12-month growth"),
            "beta": st.column_config.NumberColumn("Beta", format="%.2f")})
        st.caption("This model's baseline sits above Zillow's, so the memo takes Zillow's level and this model's "
                   "rate sensitivity. With a +2-point shock, 45 of the 50 lose value.")
    if unemp > 0:
        st.warning("**Read rising unemployment with care.** Over 2010-2026 unemployment did little on its own once the "
                   "pandemic is set aside, and rate cuts went with rising prices. So a \"recession\" with lower rates "
                   "comes out positive. This model cannot represent a 2008-style downturn; the memo uses historical "
                   "drawdowns for that.")
    st.caption("Linear and immediate: no feedback from prices to rates, nothing on how long a shock lasts. The "
               "coefficients are what usually followed such moves in 2010-2026, not causal effects.")

# =====================================================================
# TAB 4 -- HOW IT WORKS
# =====================================================================
with tab_how:
    rec = metrics_csv("03_reconciliation")
    rec_us = rec[(rec["method"] == "combo+bottom_up") & (rec["level"] == "National") & (rec["h"] == 12)].iloc[0]
    st.markdown(
        f"""
**1. Make it add up.** Zillow publishes its index separately for the nation, states, metros and counties, and those
series do not add up to each other. This project rebuilds every level from the
{D['meta']['counties']:,} counties with a full history since 2008, weighting each by its Census housing units: the
value of a place's housing stock is the sum of its counties'. That is what lets a lender's state limits and metro
limits agree.

**2. Beat the rule of thumb.** Each place is forecast by two models: a per-place **AR(2)** model of monthly growth,
and a **panel regression** across all places that reads mortgage rates and unemployment. The forecast is their
average (*combo*). The bar to beat is **momentum** ("the last 12 months continue"), which is hard to beat because
the index is smoothed.

**3. Reconcile.** Forecasting every level separately gives numbers that do not add up. Four ways of making them
coherent were compared (bottom-up, two weighted least squares versions, MinT with shrinkage), each on five base
forecasts. **Bottom-up**, forecast
the counties and add them up, was best on average. It costs a little at the top: nationally, 12 months ahead,
{rec_us['rmse_pp']:.1f} pp against {rec_us['rmse_pp_base']:.1f} pp for forecasting the nation directly.

**4. Test it honestly.** A rolling backtest: forecast from 47 past dates (each quarter, 2014-2025), using only the
data available then, and score 1, 3, 6 and 12 months ahead at every level. Then a final test with its rules written
down first: choose the method on 2014-2021 only and score it once on 2023-2025. It
**{"passed" if verdict["passes"] else "failed"}** at 12 months (no method beat "no change" a year out), so the claims
rest on 1-3 months, ranking and the rate stress. Prophet, ETS and ARIMA ran as challengers (notebook 05).

**5. Stress it.** Change the panel model's rate and unemployment inputs, scale each county's response by its beta,
and add up.
"""
    )
    sel = metrics_json("03_selection")
    rank = pd.Series(sel["ranking"]).rename("Average error (RMSE, pp)").rename_axis("Model + reconciliation").reset_index()
    with st.expander("All 20 model and reconciliation combinations, ranked"):
        st.dataframe(rank, hide_index=True, width="stretch",
                     column_config={"Average error (RMSE, pp)": st.column_config.NumberColumn(format="%.2f")})
        st.caption("Average RMSE over every level and horizon in the backtest (notebook 03).")
    st.markdown(
        """
**What not to use it for**
* **The level.** It runs about 1 pp above Zillow for the typical metro; budget on the more conservative one.
* **A recession.** A 2008-style crash with forced selling is outside what 2010-2026 can teach a linear model.
* **Precise ranges.** Its ranges ran narrow at turning points; plan on the 95% range.
* **Places whose economy has changed.** Beta is history since 2009.
"""
    )

st.divider()
st.caption(
    f"Data: Zillow Home Value Index and forecast (Zillow Research, used with attribution; data through "
    f"{D['meta']['data']}), Census housing units (V2025), FRED mortgage rates and unemployment. "
    f"[Source on GitHub]({REPO_URL}) · [Decision memo]({MEMO_URL})"
)
