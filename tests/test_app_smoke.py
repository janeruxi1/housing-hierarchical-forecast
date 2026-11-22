"""The Streamlit app runs end to end on the committed extract and shows the recorded results."""
import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "streamlit_app.py")
M = ROOT / "reports" / "metrics"


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def values(at):
    return {m.label: m.value for m in at.metric}


def test_headline_matches_recorded_metrics(app):
    z = json.loads((M / "04_vs_zillow_forecast.json").read_text())
    v = values(app)
    assert v["US growth, next 12 months"] == f"{z['us_ours_pct']:+.1f}%"
    assert v["Zillow's forecast, same month"] == f"{z['us_zillow_pct']:+.1f}%"
    stress = json.loads((M / "04_stress_falls.json").read_text())
    assert v["Large metros losing value under stress"] == f"{stress['metros_of_50_falling']} of 50"


def test_every_level_and_the_largest_places(app):
    for level in ["State", "County", "Metro"]:
        app.radio(key="level").set_value(level).run()
        for name in app.selectbox(key="place").options[:3]:
            app.selectbox(key="place").set_value(name).run()
            assert not app.exception, (level, name, app.exception)
    app.toggle(key="compare_us").set_value(False).run()
    for lvl in ["National", "State", "County"]:
        app.radio(key="acc_level").set_value(lvl).run()
    assert not app.exception, app.exception


def test_rate_shock_reproduces_notebook_04(app):
    rows = pd.read_csv(M / "04_scenarios.csv").set_index(["scenario", "coefficients"])
    cases = {("Rates +2 pt (stress)", "full sample"): (2.0, 0.0),
             ("Rates +1 pt", "pandemic left out"): (1.0, 0.0),
             ("Recession (unemployment +3, rates -1)", "full sample"): (-1.0, 3.0)}
    for (name, version), (rate, unemp) in cases.items():
        app.radio(key="version").set_value(version).run()
        app.select_slider(key="rate").set_value(rate).run()
        app.select_slider(key="unemp").set_value(unemp).run()
        assert not app.exception, app.exception
        r = rows.loc[(name, version)]
        v = values(app)
        assert v["US growth under this scenario"] == f"{r['US 12-month growth (%)']:+.1f}%"
        assert v["Change vs baseline"] == f"{r['vs baseline (pp)']:+.1f} pp"
    app.radio(key="version").set_value("full sample").run()
    app.select_slider(key="unemp").set_value(0.0).run()
    app.select_slider(key="rate").set_value(2.0).run()
    assert values(app)["Large metros losing value"] == "45 of 50"
    assert values(app)["Housing value vs baseline"] == "-$2.44 tn"
