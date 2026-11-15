import numpy as np
import pandas as pd

from housing import ingest


def test_metro_short_matches_zillow_style():
    assert ingest.metro_short("Seattle-Tacoma-Bellevue, WA") == "Seattle, WA"
    assert ingest.metro_short("New York-Newark-Jersey City, NY-NJ-PA") == "New York, NY"
    assert ingest.metro_short("Tucson, AZ") == "Tucson, AZ"          # already short
    assert ingest.metro_short("Louisville/Jefferson County, KY-IN") == "Louisville, KY"
    assert ingest.metro_short("") == ""
    assert ingest.metro_short(np.nan) == ""


def test_norm_name_joins_zillow_and_census_spellings():
    pairs = [("Doña Ana County", "Dona Ana County"), ("Saint Louis County", "St. Louis County"),
             ("De Kalb County", "DeKalb County"), ("La Salle County", "LaSalle County"),
             ("O Brien County", "O'Brien County"), ("Sainte Genevieve County", "Ste. Genevieve County")]
    for zillow, census in pairs:
        assert ingest.norm_name(zillow) == ingest.norm_name(census)
    assert ingest.norm_name("St. Louis city") != ingest.norm_name("St. Louis County")


def test_short_key_fallback_for_alaska():
    assert ingest.short_key(ingest.norm_name("Anchorage Borough")) == ingest.short_key(
        ingest.norm_name("Anchorage Municipality"))
    assert ingest.short_key(ingest.norm_name("Juneau City and Borough")) == "juneau"


def test_read_zillow_wide_is_long_and_month_end(tmp_path):
    pd.DataFrame({"RegionID": ["1"], "RegionName": ["X"], "2020-01-31": [1.0], "2020-02-29": [np.nan],
                  "2020-03-31": [3.0]}).to_csv(tmp_path / "z.csv", index=False)
    long = ingest.read_zillow_wide(tmp_path / "z.csv")
    assert len(long) == 2                                   # missing month dropped, not zero-filled
    assert (long["date"] == long["date"] + pd.offsets.MonthEnd(0)).all()


def test_housing_units_parse(built):
    from housing.config import data_dirs

    hu = ingest.load_housing_units(data_dirs()[0])
    assert len(hu) == built["counties_in_zillow"]           # title, US and note rows skipped
    assert (hu["hu"] >= hu["hu_2020"]).all()                # latest year picked, not the 2020 base


def test_weekly_mortgage_rate_becomes_monthly(built):
    from housing.config import data_dirs

    s = ingest.load_fred(data_dirs()[0], "MORTGAGE30US")
    assert (s.index == s.index + pd.offsets.MonthEnd(0)).all()
    assert s.index.to_series().diff().dropna().dt.days.max() <= 31
