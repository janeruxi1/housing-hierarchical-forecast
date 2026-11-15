# How this project was built, phase by phase

Each phase lists the question it answers, what was done, the design decisions and why they were
made, what it produced, and the objections a careful reviewer would raise along with how the
project answers them. Numbers come from `reports/metrics/` and are checked by
`tests/test_reported_numbers.py`.

| Phase | Question | Where |
|---|---|---|
| 0 | What decision does this inform, and what is already known? | this page, README |
| 1 | Can public data be turned into a county panel that is complete and joinable? | `scripts/download_data.py`, `scripts/build_data.py`, `src/housing/ingest.py`, `panel.py` |
| 2 | Can county values be added up to metros, states and the nation? | notebook 01, `src/housing/hierarchy.py`, `checks.py` |
| 3 | How good is "no model", and where does it break? | notebook 02, `src/housing/backtest.py` |
| 4 | Can models beat the best simple rule, above all at a turning point? | notebook 03, `src/housing/models.py` |
| 5 | Does making forecasts add up cost accuracy? | notebook 03, `src/housing/reconcile.py` |
| 6 | How uncertain are the forecasts, honestly? | notebook 03 |
| 7 | What does the model expect now, and how does that compare with Zillow? | notebook 04, `src/housing/scenarios.py` |
| 8 | What does a rate shock do, and to which metros? | notebook 04, decision memo |
| 9 | Can anyone rerun it and get the same numbers? | `tests/`, CI, `requirements-lock.txt` |
| 10 | Review: what was strengthened, and what is left | this page |

---

## Phase 0 · Framing

**Question.** A lender, mortgage insurer or home-buying investor sets exposure limits by metro and
holds capital for the whole book. It needs home-value forecasts 1 to 12 months ahead for every
county, metro, state and the nation that agree with each other, and it needs to know how much a
rise in mortgage rates would change them.

**What was already known.**
* Zillow publishes its Home Value Index (ZHVI) and its own metro forecast (ZHVF). It also publishes
  the method: a national model plus a model of each metro's gap to the nation. And it publishes the
  accuracy for the top 100 metros, 2019-2021: an RMSE of growth of 0.0105 at 1 month, 0.0213 at 3
  and 0.0388 at 6.
* Forecast reconciliation (bottom-up, MinTrace) is the standard way to make forecasts at several
  levels agree (Wickramasuriya, Athanasopoulos and Hyndman, 2019; Nixtla's `hierarchicalforecast`).

**Decisions.**
* *Unit:* the county, the lowest level with Census housing-unit counts. Counties group two ways,
  into states and into metros, and metros cross state lines, so the hierarchy is *grouped*, not a
  single tree.
* *Horizons:* 1, 3, 6 and 12 months, close to Zillow's own (1, 3 and 12).
* *Scope cut up front:* no listing or inventory data (not public at county level), no Spark, no
  gradient boosting. The point is the forecasting design, not model breadth.

---

## Phase 1 · Data

**Question.** Can free public files become a monthly county panel with weights?

**What was done.**
* `scripts/download_data.py` fetches Zillow's county, metro, state and national ZHVI (middle tier,
  smoothed, seasonally adjusted), Zillow's metro forecast, the Census housing-unit and population
  estimates (Vintage 2025), and FRED series (30-year mortgage rate, US and state unemployment,
  CPI). It records the URL, date and SHA-256 of every file in `data/raw/manifest.json`, and looks
  a moved Zillow file up on the data page.
* `scripts/build_data.py` joins it all into `data/processed/`.

**Decisions.**
* *Join by name.* The Census housing-unit table has no FIPS codes, so it is joined to Zillow by
  state and a normalised county name. That handles "Saint" vs "St.", "De Kalb" vs "DeKalb", "La
  Salle" vs "LaSalle", "O'Brien" and accents, with a fallback for Alaska's "Borough" vs
  "Municipality". Only 9 counties stay unmatched: Connecticut's 8 old counties (the Census now
  reports planning regions) and 1 in North Dakota.
* *Fill short holes, never extrapolate.* 433 counties had gaps of 1 to 5 months. These were filled
  by log-linear interpolation, 1,043 months in all, and recorded.
* *Start in January 2008.* A balanced panel from 2005 kept only 70% of US homes; from 2008 it keeps
  84% (1,371 counties) and still trains on the 2008-11 bust.
* *Raw data is never committed.* Zillow's data is used with attribution but not redistributed, so
  tests and CI run on synthetic files in the same formats (`src/housing/synthetic.py`).

---

## Phase 2 · Making it add up

**Question.** ZHVI is a value-weighted *average*, so county values do not sum to a metro value.
What can be added up?

**What was done.** Each county's value is its ZHVI times its housing units (held at the latest Census
year). That is the dollar value of its housing stock, and it does sum: metros, states and the nation
are sums of counties, and a node's typical home value is its total divided by its units. Notebook 01
compares the rebuilt series with Zillow's published ones.

**What it found.**
* Multi-county metros track Zillow's growth closely: a median 12-month growth RMSE of 0.29 pp.
  States track it at 0.34 pp.
* The nation is the exception. Its level sits about 25% above Zillow's, because the counties left
  out are mostly small, rural and cheaper.

**Reviewer objection:** "Your national series isn't Zillow's." Correct, and the README says so: the
national figure describes the panel's homes, 84% of the US.

---

## Phase 3 · Baselines and the backtest harness

**Question.** How accurate are rules a planner could apply without a model?

**What was done.** A rolling-origin backtest: stand at every month-end from January 2012, use only data
up to then, forecast 1, 3, 6 and 12 months ahead, and score against what happened. That is 165 forecast
dates and 5.3 million forecasts across four rules: no change, drift, momentum (the last 12 months'
growth continues) and Zillow's long-run anchor. A test changes the future and checks that past
forecasts do not move.

**What it found.** Momentum is the bar to beat: a metro 12-month RMSE of 5.9 pp against 7.6 pp for "no
change". But it fails at turning points: 10.7 pp in the 2022-23 rate shock, worse than "no change".

---

## Phase 4 · Models

**Question.** Can models beat momentum, especially when rates turn?

**What was done.** Three models, each fitted on every node with closed-form least squares, so a full
backtest runs in about two minutes:
* *ar2:* ARIMA(2,1,0) with drift per node, on a 5-year rolling window;
* *panel:* a direct multi-horizon regression, pooled across the nodes of each level. It reads recent
  growth, national growth, the change in mortgage rates over a year, rates against their 3-year
  average, and the change in unemployment. "Direct" means no future rate is ever needed;
* *zillow_style:* Zillow's published structure, with the panel's national forecast, each node's beta to
  the nation (shrunk towards 1) and a fading local gap;
* plus *combo,* the average of ar2 and panel.

**Decisions.**
* *Scoring set.* Models are scored on 47 quarterly forecast dates from 2014, so every model has at
  least four years of training data. The baselines are rescored on the same dates.
* *Selection rule.* The winner is the lowest RMSE averaged equally over the four levels and four
  horizons, so the 1,371 counties do not drown out the nation.

**What it found.** Combo, added up from counties, wins: metro 12-month RMSE 5.1 pp, 14% below momentum.
In the rate shock it falls from 10.6 to 6.0 pp. It is not better everywhere: in the 2020-22 boom it was
slightly worse than momentum (8.0 vs 7.4 pp).

---

## Phase 5 · Reconciliation

**Question.** Does forcing every level to agree cost accuracy?

**What was done.** Bottom-up and three MinTrace variants, reconciled in dollars:
* wls_struct, weighted by the tree shape;
* wls_var, weighted by each node's past error variance;
* mint_shrink, using the full error covariance, shrunk.

The MinTrace code is written for 2,000+ nodes: a sparse summing matrix, and the covariance kept as
diagonal plus low rank. Tests check it against the textbook formula and against
`hierarchicalforecast`. The weights use only errors already known at each forecast date.

**What it found.**
* At metro and county level, bottom-up is as accurate as the best MinTrace variant.
* wls_struct fails badly in dollars, because it spreads the nation's disagreement over small counties.
* Full-covariance MinTrace helps weak base models but adds noise to strong ones.
* The cost of coherence is at the top: the nation built from counties misses by 4.3 pp, against 3.8 pp
  when forecast on its own.

---

## Phase 6 · Uncertainty

**Question.** Do the forecast ranges mean what they say?

**What was done.** Time-safe conformal ranges: each forecast ± a multiple of the node's own past error
spread, the multiple set from past errors at the same level.

**What it found.** They are too narrow. Nominal 80% ranges cover 73% of metro outcomes at 12 months, and
95% ranges cover 89%, because misses bunch up at turning points. The memo plans with the 95% range.

---

## Phase 7 · The forecast from today

**Question.** What does the model expect from September 2026, and does it agree with Zillow?

**What it found.**
* About 3% national growth over 12 months (3.2%), against Zillow's own 1.1%.
* Across 439 metros the two forecasts rank places alike (correlation 0.79), but this model sits a
  median 1.2 pp higher.

**Decision.** The memo takes the *level* from the more conservative forecast, and the *ranking* and
*rate sensitivity* from this one, which is what it was built and backtested to deliver.

---

## Phase 8 · Rate scenarios

**Question.** How much would a rise in mortgage rates change the outlook, and where?

**What was done.**
* *Shock:* how the county panel regression's forecast moves when its rate inputs change.
* *Exposure:* each county's beta to the national cycle since 2009.
* Counties are then added up, so every level stays coherent.
* *Model uncertainty:* the regression is refitted without the 2020-21 pandemic years.

**What it found.**
* A 1-point rise takes 0.8 to 2.1 pp off national 12-month growth, which is $0.45 to $1.2 trillion of
  housing value. The range is the gap between the two fits.
* High-beta Sun Belt metros take about four times the hit of Rust Belt and Northeast ones (Las Vegas
  -4.2 pp, Pittsburgh -1.0 pp).
* A "recession" scenario (unemployment +3, rates -1) comes out *positive*. Over 2010-2026, rate cuts
  went with rising prices, and unemployment did little on its own once the pandemic is set aside.
  So the memo does not use this model for recession stress.

---

## Phase 9 · Reproducibility

* **Tests.** `pytest` covers:
  * the joins, gap filling and the hierarchy (every level sums to the nation);
  * no peeking in the backtest;
  * reconciliation against the textbook formula and the library;
  * scenario coherence;
  * every number quoted in the README, memo, model card and notebook summaries, checked against
    `reports/metrics`.
* **CI** runs lint, the tests, the notebook-source check, and the whole pipeline and all four notebooks
  on synthetic data. It then checks that the recorded metrics were not touched.
* **Pinned versions.** `requirements-lock.txt` holds the exact library versions that produced the
  results (Python 3.10), and CI installs it.
* **Rerun on the real data** with the commands in the README ("Run it"): about 6 minutes after the
  download. A Codespaces dev container installs the pinned versions and builds the synthetic data on start.

---

## Phase 10 · Review: what was strengthened, and what is left

**Found and fixed during the build.**
* *Joins.* County names first left 49 counties unmatched; now 9 are.
* *Louisville.* Its metro was named differently from Zillow's file, so it was missing from the comparison
  with Zillow's forecast. The fix changed one stress result (44 to 45 of 50 metros falling).
* *Rounding.* The reported-numbers test caught momentum's rate-shock error quoted as 10.7 pp where the
  47 quarterly dates give 10.6 pp. 10.7 is notebook 02's figure over 165 monthly dates.
* *An overclaim.* "Bottom-up is as accurate as the best MinTrace" holds at metro and county level, not
  nationally; the wording now says so. A README line saying reconciliation "usually improves accuracy"
  was removed, because the results here do not support it.
* *Memory.* The backtest tables (9.9 million rows) were made compact so the analysis fits in 4 GB.

**Left, in order of value.**
1. A live demo (Streamlit), like the other projects.
2. A downturn scenario built from historical peak-to-trough losses by metro, in place of the linear
   recession scenario.
3. Ranges recalibrated after large recent misses, so that 80% means 80%.
4. Listings, inventory and days-on-market data, the leading indicators in Zillow's own method, to
   close the gap with Zillow's level.
