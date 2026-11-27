# Model card: hierarchical home-value forecast

## What it does
Forecasts the growth of home values 1 to 12 months ahead for the United States, 49 states, 605 metro
groups and 1,371 counties, so that every level adds up. It also shifts that forecast under mortgage-rate
scenarios. The forecast target is the value of each county's housing stock: Zillow's typical home value
(ZHVI) times the county's housing units.

## Intended use
* Ranking places by expected growth and by sensitivity to mortgage rates.
* Stress-testing a book of loans, policies or homes against a rate shock, by metro.
* Not intended for: valuing individual homes; recession or credit-crisis scenarios (see Limitations);
  replacing a forecast that sees listings and demand data (such as Zillow's own) for the *level* of growth.

## Data
| Source | Use | Period |
|---|---|---|
| Zillow ZHVI, middle tier, smoothed and seasonally adjusted (county, metro, state, US) | target; checks | 2008-01 to 2026-09 |
| Zillow ZHVF (metro forecast, one vintage) | benchmark | base 2026-09 |
| Census Population Estimates V2025, housing units by county | weights | 2025 |
| FRED: 30-year mortgage rate, US and state unemployment | inputs | 2008-2026 |

Counties enter only with a full monthly history since January 2008, after filling gaps of up to
6 months: 1,371 counties, 84% of US housing units. Connecticut is missing: Census reports planning
regions, Zillow the old counties.

## Model
| Part | Method |
|---|---|
| Base forecast A | ARIMA(2,1,0) with drift per node, 5-year rolling window |
| Base forecast B | Direct multi-horizon panel regression, pooled by level. Inputs: growth over 1, 3 and 12 months and the year before, national 12-month growth, mortgage-rate change over 12 months, rate vs its 3-year average, unemployment change |
| Combination | Equal-weight average of A and B |
| Coherence | Bottom-up: county forecasts summed through the summing matrix |
| Ranges | Time-safe conformal: forecast +/- a pooled multiple of the node's past error spread |
| Scenarios | County panel coefficients x input change, scaled by each county's beta to national growth (shrunk to 1, average 1) |

Chosen by backtest from 26 candidates: 5 base models and 4 reconciliation methods for each, plus naive.
The selection rule was the lowest RMSE averaged equally over the four levels and four horizons.
Applied to 2014-2021 alone, the same rule picks zillow_style+mint_shrink instead, which then fails the
final test (see Evaluation).

## Evaluation
Rolling origin: 47 quarterly forecast dates from January 2014 to September 2025. Each forecast uses
only data available at its date, and a test checks this. RMSE of 12-month growth:

| Level | This model | Momentum | No change |
|---|---|---|---|
| Nation | 4.3 pp | 5.3 pp | 7.0 pp |
| States | 4.6 pp | 5.6 pp | 7.4 pp |
| Metros | 5.1 pp | 5.9 pp | 7.8 pp |
| Counties | 5.2 pp | 6.1 pp | 7.9 pp |

* **By period, metros, 12 months:**
  * 2022-23 rate shock: 6.0 pp, against 10.6 pp for momentum.
  * 2020-22 boom: 8.0 pp, against 7.4 pp for momentum, so worse there.
* **Final test, rules written first** ([plan](docs/final_test_plan.md)): choose on 2014-2021, score once
  on 2023-2025. The method picked scores 7.6 pp at metro 12 months against momentum's 4.5 pp: it fails.
  No candidate beats "no change" (4.3 pp); this model ties momentum (4.5 pp). At 1-3 months it stays well
  ahead at every level.
* **Challengers** (50 largest metros, 12 months, 2023-25): AutoETS 3.4 pp (damped trend), AutoARIMA
  5.1 pp, Prophet 6.3 pp after tuning on 2014-2021.
* **Ranges:** nominal 80% ranges cover 73% of metro outcomes at 12 months, and 95% ranges cover 89%.
* **Against Zillow:** Zillow's published RMSE for its metro forecast is 0.0105 at 1 month and 0.0388 at 6 months
  (top 100 metros, 2019-2021). Here, over the same metros and forecast dates, the figures are 0.0020
  and 0.0303. This is not like for like: Zillow scores against its unsmoothed index, which is much
  harder to forecast one month out.

## Limitations and risks
* **12-month accuracy is not proven out of sample.** The gain over momentum comes from the 2022 shock;
  on 2023-2025 nothing beats "no change" a year out. Use the model for ranking, short horizons and stress.
* **Optimistic level.** From September 2026 this model expects 3.2% national growth, against Zillow's 1.1%.
  It ranks metros like Zillow (correlation 0.79) but sits a median 1.2 pp higher.
* **Smoothing and revisions** flatter short-horizon accuracy. Backtests use today's revised history.
* **Rate response:**
  * The response is linear, immediate and associational, learned in 2010-2026.
  * It depends heavily on the 2020-21 pandemic: refitted without those years, the 1-point shock falls
    from -2.1 to -0.8 pp. Quote the range.
* **Unemployment** has a positive coefficient on the full sample, which is an artefact of 2020. Recession
  scenarios are therefore not credible here.
* **Under-covering ranges** at turning points: widen them for planning.
* **Coverage bias:** the panel is mostly urban. The national figure describes the panel's homes, not every US home.

## Maintenance
Re-run monthly after Zillow's release on the 16th:
`download_data.py` -> `build_data.py` -> `run_models.py` -> `run_challengers.py` -> `build_notebooks.py --execute`.
Then score the forecasts frozen in `reports/metrics/05_forward_forecasts_2026-09.csv` (notebook 05, section 5).
`tests/test_reported_numbers.py` fails if the README, memo or this card drift from the recorded results.
