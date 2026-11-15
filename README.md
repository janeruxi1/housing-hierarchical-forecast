# Hierarchical home-value forecasting: county to nation, coherent at every level

> **Status: in progress.** Data audit and baseline backtest are done on the real data (Zillow
> release through September 2026); models, reconciliation and scenarios are next.

**Question.** Where will US home values go over the next 1 to 12 months, for the nation, every
state, every metro and every county, so that the numbers agree with each other? And what happens
to that outlook if mortgage rates rise a point or a recession hits?

**Why it is not just "forecast each series".**
* **Forecasts must add up.** A lender or insurer that plans by county but reports by state needs
  county forecasts that sum to the state forecast. Forecasting each level on its own does not give
  that; *reconciliation* (MinTrace) does, and usually improves accuracy too.
* **Zillow's index is an average, not a total.** ZHVI is a value-weighted mean, so county values do not
  sum to a metro value. This project forecasts the *value of the housing stock* (ZHVI x housing
  units) instead, which does add up, and converts back to a typical home value at the end.
* **There is an external yardstick.** Zillow publishes the accuracy of its own metro forecast
  (RMSE of monthly growth at 1, 3 and 6 months), so the backtest can be compared with it.

## Data (all public, free)

| Source | Used for |
|---|---|
| [Zillow Research](https://www.zillow.com/research/data/): ZHVI by county, metro, state, nation (middle tier, smoothed, seasonally adjusted); ZHVF forecast | target; Zillow's published series and forecast as benchmarks |
| Census Population Estimates, Vintage 2025 | county housing units (weights), population and net domestic migration |
| FRED | 30-year mortgage rate, US and state unemployment, CPI |

Raw files are not committed. `scripts/download_data.py` fetches them and records URL, date and
SHA-256 in `data/raw/manifest.json`. Data from Zillow is used with attribution.

## Approach

```
download ─> county panel (balanced from 2005) ─> hierarchy: US > state / metro > county (grouped)
        ─> base forecasts (baselines, ETS/ARIMA, panel regression, Zillow-style national + regional)
        ─> reconcile (bottom-up, MinTrace) ─> rolling-origin backtest at 1/3/6/12 months
        ─> scenarios (rates +1pp, recession) ─> decision memo
```

| Stage | Notebook | Status |
|---|---|---|
| Data audit and coherence check | [01](notebooks/01_data_audit.ipynb) | done |
| Baselines and backtest harness | [02](notebooks/02_baselines_backtest.ipynb) | done |
| Models and reconciliation | 03 | planned |
| Scenarios | 04 | planned |

## Results so far

* **The panel.** 1,371 counties with a full monthly history since 2008 hold 84% of US homes and roll up to
  49 states and 605 metro groups. Rebuilt multi-county metros track Zillow's published growth closely
  (median 12-month growth RMSE 0.29 pp).
* **The bar to beat.** Of four no-model rules, *momentum* (last 12 months' growth continues) is best: metro
  12-month RMSE 5.9 pp vs 7.6 pp for "no change", direction right 84% of the time. It fails at turning
  points: 10.7 pp in the 2022-23 rate shock, worse than "no change" (8.1 pp).

![Metro 12-month errors over time](reports/figures/02_metro_error_over_time.png)

## Run it

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py      # about 60 files, a few minutes
python scripts/build_data.py         # builds data/processed/ and reports/metrics/data_audit.json
python scripts/build_notebooks.py --execute
pytest -q                            # runs on synthetic data, as CI does
```

## Limitations (stated up front)
* **Revised history.** Zillow restates ZHVI history (e.g. the January 2023 neural-Zestimate
  upgrade), so backtests use today's series, not what was known at each forecast date.
* **No historical Zillow forecasts.** Only the current ZHVF vintage is public, so the comparison with
  Zillow's own forecast is against its *published accuracy*, plus a forward check of today's vintage.
* **Fixed weights.** Housing units are held at the latest Census year, so node totals move with prices,
  not construction.

## Repository layout

```
src/housing/      config, ingest, hierarchy, panel build, checks, synthetic data, plots
scripts/          download_data.py, build_data.py, build_notebooks.py
notebooks/        percent-format .py sources and the generated .ipynb
tests/            pytest on synthetic data
data/             raw/ and processed/ (not committed), READMEs describe each file
reports/          figures/ and metrics/ written by the notebooks
```

MIT licence. Data: Zillow Research (attribution), US Census Bureau, FRED.
