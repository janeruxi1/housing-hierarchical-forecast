# Notebooks: reading guide

Read them in order. Each notebook opens with its question and a short answer.

| # | Notebook | The question in plain words | Main figure |
|---|---|---|---|
| 01 | [Data audit](01_data_audit.ipynb) | Can county values be added up to metros, states and the nation, and do they track Zillow's own series? | Rebuilt vs published US series; metro gap histogram |
| 02 | [Baselines and backtest](02_baselines_backtest.ipynb) | How well do simple rules forecast, at each level and horizon, and where do they break? | Error by horizon; metro errors over time |
| 03 | [Models and reconciliation](03_models_reconciliation.ipynb) | Do ARIMA, panel and Zillow-style models beat momentum, and does making forecasts add up cost accuracy? | Base models vs baselines; reconciliation skill |
| 04 | [Forecast and scenarios](04_scenarios.ipynb) | What does the model expect for the next 12 months, how does that compare with Zillow's forecast, and which metros take the biggest hit if rates rise? | Baseline vs scenarios; metro exposure |
| 05 | [Final test and challengers](05_final_test.ipynb) | Does the method picked on 2014-2021 still beat simple rules on 2023-2025, and how do Prophet, ETS and ARIMA compare? | Error by test date; challengers |

**Running them.** Each `.ipynb` is generated from the `.py` file beside it (the `.py` is the
source). Notebook 03 reads `data/processed/backtest_models.parquet`, written by
`python scripts/run_models.py` (about 2 minutes); notebook 05 also reads
`data/processed/challengers.parquet`, written by `python scripts/run_challengers.py` (about 25 minutes). With `HOUSING_SYNTHETIC=1` everything runs on
made-up data, as CI does.
