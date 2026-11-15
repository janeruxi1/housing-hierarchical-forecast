# Notebooks: reading guide

Read them in order. Each notebook opens with its question and a short answer.

| # | Notebook | The question in plain words | Main figure |
|---|---|---|---|
| 01 | [Data audit](01_data_audit.ipynb) | Can county values be added up to metros, states and the nation, and do they track Zillow's own series? | Rebuilt vs published US series; metro gap histogram |
| 02 | [Baselines and backtest](02_baselines_backtest.ipynb) | How well do simple rules forecast, at each level and horizon, and where do they break? | Error by horizon; metro errors over time |
| 03 | Models and reconciliation _(next)_ | Do statistical, panel and Zillow-style models beat them, and does reconciliation help? | |
| 04 | Scenarios _(planned)_ | What happens to each metro if rates rise 1 point, or a recession hits? | |

**Running them.** Each `.ipynb` is generated from the `.py` file beside it (the `.py` is the
source). With `HOUSING_SYNTHETIC=1` they run on made-up data, as CI does.
