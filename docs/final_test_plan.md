# Final test: plan written before the test scores were computed

Written 2026-11-25, before any forecast in the test period below was scored. Notebook 05 follows
this plan exactly; any departure is listed in the notebook as a departure.

## Why

Notebook 03 chose the forecasting method and reported its accuracy on the same 47 forecast dates.
Choosing the best of 26 candidates and then scoring the winner on the dates it was chosen on
flatters it. The fix is to choose on one period and score once on a later one.

## Periods (quarterly forecast dates, as in notebook 03)

| Period | Forecast dates | Count | Used for |
|---|---|---|---|
| Selection | Jan 2014 to Oct 2021 | 32 | choosing the method |
| Gap | Jan to Oct 2022 | 4 | nothing: their 12-month outcomes run into the test period |
| Test | Jan 2023 to Sep 2025 | 11 | scoring the chosen method, once |

## Choice rule (selection period only)

The same rule as notebook 03, applied to the selection dates alone: among the coherent methods
(the five base forecasts, each reconciled four ways), choose the one with the lowest RMSE of growth
averaged over the four levels (nation, state, metro, county) and the four horizons (1, 3, 6 and 12 months).

## Pass rule (test period)

The chosen method counts as better than momentum (the last 12 months' growth continues) only if
both hold for metros at 12 months:

1. its RMSE of 12-month growth over the 11 test dates is lower than momentum's, and
2. it has the lower RMSE in at least 7 of the 11 test dates.

All other levels and horizons are reported, but they do not change the verdict.

## Challengers (reported, not part of the pass rule)

Three standard single-series models run in the same backtest:

* **Prophet** (piecewise-linear trend; no seasonality terms, since the series is already
  seasonally adjusted);
* **AutoETS** and **AutoARIMA** (statsforecast, non-seasonal).

They run on 100 series: the nation, the 49 states and the 50 largest metros. They use the 11 test
dates plus every other selection date (16), so 27 dates in all. Each forecasts its series on its own,
so they cannot be made to add up across levels. They are compared with the chosen method and with
momentum on the same series and dates.

## What this test cannot rule out

The model designs (for example, adding mortgage-rate inputs) were made after looking at the whole
history, including the 2022-23 rate shock. The test period was not used to *choose among* the
methods, but it is not data no one had seen. The only fully untouched test is the future: the
forecasts published from September 2026, scored as the outcomes arrive.

## Addendum, 2026-11-26: a fairer Prophet (added after the first test scores were seen)

With its default settings Prophet scored far worse than every other method, including at one month
ahead. Its trend may only bend in the first 80% of the history, so a forecast made in 2023 still
carries the 2020-22 boom forward. That is a fault of the setting, not of Prophet, so two more
settings are tried:

* `changepoint_range=0.95` (the trend may bend until the last 5% of the history);
* the same with `changepoint_prior_scale=0.5` (the trend bends more easily).

**Choice:** of the three settings, the one with the lowest RMSE averaged over levels (nation, state,
metro) and horizons on 8 selection dates (each January, 2014-2021). Only that setting is then run
on the 11 test dates. **Disclosed:** this addendum was written after the default Prophet's test
scores were seen; the test dates play no part in the choice.
