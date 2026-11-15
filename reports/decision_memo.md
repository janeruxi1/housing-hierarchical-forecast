# Decision memo: where to limit housing exposure over the next 12 months

**To:** Portfolio risk committee (mortgage lending, mortgage insurance or a home-buying investor) · **From:** Xi Ru, Data Science · **Status:** recommendation
**Forecast date: September 2026.** Based on the Zillow Home Value Index for 1,371 counties holding 84% of US homes, Census housing-unit counts, and FRED mortgage rates and unemployment. A statistical forecast, not a guarantee; see Limitations.

## Recommendation

1. **Treat a rise in mortgage rates as the main 12-month risk, and size it as a range.** A sudden
   1-point rise lowers next-year home-value growth by 0.8 to 2.1 pp nationally, which is $0.45 to $1.2
   trillion of housing value. A 2-point rise lowers it by 1.6 to 4.2 pp. The range is model
   uncertainty: the larger number comes from the model as backtested, the smaller from the same model
   refitted without the 2020-21 pandemic years.
2. **Tighten in the ten most rate-sensitive large metros:** Las Vegas, Phoenix, Cape Coral, Riverside,
   Orlando, Miami, Tampa, North Port, Sacramento and Detroit. These metros swing 1.5 to 2.1 times as hard as
   the nation (their beta). A 1-point rate rise takes 2.9 to 4.2 pp off their 12-month growth, against
   1.0 to 1.2 pp in Pittsburgh, Buffalo, Rochester, Oklahoma City and Louisville. In practice:
   * *Lender:* a lower maximum loan-to-value, or risk-based pricing, for new loans in these metros.
   * *Mortgage insurer:* concentration limits on new insurance written there.
   * *Home-buying investor:* a smaller buy box and shorter planned holding periods.
3. **Budget on the conservative growth level, and use this model for ranking and stress.** This model
   expects 3.2% growth over 12 months for the nation, against Zillow's 1.1%. Across 439 metros the
   two rank places alike (correlation 0.79), but this model sits a median 1.2 pp higher. Taking
   Zillow's level and adding this model's 2-point rate shock, 45 of the 50 largest metros lose value
   over the year. The largest falls are Cape Coral (-10.4%), Las Vegas (-9.5%) and Phoenix (-8.9%). Seven of
   the ten metros in item 2 already have a negative Zillow forecast, led by Cape Coral (-3.6%), North
   Port (-1.9%), Phoenix (-1.8%), Tampa (-1.7%) and Las Vegas (-1.5%).
4. **Plan with wider ranges than the model states.** Its 80% ranges covered only 73% of metro outcomes
   12 months ahead in the backtest, because misses bunch up at turning points. Use its 95% range (89%
   coverage; a median width of 16.0 pp for a metro) as the planning range.
5. **Do not use this model for a recession stress test.** Over 2010-2026, rate cuts went with rising
   prices, and unemployment did little on its own once the pandemic is set aside. So a "recession"
   scenario comes out positive. For a 2008-type downturn, use historical drawdowns instead.
6. **Re-run monthly**, after Zillow's release on the 16th. Review the limits early if mortgage rates
   move by half a point or more in a quarter.

## Why

1. **It beats the rule of thumb people actually use.** The usual shortcut is "the last 12 months'
   growth continues" (momentum). The chosen forecast averages a per-place ARIMA with a panel
   regression that reads mortgage rates and unemployment, and adds counties up so every level
   agrees. Over 47 quarterly forecast dates from 2014 to 2025, its 12-month error (RMSE of growth):

| Level | This forecast | Momentum | No change |
|---|---|---|---|
| Nation | 4.3 pp | 5.3 pp | 7.0 pp |
| States | 4.6 pp | 5.6 pp | 7.4 pp |
| Metros | 5.1 pp | 5.9 pp | 7.8 pp |
| Counties | 5.2 pp | 6.1 pp | 7.9 pp |

2. **It holds up at the turning point that matters for this decision.** In the 2022-23 rate shock,
   momentum kept forecasting pandemic-era growth. Metro 12-month error was 10.6 pp for momentum and
   8.8 pp for "no change", against 6.0 pp here. The model is not better everywhere: in the 2020-22
   boom it was slightly worse than momentum (8.0 vs 7.4 pp).
3. **The numbers add up.** Counties sum exactly to metros, states and the nation, so limits set by
   metro and capital set by nation describe the same book. The cost is small: the national forecast
   built from counties misses by 4.3 pp, against 3.8 pp when the nation is forecast on its own.
4. **The rate effect comes from 1,371 counties over 2010-2026,** not a single national series. A
   mortgage rate that has risen a point in a year and sits a point above its 3-year average has
   usually been followed by about 2 pp less county price growth over the next 12 months.

## Limitations

* **Coverage.** The model covers counties with a full monthly history since 2008: 84% of US homes,
  mostly urban. Its national "typical home" is about 25% pricier than Zillow's, and Connecticut is
  missing (its counties became planning regions in Census data).
* **Smoothed, revised data.** Zillow's index is smoothed and its history is revised, so short-horizon
  accuracy looks better than it would in real time. The backtest also uses today's version of history.
* **Associations, not causes.** The rate response is what usually followed rate moves in one
  boom-and-shock period. It is linear and immediate, and carried by one model.
* **Exposure is history.** Each metro's beta is measured from 2009 on. A place whose economy has changed
  since then will be mis-scaled.

## Next steps

1. Add listings, inventory and days-on-market data, the leading indicators in Zillow's own method,
   to close the gap with Zillow's level.
2. Build a 2008-style downturn scenario from historical peak-to-trough losses by metro.
3. Recalibrate the forecast ranges so 80% means 80%, for example by widening them after large recent misses.

*Details: [notebooks 01-04](../notebooks/README.md), [model card](../MODEL_CARD.md).*
