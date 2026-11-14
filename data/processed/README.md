# data/processed (not committed)

Built by `python scripts/build_data.py` from `data/raw/`.

| File | One row per | What |
|---|---|---|
| `county_month.parquet` | county x month | ZHVI, housing units (fixed, latest Census year), `value` = ZHVI x units |
| `hierarchy_month.parquet` | node x month | `y` = total value ($), `hu` = housing units, `typical_value` = y / hu, for US, states, metros, counties |
| `S.parquet`, `nodes.parquet` | node / county | summing matrix and each county's parents |
| `coverage.parquet` | node | share of the node's housing units in the balanced panel |
| `zillow_published.parquet` | region x month | Zillow's own metro, state and national ZHVI (for the coherence check) |
| `zhvf_metro.parquet` | metro x horizon | Zillow's published forecast (one vintage) |
| `macro_month.parquet`, `state_unemp.parquet`, `population_county.parquet` | month / state-month / county-year | drivers |
