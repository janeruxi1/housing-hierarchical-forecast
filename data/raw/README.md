# data/raw (not committed)

Filled by `python scripts/download_data.py`. `manifest.json` records the URL, pull date, size and
SHA-256 of every file, so results can be tied to a data vintage.

| File | Source | Licence / terms |
|---|---|---|
| `zhvi_county.csv`, `zhvi_metro.csv`, `zhvi_state.csv` | Zillow Home Value Index, middle tier, smoothed & seasonally adjusted | Zillow Research data, free with attribution to Zillow. Not redistributed here. |
| `zhvf_metro.csv` | Zillow Home Value Forecast (growth), same cut | as above |
| `housing_units_county.xlsx` | Census Population Estimates V2025, CO-EST2025-HU | Public domain (US government) |
| `population_county.csv` | Census Population Estimates V2025, co-est2025-alldata | Public domain |
| `fred/*.csv` | FRED: MORTGAGE30US, UNRATE, CPIAUCSL, state unemployment (`{ST}UR`) | Freddie Mac / BLS via FRED; see FRED terms |
