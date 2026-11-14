# %% [markdown]
# # 01 · Data audit: can county ZHVI be added up to metros, states and the nation?
#
# Zillow's Home Value Index (ZHVI) is a *typical* home value: since 2019 it is a value-weighted
# mean of each home's Zestimate appreciation, with the level set to the mean Zestimate. An average
# does not add up: two counties' ZHVIs do not sum to their metro's. Forecasts that must agree from
# county to nation need something that does add up.
#
# This project forecasts **the value of each county's housing stock**: county ZHVI x county housing
# units (Census estimate, held fixed at the latest year). Totals add up exactly; a node's typical
# value is its total divided by its housing units.
#
# Before any modelling, three checks:
# 1. **Coverage.** How many counties have a full monthly history, and what share of US homes do they hold?
# 2. **Joins.** Census housing units have no FIPS codes, so they are matched to Zillow by county name.
#    Which counties fail to match?
# 3. **Coherence gap.** Rebuilt from counties, do our metro, state and national series track
#    Zillow's own published ones? Levels may differ (different weights); growth, the thing being
#    forecast, should agree closely.

# %% [markdown]
# | | |
# |---|---|
# | **Question** | Is a county-level, housing-unit-weighted panel a faithful base for forecasts at every level? |
# | **Answer** | _Filled in after the first run on the real data._ |
# | **Next** | Notebook 02 sets up baselines and the rolling backtest. |

# %%
import os
import sys
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from housing import checks, panel
from housing.config import data_dirs, metrics_dir, use_synthetic
from housing.plots import COLORS, save, shade_regimes, style, write_metrics

style()
if use_synthetic():
    print("SYNTHETIC DATA: numbers below are made up (CI run).")
audit = json.loads((metrics_dir() / "data_audit.json").read_text())
Y, S, tags, nodes = panel.load_hierarchy()
counties = panel.load_counties()
pub = panel.load_published()
coverage = pd.read_parquet(data_dirs()[1] / "coverage.parquet")
pd.Series(audit).to_frame("value")

# %% [markdown]
# ## 1. Coverage
#
# Counties enter the panel only if Zillow has a value for every month from January 2005 on.
# Thin rural counties often start later, so they drop out; the share of US housing units kept
# says how much that matters.

# %%
by_level = pd.DataFrame({"nodes": {k: len(v) for k, v in tags.items()}})
by_level["median housing units"] = [
    Y.loc[Y["unique_id"].isin(tags[k]), ["unique_id", "hu"]].drop_duplicates()["hu"].median() for k in by_level.index
]
by_level.round(0)

# %% [markdown]
# Coverage of each state's homes by the panel (housing units in kept counties / all matched counties):

# %%
cov_state = coverage[coverage["unique_id"].str.startswith("ST:")].sort_values("hu_coverage")
fig, ax = plt.subplots(figsize=(9, 2.8))
ax.bar(cov_state["unique_id"].str[3:], 100 * cov_state["hu_coverage"], color=COLORS["ours"])
ax.set_ylabel("% of housing units kept")
ax.set_ylim(0, 105)
ax.tick_params(axis="x", labelsize=7, rotation=90)
ax.set_title("Share of each state's homes in the balanced county panel")
save(fig, "01_state_coverage")

# %% [markdown]
# ## 2. Joins: counties that did not match a Census housing-unit count
#
# Matching is on state + normalised county name. Known mismatch: Connecticut replaced its 8
# counties with 9 planning regions in Census products from 2022; if Zillow still reports the old
# counties, they cannot be weighted and drop out.

# %%
pd.Series(audit["unmatched_counties_by_state"], name="unmatched counties").to_frame()

# %% [markdown]
# ## 3. Coherence gap: rebuilt series vs Zillow's published ones
#
# Compared only for nodes where the panel holds at least 95% of the housing units. Three numbers per node:
# * **level gap**: our typical value vs Zillow's, last month (expected to differ: unit vs value weights);
# * **growth correlation** of month-on-month changes;
# * **growth RMSE** in percentage points, monthly and year-on-year. This is the error we would
#   carry into every forecast just from rebuilding the series, so it is a floor to compare
#   forecast errors against.

# %%
gap = checks.coherence_gap(Y, pub, coverage)
write_metrics("01_coherence_gap", gap)
summary = gap.groupby("level").agg(
    nodes=("unique_id", "size"),
    median_level_gap_pct=("level_gap_pct_last", "median"),
    median_growth_corr=("growth_corr_1m", "median"),
    median_rmse_1m_pp=("growth_rmse_1m_pp", "median"),
    median_rmse_12m_pp=("growth_rmse_12m_pp", "median"),
).reindex(["National", "State", "Metro"]).dropna(how="all")
write_metrics("01_coherence_summary", summary.reset_index())
summary.round(3)

# %%
us = Y[Y["unique_id"] == "US"].set_index("ds")["typical_value"]
zus = checks.published_ids(pub).query("unique_id == 'US'").set_index("date")["zhvi"]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
axes[0].plot(us.index, us / 1e3, color=COLORS["ours"], label="Rebuilt from counties")
axes[0].plot(zus.index, zus / 1e3, color=COLORS["zillow"], label="Zillow published", ls="--")
axes[0].set_ylabel("Typical home value ($000)")
axes[0].set_title("United States: level")
axes[0].legend(frameon=False)
g_ours, g_z = 100 * np.log(us).diff(12), 100 * np.log(zus).diff(12)
axes[1].plot(g_ours.index, g_ours, color=COLORS["ours"])
axes[1].plot(g_z.index, g_z, color=COLORS["zillow"], ls="--")
axes[1].axhline(0, color="k", lw=0.6)
axes[1].set_ylabel("Year-on-year growth (%)")
axes[1].set_title("United States: growth")
for ax in axes:
    shade_regimes(ax)
fig.tight_layout()
save(fig, "01_us_rebuilt_vs_published")

# %%
metros = gap[gap["level"] == "Metro"]
fig, ax = plt.subplots(figsize=(6, 3))
ax.hist(metros["growth_rmse_12m_pp"], bins=30, color=COLORS["ours"])
ax.set_xlabel("Year-on-year growth RMSE vs Zillow (pp)")
ax.set_ylabel("Metros")
ax.set_title("How closely each rebuilt metro tracks Zillow's")
save(fig, "01_metro_gap_hist")
metros.sort_values("growth_rmse_12m_pp", ascending=False).head(10).round(3)

# %% [markdown]
# ## 4. What is being forecast
#
# Month-on-month log growth of the housing-stock value, by level. Counties are noisier than
# metros, metros than states: the usual reason reconciliation helps, since the smoother upper
# levels can steady the noisy bottom.

# %%
Y = Y.sort_values(["unique_id", "ds"])
Y["growth_pct"] = 100 * Y.groupby("unique_id")["y"].transform(lambda s: np.log(s).diff())
Y["level"] = Y["unique_id"].str.split(":").str[0].map({"US": "National", "ST": "State", "MSA": "Metro", "CTY": "County"})
vol = Y.groupby(["level", "unique_id"])["growth_pct"].std().groupby("level").median()
vol = vol.reindex(["National", "State", "Metro", "County"]).rename("median monthly growth s.d. (pp)")
write_metrics("01_volatility_by_level", vol.reset_index())
vol.round(3).to_frame()
