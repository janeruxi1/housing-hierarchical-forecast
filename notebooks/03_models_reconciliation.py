# %% [markdown]
# # 03 · Models and reconciliation: can anything beat momentum, and do the numbers add up?
#
# Notebook 02 set the bar: *momentum* (the last 12 months' growth continues) is hard to beat in calm
# markets and badly late at turning points. This notebook tries three models on every node, then
# makes the forecasts **coherent**: counties that add up to their metro and state, which add up to
# the nation.
#
# **The models** (details in `src/housing/models.py`; all fitted with closed-form least squares, so
# the full backtest runs in minutes):
#
# | Model | Idea | In plain words |
# |---|---|---|
# | **ar2** | ARIMA(2,1,0) with drift per node, 5-year window | each place extrapolates its own recent rhythm |
# | **panel** | direct regression, pooled across the nodes of a level, one per horizon: recent growth (1, 3, 12 months and the year before), national growth, mortgage-rate change and rate vs its 3-year average, unemployment change | what usually happens next to a place with this recent history, in this rate environment |
# | **zillow_style** | Zillow's published structure: national forecast x each node's (shrunk) sensitivity to the nation + a fading local gap | the nation sets the tide; each place rides it with its own beta |
# | **combo** | equal-weight average of ar2 and panel | two forecasters who look at different evidence, averaged |
#
# **Reconciliation**, in plain words: think of a company budget. Each department forecasts its
# own spending, and the department numbers never add up to the CFO's total. You could just add up
# the departments (*bottom-up*), or adjust everyone a little, adjusting most the forecasts that
# have been least reliable. That is **MinTrace**: the coherent set of forecasts closest to the
# originals, where "close" is measured by each forecast's past error (and, for *mint_shrink*, how
# errors move together). Four versions are compared: bottom-up, wls_struct (weights from the tree
# shape only), wls_var (weights from each node's past error variance) and mint_shrink (full
# error covariance, shrunk).
#
# **Scoring.** All methods, baselines included, are scored on the same 48 quarterly forecast dates
# (January 2014 to the last date with a 12-month outcome), at 1, 3, 6 and 12 months ahead.

# %% [markdown]
# | | |
# |---|---|
# | **Question** | Do models beat momentum, especially at turning points, and does making forecasts add up help or hurt accuracy? |
# | **Answer** | Yes. The chosen forecast averages a per-place ARIMA with a rates-aware panel regression, then adds counties up to every level (bottom-up). It cuts the metro 12-month error from 5.9 pp (momentum) to 5.1 pp, 14% lower. In the 2022-23 rate shock the drop is from 10.6 pp to 6.0 pp, 44% lower. Making the numbers add up costs almost nothing: at metro and county level, bottom-up is as accurate as the best MinTrace variant. Weighting by tree shape alone (wls_struct) is badly wrong once forecasts are in dollars. The intervals are too narrow: nominal 80% intervals cover about 73% of metro outcomes at 12 months. |
# | **Next** | Notebook 04 runs the rate and recession scenarios through the chosen model. |

# %%
import os
import sys
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from housing import backtest as bt
from housing import panel
from housing.config import data_dirs, use_synthetic
from housing.pipeline import BASES, RECONCILERS, eval_origins, run_all
from housing.plots import COLORS, save, style, write_metrics

style()
if use_synthetic():
    print("SYNTHETIC DATA: numbers below are made up (CI run).")
out = data_dirs()[1]
if (out / "backtest_models.parquet").exists() and not use_synthetic():
    long = pd.read_parquet(out / "backtest_models.parquet")       # written by scripts/run_models.py
    coefs = pd.read_parquet(out / "panel_coefficients.parquet")
else:
    r = run_all(save=False)
    long, coefs = r["long"], r["coefs"]
Y, S_df, tags, nodes = panel.load_hierarchy()
L = bt.log_wide(Y)
evo = eval_origins(L)
res = bt.add_context(long, bt.mase_scale(L))
del long
print(f"{len(evo)} forecast dates, {res['method'].nunique()} methods, {len(res):,} scored forecasts")

# %% [markdown]
# ## 1. Base models, before reconciliation
#
# RMSE of growth (pp). The column to watch is metro and county at 12 months, where momentum scored
# about 6 pp in notebook 02.

# %%
order = ["National", "State", "Metro", "County"]
base_methods = ["naive", *BASES]
scores = bt.score(res)
write_metrics("03_scores_all", scores)
base = scores[scores["method"].isin(base_methods)]
board = base.pivot_table(index=["level", "method"], columns="h", values="rmse_pp", observed=True).reindex(order, level=0)
board.round(2)

# %%
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
colors = {"naive": COLORS["grey"], "momentum": COLORS["ours"], "ar2": "#9467bd", "panel": "#2ca02c",
          "zillow_style": COLORS["zillow"], "combo": "#ff7f0e"}
for ax, level in zip(axes, order):
    for m in base_methods:
        g = base[(base["level"] == level) & (base["method"] == m)]
        ax.plot(g["h"], g["rmse_pp"], marker="o", label=m, color=colors[m])
    ax.set_title(level)
    ax.set_xlabel("Months ahead")
    ax.set_xticks([1, 3, 6, 12])
axes[0].set_ylabel("RMSE of growth (pp)")
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Base models vs the baselines, same 48 forecast dates", y=1.03)
save(fig, "03_base_models")

# %% [markdown]
# ## 2. Turning points: errors by period
#
# The 2022-23 rate shock is where momentum failed. Did models that see mortgage rates, or that
# lean on national growth, do better there?

# %%
reg = bt.score(res[res["method"].isin(base_methods) & res["level"].isin(["Metro", "County"])],
               by=("method", "level", "regime", "h"))
write_metrics("03_base_by_regime", reg)
reg[reg["h"] == 12].pivot_table(index=["level", "regime"], columns="method", values="rmse_pp", observed=True)[
    base_methods].round(2)

# %% [markdown]
# ## 3. Reconciliation: does making the numbers add up cost accuracy?
#
# For each base model, the coherent versions. Skill = 1 - RMSE / RMSE of the same model before
# reconciliation (positive = reconciliation helped).

# %%
rec_rows = []
for b in BASES:
    for rname in RECONCILERS:
        s = scores[scores["method"] == f"{b}+{rname}"].merge(
            scores[scores["method"] == b][["level", "h", "rmse_pp"]], on=["level", "h"], suffixes=("", "_base"))
        s["base"], s["reconciler"] = b, rname
        s["skill_vs_base"] = 1 - s["rmse_pp"] / s["rmse_pp_base"]
        rec_rows.append(s)
rec = pd.concat(rec_rows)
write_metrics("03_reconciliation", rec)
rec[rec["h"] == 12].pivot_table(index=["base", "reconciler"], columns="level", values="rmse_pp", observed=True)[order].round(2)

# %%
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
for ax, level in zip(axes, order):
    d = rec[(rec["level"] == level) & (rec["h"] == 12)].pivot(index="base", columns="reconciler", values="skill_vs_base")
    d = d[RECONCILERS] * 100
    d.plot.bar(ax=ax, legend=(level == "National"), width=0.8, rot=30)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylim(-40, 20)
    for i, b in enumerate(d.index):            # bars beyond the axis: print their value at the edge
        for j, rname in enumerate(RECONCILERS):
            v = d.loc[b, rname]
            if v < -40:
                ax.text(i - 0.3 + 0.2 * j, -39, f"{v:.0f}", rotation=90, fontsize=7, va="bottom", ha="center")
    ax.set_title(level)
    ax.set_xlabel("")
axes[0].set_ylabel("RMSE change vs base (%, + = better)")
axes[0].legend(frameon=False, fontsize=7)
fig.suptitle("12 months ahead: how each reconciliation changes each model's error", y=1.03)
save(fig, "03_reconciliation_skill")

# %% [markdown]
# **Reading the reconciliation result.**
# * **Bottom-up and wls_var are almost identical.** In dollars, a county's past error variance is
#   tiny next to a state's or the nation's, so wls_var leaves the counties nearly untouched and moves the
#   aggregates: in effect, bottom-up.
# * **wls_struct is badly wrong.** Its weights count counties but ignore size. It spreads the nation's
#   dollar disagreement over counties as if a rural county and Los Angeles were equally able to
#   absorb it, which wrecks the small ones.
# * **mint_shrink helps the weaker models and hurts the stronger ones.** With roughly 60 past error vectors for 2,000+
#   nodes, the estimated covariance is noisy even after shrinkage. It improves momentum and the
#   Zillow-style model, but adds noise to forecasts that are already good at county level.
# * **The trade-off.** Bottom-up makes the national forecast exactly the sum of counties, so the nation
#   gives up a little accuracy (combo: 3.8 pp as a direct forecast vs 4.3 pp added up at 12 months) in return for
#   every level agreeing.

# %% [markdown]
# ## 4. The chosen forecast
#
# Chosen on the backtest alone: the coherent method with the lowest average RMSE across the four
# levels and four horizons (each level weighted equally, so the 1,371 counties do not drown out the nation).

# %%
coherent = scores[scores["method"].str.contains(r"\+")]
avg = coherent.groupby("method", observed=True)["rmse_pp"].mean().sort_values()
chosen = avg.index[0]
mom = scores[scores["method"] == "momentum"].groupby("method", observed=True)["rmse_pp"].mean().iloc[0]
write_metrics("03_selection", {"chosen": chosen, "avg_rmse_pp": float(avg.iloc[0]),
                               "momentum_avg_rmse_pp": float(mom), "ranking": avg.round(4).to_dict()})
print(f"Chosen: {chosen}   (average RMSE {avg.iloc[0]:.2f} pp vs momentum {mom:.2f} pp)")
avg.head(8).round(3).to_frame("average RMSE (pp)")

# %%
final = scores[scores["method"].isin([chosen, "momentum", "naive"])]
tab = final.pivot_table(index=["level", "h"], columns="method", values="rmse_pp", observed=True).reindex(order, level=0)
tab["skill_vs_momentum"] = 1 - tab[chosen] / tab["momentum"]
write_metrics("03_final_vs_momentum", tab.reset_index())
tab.round(3)

# %% [markdown]
# By period, metros, 12 months ahead:

# %%
fr = bt.score(res[res["method"].isin([chosen, "momentum", "naive"]) & (res["level"] == "Metro")],
              by=("method", "regime", "h"))
write_metrics("03_final_by_regime", fr)
fr[fr["h"] == 12].pivot_table(index="regime", columns="method", values="rmse_pp", observed=True).round(2)

# %% [markdown]
# ## 5. What the panel model learned
#
# Coefficients of the 12-month county and metro regressions at the last forecast date, in natural
# units (growth in log points per unit of each feature). The mortgage-rate rows answer "what does
# a 1-point rise in mortgage rates over a year usually mean for the next 12 months' price growth?"

# %%
c12 = coefs[coefs["h"] == 12].set_index("level").drop(columns="h").T[["National", "State", "Metro", "County"]]
write_metrics("03_panel_coefficients_h12", c12.reset_index(names="feature"))
c12.round(4)

# %% [markdown]
# ## 6. How uncertain? Time-safe prediction intervals
#
# Each interval is the forecast +/- a multiple of the node's own past error spread, the multiple set
# from past errors of nodes at the same level (a simple conformal method). Only errors already known at
# the forecast date are used. A good 80% interval covers about 80% of outcomes.

# %%
cov = bt.conformal_coverage(res[res["method"] == chosen])
write_metrics("03_interval_coverage", cov)
cov.pivot_table(index=["level", "h"], columns="nominal", values=["coverage", "median_width_pp"]).reindex(
    order, level=0).round(3)

# %% [markdown]
# ## 7. Against Zillow's published accuracy (same caveats as notebook 02)

# %%
zc = bt.zillow_comparison(res[res["method"].isin([chosen, "momentum"])], Y)
write_metrics("03_zillow_comparison", zc.reset_index())
zc.round(4)
