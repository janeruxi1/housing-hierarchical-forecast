"""Draw the README's figures from the recorded results (no data needed).

    python scripts/make_readme_figure.py
        ->  reports/figures/00_readme_hero.png     the 50 largest metros under a 2-point rate rise
        ->  reports/figures/00_readme_summary.png  accuracy by period · this model vs Zillow · rate shock range
"""
import csv
import json
from pathlib import Path

# --- shared README figure style (same block in every repo) ---------------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29d"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 12, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.facecolor": "white", "savefig.facecolor": "white",
    "xtick.major.size": 0, "ytick.major.size": 0,
})


def save(fig, path):
    fig.tight_layout(w_pad=3)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print("wrote", path.name)
# ---------------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "reports" / "metrics"
FIG = ROOT / "reports" / "figures"


def rows(name):
    with open(M / name, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def minus(s):
    return s.replace("-", "−")


# --- Hero: Zillow's forecast, and the same with this model's 2-point rate shock -----------------
expo = rows("04_metro_exposure.csv")
falls = json.loads((M / "04_stress_falls.json").read_text())
m = sorted(({"name": r["metro"].split(",")[0], "z": float(r["Zillow 12m (%)"]),
             "s": float(r["Zillow + rates +2 (%)"])} for r in expo), key=lambda r: r["s"])
fig, ax = plt.subplots(figsize=(15, 5.6))
for i, r in enumerate(m):
    c = ORANGE if r["s"] < 0 else AQUA
    ax.plot([i, i], [r["z"], r["s"]], color=GRID, lw=2.2, zorder=1)
    ax.scatter(i, r["z"], s=34, color=GREY, zorder=2, label="Zillow's 12-month forecast" if i == 0 else None)
    ax.scatter(i, r["s"], s=46, color=c, zorder=3,
               label="… with a 2-point rate rise (this model's shock)" if i == 0 else None)
ax.axhline(0, color=INK, lw=1)
ax.set_xticks(range(len(m)))
ax.set_xticklabels([r["name"] for r in m], rotation=60, ha="right", fontsize=9)
ax.set_xlim(-0.8, len(m) - 0.2)
ax.set_ylabel("Home-value growth, next 12 months (%)")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: minus(f"{v:+.0f}%")))
ax.set_title(f"Under a 2-point rate rise, {falls['metros_of_50_falling']} of the 50 largest metros lose value "
             "over the next year")
ax.legend(frameon=False, loc="lower right")
ax.grid(axis="x", visible=False)
save(fig, FIG / "00_readme_hero.png")

# --- Summary: three panels ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": [1.2, 1, 0.9]})

ax = axes[0]
reg = {(r["method"], r["regime"]): float(r["rmse_pp"]) for r in rows("03_final_by_regime.csv") if r["h"] == "12"}
periods = ["Recovery 2012-20", "Pandemic boom", "Rate shock", "Flat 2024+"]
labels = {"combo+bottom_up": ("Chosen forecast", BLUE), "momentum": ("Momentum", GREY), "naive": ("No change", GRID)}
w = 0.27
for k, (meth, (lab, col)) in enumerate(labels.items()):
    xs = [i + (k - 1) * w for i in range(len(periods))]
    ax.bar(xs, [reg[(meth, p)] for p in periods], w, color=col, label=lab)
ax.set_xticks(range(len(periods)))
ax.set_xticklabels([p.replace(" 2012-20", "\n2012-20").replace(" boom", "\nboom").replace(" shock", "\nshock")
                    .replace(" 2024+", "\n2024+") for p in periods])
ax.set_ylabel("Metro 12-month error (RMSE, pp)")
ax.set_title("It beats the rule of thumb at the turn")
ax.legend(frameon=False, fontsize=9.5)
ax.grid(axis="x", visible=False)

ax = axes[1]
zf = json.loads((M / "04_vs_zillow_forecast.json").read_text())
pts = [(float(r["growth_pct"]), float(r["ours_pct"])) for r in rows("04_vs_zillow_forecast_metros.csv")
       if r["unique_id"] != "US"]
ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=12, alpha=0.55, color=BLUE, lw=0)
lo, hi = -6, 9
ax.plot([lo, hi], [lo, hi], color=MUTED, ls="--", lw=1)
ax.text(hi - 0.4, hi - 2.3, "equal", color=MUTED, ha="right", fontsize=9.5)
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel("Zillow's forecast, 12 months (%)")
ax.set_ylabel("This model (%)")
ax.set_title(f"Same ranking as Zillow, higher level\n(correlation {zf['correlation']:.2f}, "
             f"{zf['metros_compared']} metros)")

ax = axes[2]
sc = {(r["scenario"], r["coefficients"]): float(r["vs baseline (pp)"]) for r in rows("04_scenarios.csv")}
names = [("Rates +1 pt", "Rates +1"), ("Rates +2 pt (stress)", "Rates +2"), ("Rates -1 pt", "Rates −1")]
for i, (key, lab) in enumerate(names):
    a, b = sc[(key, "pandemic left out")], sc[(key, "full sample")]
    col = ORANGE if b < 0 else AQUA
    ax.plot([a, b], [i, i], color=col, lw=7, solid_capstyle="round", alpha=0.85)
    ax.text(min(a, b) - 0.15, i, minus(f"{min(a, b):+.1f}"), ha="right", va="center", fontsize=9.5, color=MUTED)
    ax.text(max(a, b) + 0.15, i, minus(f"{max(a, b):+.1f}"), ha="left", va="center", fontsize=9.5, color=MUTED)
ax.axvline(0, color=INK, lw=1)
ax.set_yticks(range(len(names)))
ax.set_yticklabels([n[1] for n in names])
ax.set_ylim(len(names) - 0.4, -0.6)
ax.set_xlim(-5.8, 3.6)
ax.set_xlabel("US 12-month growth vs baseline (pp)")
ax.set_title("A rate shock, as a range\n(with and without 2020-21 in training)")
ax.grid(axis="y", visible=False)
save(fig, FIG / "00_readme_summary.png")
