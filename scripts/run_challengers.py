"""Run the Prophet, AutoETS and AutoARIMA challengers through the backtest (notebook 05 reads the output).

Each (model, forecast date) is saved as it finishes, so the run can be stopped and resumed:

    python scripts/run_challengers.py                     # everything still missing
    python scripts/run_challengers.py --model prophet --max-seconds 150   # stop cleanly after ~150 s
    HOUSING_SYNTHETIC=1 python scripts/run_challengers.py  # small synthetic run (CI)

About 20 minutes in all on the real data with two cores (Prophet is most of it).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from housing import backtest as bt  # noqa: E402
from housing import finaltest, panel  # noqa: E402
from housing.config import data_dirs, use_synthetic  # noqa: E402
from housing.pipeline import eval_origins  # noqa: E402

MODELS = ("autoets", "autoarima", "prophet")
TUNING = ("prophet_cr95", "prophet_cr95_flex")      # run with --tune


def out_dir() -> Path:
    d = data_dirs()[1] / "challengers"
    d.mkdir(parents=True, exist_ok=True)
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=MODELS + TUNING, action="append")
    ap.add_argument("--tune", action="store_true", help="the two extra Prophet settings, on the 8 tuning dates")
    ap.add_argument("--max-seconds", type=float, default=float("inf"))
    args = ap.parse_args(argv)
    t0 = time.time()
    Y, S_df, tags, _ = panel.load_hierarchy()
    L = bt.log_wide(Y)
    series = finaltest.challenger_series(Y, tags)
    evo = eval_origins(L)
    origins = finaltest.tuning_origins(evo) if args.tune else finaltest.challenger_origins(evo)
    models = args.model or (TUNING if args.tune else MODELS)
    if use_synthetic():                       # keep CI quick
        series, origins = series[:4], origins[-4:]
    d = out_dir()
    todo = [(m, o) for m in models for o in origins
            if not (d / f"{m}_{o:%Y%m}.parquet").exists()]
    print(f"{len(series)} series x {len(origins)} dates; {len(todo)} (model, date) pieces to run")
    for m, o in todo:
        if time.time() - t0 > args.max_seconds:
            print("time limit reached; run again to continue")
            return 2
        t = time.time()
        finaltest.forecast_challengers(L, series, [o], m).to_parquet(d / f"{m}_{o:%Y%m}.parquet", index=False)
        print(f"  {m} {o:%Y-%m}: {time.time() - t:.0f} s", flush=True)
    parts = sorted(d.glob("*_*.parquet"))
    allp = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    allp.to_parquet(data_dirs()[1] / "challengers.parquet", index=False)
    print(f"done: {len(parts)} pieces, {len(allp):,} forecasts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
