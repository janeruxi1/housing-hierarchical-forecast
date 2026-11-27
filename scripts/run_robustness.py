"""Robustness check on Zillow's raw index (notebook 06 reads the output).

    python scripts/download_data.py      # fetches data/raw/zhvi_county_raw.csv if missing
    python scripts/run_robustness.py     # build data/processed_raw/ and run the model backtest on it (~3 minutes)
    HOUSING_SYNTHETIC=1 python scripts/run_robustness.py

Steps can be run on their own with --build or --models.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from housing.config import data_dirs  # noqa: E402
from housing.robustness import RAW_FILE, build_raw_panel  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true", help="only build data/processed_raw/")
    ap.add_argument("--models", action="store_true", help="only run the model backtest on it")
    args = ap.parse_args(argv)
    both = not (args.build or args.models)
    raw, smoothed = data_dirs()
    if args.build or both:
        if not (raw / RAW_FILE).exists():
            print(f"{raw / RAW_FILE} is missing: run python scripts/download_data.py first.")
            return 1
        os.environ["HOUSING_ZHVI"] = "raw"
        audit = build_raw_panel(raw, smoothed, data_dirs()[1])
        print(json.dumps(audit, indent=2))
    if args.models or both:
        os.environ["HOUSING_ZHVI"] = "raw"
        from housing.pipeline import run_all
        print(json.dumps(run_all()["info"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
