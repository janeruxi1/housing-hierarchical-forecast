"""Run the model backtest (notebook 03 reads its output). Takes a few minutes on the real data.

    python scripts/run_models.py
    HOUSING_SYNTHETIC=1 python scripts/run_models.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from housing.pipeline import run_all  # noqa: E402

if __name__ == "__main__":
    print(json.dumps(run_all()["info"], indent=2))
