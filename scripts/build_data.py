"""Build data/processed/ from data/raw/ (run scripts/download_data.py first).

    python scripts/build_data.py              # real data
    HOUSING_SYNTHETIC=1 python scripts/build_data.py   # writes synthetic raw files, then processes them (CI)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from housing import panel, synthetic  # noqa: E402
from housing.config import data_dirs, use_synthetic  # noqa: E402


def main() -> int:
    raw, out = data_dirs()
    if use_synthetic():
        synthetic.write_raw(raw)
    elif not (raw / "zhvi_county.csv").exists():
        print("data/raw is empty: run python scripts/download_data.py first.")
        return 1
    audit = panel.build(raw, out)
    print(json.dumps(audit, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
