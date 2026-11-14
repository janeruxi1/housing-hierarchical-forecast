"""Download the public source files into data/raw/ (nothing here is committed to git).

Sources (all free, no login):
  * Zillow Research: ZHVI by county, metro (incl. national) and state; ZHVF metro forecast.
  * Census Population Estimates, Vintage 2025: housing units and population by county.
  * FRED CSV export: 30-year mortgage rate, US and state unemployment, CPI.

Usage (from the project root):
    python scripts/download_data.py            # download anything missing
    python scripts/download_data.py --force    # re-download everything (e.g. after Zillow's monthly update on the 16th)

Each download is recorded in data/raw/manifest.json (URL, date pulled, bytes, sha256), so the
README can say exactly which vintage the results come from.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from housing.config import (  # noqa: E402
    CENSUS_FILES, FRED_CSV, FRED_NATIONAL, FRED_STATE_UNEMP, RAW, ZILLOW_DATA_PAGE, ZILLOW_FILES,
)

HEADERS = {"User-Agent": "Mozilla/5.0 (portfolio research project; contact via GitHub)"}


def fetch(url: str, tries: int = 3) -> bytes:
    for i in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=120)
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            if i == tries - 1:
                raise
            print(f"   retry {i + 1} after: {e}")
            time.sleep(3 * (i + 1))
    raise RuntimeError("unreachable")


def zillow_fallback(url: str) -> str | None:
    """Zillow occasionally moves its CSVs: look the file name up on the data page."""
    name = url.rsplit("/", 1)[-1]
    try:
        page = fetch(ZILLOW_DATA_PAGE).decode("utf-8", "ignore")
    except requests.RequestException:
        return None
    hits = re.findall(r'https://[^"\']+?' + re.escape(name), page)
    return hits[0] if hits else None


def save(name: str, content: bytes, url: str, manifest: dict) -> None:
    path = RAW / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    manifest[name] = {
        "url": url,
        "pulled": dt.date.today().isoformat(),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    print(f"   saved {name} ({len(content) / 1e6:.1f} MB)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = ap.parse_args(argv)
    RAW.mkdir(parents=True, exist_ok=True)
    mpath = RAW / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}

    jobs = list(ZILLOW_FILES.items()) + list(CENSUS_FILES.items())
    jobs += [(f"fred/{s}.csv", FRED_CSV.format(series=s)) for s in [*FRED_NATIONAL, *FRED_STATE_UNEMP]]
    failed = []
    for name, url in jobs:
        if (RAW / name).exists() and not args.force:
            continue
        print(f"-> {name}")
        try:
            save(name, fetch(url), url, manifest)
        except requests.RequestException as e:
            alt = zillow_fallback(url) if "zillowstatic" in url else None
            if alt and alt != url:
                print(f"   moved; trying {alt}")
                try:
                    save(name, fetch(alt), alt, manifest)
                    continue
                except requests.RequestException:
                    pass
            print(f"   FAILED: {e}")
            failed.append((name, url))
        time.sleep(0.3)  # be polite to FRED

    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    if failed:
        print("\nCould not download (get them by hand from the URL, save under data/raw/ with this name):")
        for name, url in failed:
            print(f"  {name}  <-  {url}")
        return 1
    print("\nAll files present. Next: python scripts/build_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
