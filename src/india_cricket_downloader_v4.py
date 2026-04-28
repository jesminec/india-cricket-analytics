"""
india_cricket_downloader_v4.py
-------------------------------
Downloads and filters Cricsheet ball-by-ball CSV2 datasets and
ESPNcricinfo Statsguru innings-level batting/bowling data for India.

Usage:
    python india_cricket_downloader_v4.py --output-dir ../data/raw
"""

import os
import io
import zipfile
import argparse
import requests
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Cricsheet download config
# ---------------------------------------------------------------------------
CRICSHEET_URLS = {
    "cricsheet_tests.csv":  "https://cricsheet.org/downloads/tests_male_csv2.zip",
    "cricsheet_odis.csv":   "https://cricsheet.org/downloads/odis_male_csv2.zip",
    "cricsheet_t20is.csv":  "https://cricsheet.org/downloads/t20s_male_csv2.zip",
    "cricsheet_ipl.csv":    "https://cricsheet.org/downloads/ipl_male_csv2.zip",
}

INDIA_TEAMS = {"India"}


def download_cricsheet(output_dir: Path) -> None:
    """Download Cricsheet ZIP archives and filter for India matches."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in CRICSHEET_URLS.items():
        out_path = output_dir / filename
        if out_path.exists():
            print(f"  [skip] {filename} already exists")
            continue
        print(f"  Downloading {url} ...")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                raise ValueError(f"No CSV found in {url}")
            with zf.open(csv_names[0]) as f:
                df = pd.read_csv(f, low_memory=False)
        india_mask = (
            df["batting_team"].isin(INDIA_TEAMS) |
            df["bowling_team"].isin(INDIA_TEAMS)
        )
        df_india = df[india_mask].reset_index(drop=True)
        df_india.to_csv(out_path, index=False)
        print(f"  Saved {len(df_india):,} rows -> {out_path}")


def scrape_espncricinfo(output_dir: Path) -> None:
    """
    Scrape ESPNcricinfo Statsguru innings tables.
    Full pagination logic is in notebook 01_data_acquisition.ipynb.
    """
    print("ESPNcricinfo scraping: see notebook 01_data_acquisition.ipynb for full implementation.")


def main():
    parser = argparse.ArgumentParser(description="Download India cricket datasets")
    parser.add_argument("--output-dir", default="../data/raw", help="Output directory")
    parser.add_argument("--cricsheet-only", action="store_true")
    parser.add_argument("--espn-only", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    if not args.espn_only:
        print("Downloading Cricsheet data...")
        download_cricsheet(out)
    if not args.cricsheet_only:
        print("ESPNcricinfo data:")
        scrape_espncricinfo(out)
    print("Done.")


if __name__ == "__main__":
    main()
