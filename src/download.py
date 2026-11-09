"""Download NOAA Storm Events 'details' CSVs for the project years.

NOAA posts one gzipped file per year, named like
  StormEvents_details-ftp_v1.0_d2024_c20260323.csv.gz
where the c-date is a revision date that changes over time, so we read the
directory listing and take the newest revision for each year.

Usage:  python src/download.py
"""
import re
import sys

import requests

from config import ALL_YEARS, RAW_DIR

BASE_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
PATTERN = re.compile(r"StormEvents_details-ftp_v1\.0_d(\d{4})_c(\d{8})\.csv\.gz")


def latest_files(years):
    """Map each requested year to its newest details filename."""
    html = requests.get(BASE_URL, timeout=60).text
    newest = {}
    for m in PATTERN.finditer(html):
        name, year, cdate = m.group(0), int(m.group(1)), m.group(2)
        if year in years and (year not in newest or cdate > newest[year][1]):
            newest[year] = (name, cdate)
    return {y: name for y, (name, _) in newest.items()}


def download(years=ALL_YEARS):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = latest_files(years)
    missing = sorted(set(years) - set(files))
    if missing:
        print(f"Warning: no file found for {missing}", file=sys.stderr)

    for year, name in sorted(files.items()):
        out = RAW_DIR / name
        # Remove older revisions of the same year so clean.py reads one file per year
        for old in RAW_DIR.glob(f"StormEvents_details-ftp_v1.0_d{year}_c*.csv.gz"):
            if old.name != name:
                old.unlink()
        if out.exists():
            print(f"{year}: already have {name}")
            continue
        print(f"{year}: downloading {name}")
        with requests.get(BASE_URL + name, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(out, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)


if __name__ == "__main__":
    download()
