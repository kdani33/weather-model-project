"""Turn raw NOAA files into one clean modeling table.

Steps:
  1. Load the yearly details files and keep only our event types.
  2. Parse damage strings ("10.00K", "2.5M") into dollars, adjust to 2024 dollars.
  3. Build the label: significant = any injury/death OR damage >= threshold.
  4. Keep only features known at the time of the event.

Output: data/processed/events.csv
Usage:  python src/clean.py
"""
import numpy as np
import pandas as pd

from config import (ALL_YEARS, BASE_CPI_YEAR, CPI, DAMAGE_THRESHOLD,
                    EVENT_TYPES, PROCESSED_DIR, RAW_DIR)

RAW_COLS = [
    "EVENT_ID", "EPISODE_ID", "YEAR", "BEGIN_YEARMONTH", "BEGIN_TIME",
    "EVENT_TYPE", "STATE", "CZ_TYPE", "MAGNITUDE", "MAGNITUDE_TYPE",
    "BEGIN_LAT", "BEGIN_LON",
    "INJURIES_DIRECT", "INJURIES_INDIRECT", "DEATHS_DIRECT", "DEATHS_INDIRECT",
    "DAMAGE_PROPERTY", "DAMAGE_CROPS",
]

MULTIPLIERS = {"H": 1e2, "K": 1e3, "M": 1e6, "B": 1e9}


def parse_damage(value):
    """'10.00K' -> 10000.0, '2.5M' -> 2500000.0, '0' -> 0.0, blank -> NaN."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip().upper()
    if s == "":
        return np.nan
    mult = 1.0
    if s[-1] in MULTIPLIERS:
        mult = MULTIPLIERS[s[-1]]
        s = s[:-1]
    try:
        return float(s or 0) * mult
    except ValueError:
        return np.nan


def load_raw(years=ALL_YEARS):
    frames = []
    for year in years:
        files = sorted(RAW_DIR.glob(f"StormEvents_details-ftp_v1.0_d{year}_c*.csv.gz"))
        if not files:
            print(f"Warning: no raw file for {year}, run download.py")
            continue
        df = pd.read_csv(files[-1], usecols=RAW_COLS, low_memory=False)
        frames.append(df)
        print(f"{year}: {len(df):,} events")
    return pd.concat(frames, ignore_index=True)


def clean(df):
    df = df[df["EVENT_TYPE"].isin(EVENT_TYPES)].copy()

    # --- Damage -> dollars, adjusted for inflation ---
    # DECISION: blank damage is treated as $0 (NOAA leaves it blank when none reported).
    prop = df["DAMAGE_PROPERTY"].map(parse_damage).fillna(0)
    crop = df["DAMAGE_CROPS"].map(parse_damage).fillna(0)
    inflation = df["YEAR"].map(lambda y: CPI[BASE_CPI_YEAR] / CPI[y])
    df["damage_2024"] = (prop + crop) * inflation

    # --- Label ---
    casualties = df[["INJURIES_DIRECT", "INJURIES_INDIRECT",
                     "DEATHS_DIRECT", "DEATHS_INDIRECT"]].fillna(0).sum(axis=1)
    df["casualties"] = casualties
    df["significant"] = ((casualties > 0) |
                         (df["damage_2024"] >= DAMAGE_THRESHOLD)).astype(int)

    # --- Features known at event time ---
    df["month"] = df["BEGIN_YEARMONTH"] % 100
    df["hour"] = (df["BEGIN_TIME"] // 100).clip(0, 23)
    # Magnitude means different things per event type, so split it into two columns
    df["hail_size_in"] = np.where(df["EVENT_TYPE"] == "Hail", df["MAGNITUDE"], np.nan)
    df["wind_speed_kt"] = np.where(df["EVENT_TYPE"] == "Thunderstorm Wind",
                                   df["MAGNITUDE"], np.nan)
    # MAGNITUDE_TYPE: EG/MG = estimated/measured gust, ES/MS = estimated/measured sustained
    df["magnitude_measured"] = df["MAGNITUDE_TYPE"].astype(str).str.startswith("M").astype(int)

    out = df.rename(columns={
        "EVENT_ID": "event_id", "EPISODE_ID": "episode_id", "YEAR": "year",
        "EVENT_TYPE": "event_type", "STATE": "state", "CZ_TYPE": "cz_type",
        "BEGIN_LAT": "lat", "BEGIN_LON": "lon",
    })[[
        "event_id", "episode_id", "year", "event_type", "state", "cz_type",
        "month", "hour", "hail_size_in", "wind_speed_kt", "magnitude_measured",
        "lat", "lon", "damage_2024", "casualties", "significant",
    ]]
    return out


def summarize(df):
    print(f"\nKept {len(df):,} events ({', '.join(EVENT_TYPES)})")
    print(f"Positive rate (significant): {df['significant'].mean():.2%}")
    print("\nPositive rate by year:")
    print(df.groupby("year")["significant"].agg(["count", "mean"]).round(3))
    print("\nPositive rate by event type:")
    print(df.groupby("event_type")["significant"].agg(["count", "mean"]).round(3))
    print("\nMissing values in features:")
    print(df.isna().mean().round(3)[lambda s: s > 0])


if __name__ == "__main__":
    raw = load_raw()
    events = clean(raw)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    events.to_csv(PROCESSED_DIR / "events.csv", index=False)
    summarize(events)
    print(f"\nSaved to {PROCESSED_DIR / 'events.csv'}")
