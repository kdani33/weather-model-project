"""Project-wide scope decisions. Change them here, not in the scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

# Years: train on 2015-2022, tune on 2023, final test on 2024
TRAIN_YEARS = list(range(2015, 2023))
VAL_YEARS = [2023]
TEST_YEARS = [2024]
ALL_YEARS = TRAIN_YEARS + VAL_YEARS + TEST_YEARS

# Event types with a numeric magnitude measured at the event.
# Tornado is excluded: its EF rating is assigned from the damage survey (leakage).
EVENT_TYPES = ["Hail", "Thunderstorm Wind"]

# Label: significant = any injury/death OR damage >= threshold (2024 dollars)
DAMAGE_THRESHOLD = 50_000

# Annual average CPI-U (BLS series CUUR0000SA0), used to convert damage to 2024 dollars.
# Double-check these against bls.gov before the final report.
CPI = {
    2015: 237.017, 2016: 240.007, 2017: 245.120, 2018: 251.107,
    2019: 255.657, 2020: 258.811, 2021: 270.970, 2022: 292.655,
    2023: 304.702, 2024: 313.689,
}
BASE_CPI_YEAR = 2024
