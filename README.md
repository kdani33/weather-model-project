# Predicting Damaging Severe Weather Events

Can we predict whether a hail or thunderstorm wind event will cause significant damage
or casualties, using only information known at the time of the event? Data comes from the
[NOAA Storm Events Database](https://www.ncei.noaa.gov/access/storm-events-database/).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
cd src
python download.py   # ~10 yearly files from NOAA into data/raw/
python clean.py      # builds data/processed/events.csv and prints class balance
python baseline.py   # logistic regression; writes results/baseline_metrics.json
```

## Scope decisions (all in `src/config.py`)

| Decision | Choice | Why |
|---|---|---|
| Years | Train 2015–2022, validate 2023, test 2024 | Testing on unseen years avoids leakage between events in the same storm |
| Event types | Hail, Thunderstorm Wind | Both have a numeric magnitude measured at the event; tornado EF ratings come from damage surveys (leakage) |
| Label | Any injury/death, or ≥ $50K property + crop damage in 2024 dollars | Threshold may be adjusted after looking at the damage distribution |
| Blank damage | Treated as $0 | NOAA leaves the field blank when no damage is reported |
| Features | Event type, hail size / wind speed, measured vs. estimated, month, hour, state, zone type, lat/lon | Only information known at event time |

## Plan

- [x] Download and cleaning pipeline
- [x] Logistic regression baseline
- [ ] Random forest (scikit-learn)
- [ ] Neural network (PyTorch)
- [ ] Feature importance: coefficients, permutation importance
- [ ] Performance across location types (reporting-bias check)
