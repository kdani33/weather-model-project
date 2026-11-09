"""Random forest (WORK IN PROGRESS).

Reuses the baseline's preprocessing and year-based split so the comparison
with logistic regression is apples to apples.

Done:
  - Fit a random forest on TRAIN_YEARS and evaluate on VAL / TEST years.

TODO:
  - Tune the decision threshold on VAL (2023) instead of using 0.5.
  - Tune hyperparameters (n_estimators, max_depth, min_samples_leaf) on VAL.
  - Permutation importance, grouped by original feature, scored with PR-AUC.
  - Reporting-bias check: metrics by cz_type and by state on TEST.
  - Ablation: refit without state/lat/lon/cz_type and compare PR-AUC.

"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from baseline import CATEGORICAL, NUMERIC, TARGET, evaluate, make_preprocessor, split
from config import PROCESSED_DIR

FEATURES = CATEGORICAL + NUMERIC


def main():
    df = pd.read_csv(PROCESSED_DIR / "events.csv")
    train, val, test = split(df)
    print(f"train {len(train):,} | val {len(val):,} | test {len(test):,}")

    model = Pipeline([
        ("prep", make_preprocessor()),
        ("clf", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=-1, random_state=0)),
    ])
    model.fit(train[FEATURES], train[TARGET])

    # Baseline's evaluate() uses a fixed 0.5 threshold; PR-AUC is threshold-free.
    for name, data in [("val", val), ("test", test)]:
        print(name, evaluate(model, data[FEATURES], data[TARGET]))

    # TODO: threshold tuning on VAL
    # TODO: permutation importance (sklearn.inspection.permutation_importance)
    # TODO: per-cz_type / per-state metrics on TEST
    # TODO: location-feature ablation


if __name__ == "__main__":
    main()