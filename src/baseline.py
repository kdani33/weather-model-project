"""Baseline: logistic regression on the cleaned events table.

Trains on TRAIN_YEARS, reports metrics on VAL_YEARS and TEST_YEARS.
Also reports a majority-class dummy so the numbers have context.

Usage:  python src/baseline.py
"""
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import PROCESSED_DIR, RESULTS_DIR, TEST_YEARS, TRAIN_YEARS, VAL_YEARS

CATEGORICAL = ["event_type", "state", "cz_type", "month"]
NUMERIC = ["hour", "hail_size_in", "wind_speed_kt", "magnitude_measured", "lat", "lon"]
TARGET = "significant"


def make_preprocessor():
    numeric = Pipeline([
        # add_indicator lets the model see "this value was missing"
        # (e.g. hail_size is always missing for wind events)
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = OneHotEncoder(handle_unknown="ignore", min_frequency=20)
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ])


def split(df):
    train = df[df["year"].isin(TRAIN_YEARS)]
    val = df[df["year"].isin(VAL_YEARS)]
    test = df[df["year"].isin(TEST_YEARS)]
    return train, val, test


def evaluate(model, X, y):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "positive_rate": round(float(y.mean()), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred, zero_division=0), 4),
        "f1": round(f1_score(y, pred, zero_division=0), 4),
        "pr_auc": round(average_precision_score(y, prob), 4),
        "roc_auc": round(roc_auc_score(y, prob), 4) if y.nunique() > 1 else None,
    }


def top_coefficients(model, k=15):
    names = model.named_steps["prep"].get_feature_names_out()
    coefs = model.named_steps["clf"].coef_[0]
    order = np.argsort(-np.abs(coefs))[:k]
    return [(names[i], round(float(coefs[i]), 3)) for i in order]


def main():
    df = pd.read_csv(PROCESSED_DIR / "events.csv")
    train, val, test = split(df)
    print(f"train {len(train):,} | val {len(val):,} | test {len(test):,}")

    features = CATEGORICAL + NUMERIC
    models = {
        "majority_dummy": Pipeline([("prep", make_preprocessor()),
                                    ("clf", DummyClassifier(strategy="prior"))]),
        "logistic_regression": Pipeline([
            ("prep", make_preprocessor()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
        ]),
    }

    results = {}
    for name, model in models.items():
        model.fit(train[features], train[TARGET])
        results[name] = {
            "val": evaluate(model, val[features], val[TARGET]),
            "test": evaluate(model, test[features], test[TARGET]),
        }

    print(pd.DataFrame({(m, s): r for m, sets in results.items()
                        for s, r in sets.items()}).T.to_string())

    coefs = top_coefficients(models["logistic_regression"])
    print("\nLargest logistic regression coefficients (scaled features):")
    for feat, c in coefs:
        print(f"  {c:+.3f}  {feat}")
    results["top_coefficients"] = coefs

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "baseline_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'baseline_metrics.json'}")


if __name__ == "__main__":
    main()
