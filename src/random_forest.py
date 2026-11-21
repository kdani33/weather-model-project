"""Random forest and the analyses built around it.

Reuses the baseline's split and feature lists so the comparison with logistic
regression is apples to apples. Everything is tuned on VAL (2023) only; TEST
(2024) is looked at once, at the end.

Steps:
  1. Tune the random forest on VAL (small grid, selected by PR-AUC).
  2. Choose each model's decision threshold on VAL (max F1), report VAL and TEST.
  3. Permutation importance (grouped by original feature, scored by PR-AUC).
  4. Reporting-bias check: metrics by cz_type and by state on TEST.
  5. Ablation: refit without location features (state, lat, lon, cz_type).

Usage:  cd src && python random_forest.py
Writes to results/: rf_metrics.json, rf_permutation_importance.csv,
                    rf_by_cz_type.csv, rf_by_state.csv
"""
import itertools
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from baseline import CATEGORICAL, NUMERIC, TARGET, split
from config import PROCESSED_DIR, RESULTS_DIR

# settings: seed keeps results the same every run
SEED = 0
# these are the columns we drop for the ablation
LOCATION_FEATURES = ["state", "lat", "lon", "cz_type"]
MIN_GROUP_N = 500  # skip tiny groups, their metrics are too noisy
# the few settings we try when tuning the forest
GRID = {"max_depth": [None, 20], "min_samples_leaf": [5, 20]}


def make_preprocessor(categorical, numeric):
    """Same recipe as baseline.make_preprocessor, but with configurable columns
    so the ablation can drop location features."""
    # same cleanup as the baseline, just lets us pick which columns to use
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    cat = OneHotEncoder(handle_unknown="ignore", min_frequency=20)
    parts = [("num", num, numeric)]
    if categorical:
        parts.append(("cat", cat, categorical))
    return ColumnTransformer(parts)


def forest(categorical, numeric, n_estimators=300, **params):
    # preprocessing + random forest in one pipeline
    return Pipeline([
        ("prep", make_preprocessor(categorical, numeric)),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED, **params)),
    ])


def best_f1_threshold(y, prob):
    # finds the cutoff that gives the best f1 score
    p, r, t = precision_recall_curve(y, prob)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[np.argmax(f1)])


def metrics(y, prob, threshold):
    # turns probabilities into yes/no and scores them (precision, recall, etc.)
    y = np.asarray(y)
    pred = (prob >= threshold).astype(int)
    two = len(np.unique(y)) > 1
    return {
        "n": int(len(y)),
        "positive_rate": round(float(y.mean()), 4),
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(y, prob)), 4) if two else None,
        "roc_auc": round(float(roc_auc_score(y, prob)), 4) if two else None,
    }


def evaluate_tuned(model, features, val, test):
    """Pick threshold on VAL, report VAL and TEST at that threshold."""
    # pick the cutoff on 2023 only, then use that same cutoff on 2024
    p_val = model.predict_proba(val[features])[:, 1]
    p_test = model.predict_proba(test[features])[:, 1]
    thr = best_f1_threshold(val[TARGET], p_val)
    return {"val": metrics(val[TARGET], p_val, thr),
            "test": metrics(test[TARGET], p_test, thr)}, p_test, thr


def breakdown(df, prob, threshold, col):
    # scores the model separately for each group (like each state)
    rows = []
    y = df[TARGET].to_numpy()
    for value, idx in df.groupby(col).indices.items():
        if len(idx) < MIN_GROUP_N:
            continue
        rows.append({col: value, **metrics(y[idx], prob[idx], threshold)})
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def main():
    # load the data and split it by year
    df = pd.read_csv(PROCESSED_DIR / "events.csv")
    train, val, test = split(df)
    val, test = val.reset_index(drop=True), test.reset_index(drop=True)
    features = CATEGORICAL + NUMERIC
    print(f"train {len(train):,} | val {len(val):,} | test {len(test):,}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. tuning ---
    # try each setting combo and keep the one with the best pr-auc on 2023
    print("\n[1/5] Tuning random forest on 2023 ...")
    tuning, best, best_score = [], None, -1.0
    for depth, leaf in itertools.product(GRID["max_depth"], GRID["min_samples_leaf"]):
        m = forest(CATEGORICAL, NUMERIC, n_estimators=200,
                   max_depth=depth, min_samples_leaf=leaf)
        m.fit(train[features], train[TARGET])
        score = average_precision_score(val[TARGET], m.predict_proba(val[features])[:, 1])
        tuning.append({"max_depth": depth, "min_samples_leaf": leaf,
                       "val_pr_auc": round(float(score), 4)})
        print(f"  max_depth={depth}, min_samples_leaf={leaf}: val PR-AUC {score:.4f}")
        if score > best_score:
            best, best_score = {"max_depth": depth, "min_samples_leaf": leaf}, score
    print(f"  best: {best}")

    # --- 2. final models ---
    # train logistic regression and the tuned forest, then score both on 2023 and 2024
    print("\n[2/5] Fitting final models ...")
    models = {
        "logistic_regression": Pipeline([
            ("prep", make_preprocessor(CATEGORICAL, NUMERIC)),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))]),
        "random_forest": forest(CATEGORICAL, NUMERIC, **best),
    }
    results, test_probs, thresholds = {"rf_tuning": tuning, "rf_best_params": best}, {}, {}
    for name, model in models.items():
        model.fit(train[features], train[TARGET])
        res, p_test, thr = evaluate_tuned(model, features, val, test)
        results[name], test_probs[name], thresholds[name] = res, p_test, thr
    print(pd.DataFrame({(m, s): results[m][s] for m in models
                        for s in ("val", "test")}).T.to_string())

    # --- 3. permutation importance ---
    # shuffle one feature at a time and see how much the score drops, bigger drop = more important
    print("\n[3/5] Permutation importance (random forest, 2023 sample) ...")
    sample = val.sample(n=min(10_000, len(val)), random_state=SEED)
    pi = permutation_importance(models["random_forest"], sample[features],
                                sample[TARGET], scoring="average_precision",
                                n_repeats=5, random_state=SEED, n_jobs=-1)
    imp = (pd.DataFrame({"feature": features, "pr_auc_drop": pi.importances_mean,
                         "std": pi.importances_std})
           .sort_values("pr_auc_drop", ascending=False))
    imp.to_csv(RESULTS_DIR / "rf_permutation_importance.csv", index=False)
    print(imp.round(4).to_string(index=False))

    # --- 4. bias check ---
    # see if the model works equally well across location types and states in 2024
    print("\n[4/5] Metrics by location type and state (2024) ...")
    for col in ["cz_type", "state"]:
        tables = []
        for name in models:
            t = breakdown(test, test_probs[name], thresholds[name], col)
            t.insert(0, "model", name)
            tables.append(t)
        out = pd.concat(tables, ignore_index=True)
        out.to_csv(RESULTS_DIR / f"rf_by_{col}.csv", index=False)
        if col == "cz_type":
            print(out[out["model"] == "random_forest"].to_string(index=False))
    print(f"  (per-state table saved to {RESULTS_DIR / 'rf_by_state.csv'})")

    # --- 5. ablation ---
    # retrain without any location info to see how much the model leans on where the event happened
    print("\n[5/5] Ablation: no state / lat / lon / cz_type ...")
    cat_nl = [c for c in CATEGORICAL if c not in LOCATION_FEATURES]
    num_nl = [c for c in NUMERIC if c not in LOCATION_FEATURES]
    ablated = forest(cat_nl, num_nl, **best)
    feats_nl = cat_nl + num_nl
    ablated.fit(train[feats_nl], train[TARGET])
    res_nl, _, _ = evaluate_tuned(ablated, feats_nl, val, test)
    results["random_forest_no_location"] = res_nl
    print(f"  with location:    test PR-AUC {results['random_forest']['test']['pr_auc']}, "
          f"ROC-AUC {results['random_forest']['test']['roc_auc']}")
    print(f"  without location: test PR-AUC {res_nl['test']['pr_auc']}, "
          f"ROC-AUC {res_nl['test']['roc_auc']}")

    # save everything to results/ so we don't have to copy from the terminal
    with open(RESULTS_DIR / "rf_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'rf_metrics.json'}")


if __name__ == "__main__":
    main()