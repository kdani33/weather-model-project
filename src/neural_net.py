"""Neural network (PyTorch) for predicting significant storm events.

Same split and preprocessing as the baseline and random forest, so the
comparison is apples to apples. Everything is tuned on VAL (2023) only;
TEST (2024) is looked at once, at the end.

Usage:  cd src && python neural_net.py
Writes to results/: nn_metrics.json, nn_permutation_importance.csv, nn_by_state.csv
"""
import copy
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from baseline import CATEGORICAL, NUMERIC, TARGET, make_preprocessor, split
from config import PROCESSED_DIR, RESULTS_DIR
from random_forest import best_f1_threshold, breakdown, metrics

# settings: seed keeps results the same every run
SEED = 0
HIDDEN = (64, 32)   # two hidden layers
DROPOUT = 0.3
LR = 1e-3
BATCH_SIZE = 512
MAX_EPOCHS = 40
PATIENCE = 5        # stop if val pr-auc doesn't improve for this many epochs


class Net(nn.Module):
    # small feed-forward network: linear -> relu -> dropout, twice, then 1 output
    def __init__(self, n_in):
        super().__init__()
        h1, h2 = HIDDEN
        self.layers = nn.Sequential(
            nn.Linear(n_in, h1), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(h2, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(1)  # raw score (logit), sigmoid comes later


def to_dense(x):
    # one-hot output can be sparse, torch wants a plain float array
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    return x.astype(np.float32)


def predict_proba(model, X):
    # turn raw scores into probabilities (no training, no dropout)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X))).numpy()


def train(X_train, y_train, X_val, y_val):
    # train with a weighted loss and keep the epoch with the best val pr-auc
    torch.manual_seed(SEED)
    model = Net(X_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # rare positives count more, same idea as class_weight="balanced"
    pos_weight = torch.tensor((1 - y_train.mean()) / y_train.mean(), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xt, yt = torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.float32))
    gen = torch.Generator().manual_seed(SEED)

    best_score, best_state, bad_epochs, history = -1.0, None, 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(Xt), generator=gen)
        total = 0.0
        for i in range(0, len(Xt), BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)

        score = average_precision_score(y_val, predict_proba(model, X_val))
        history.append({"epoch": epoch, "train_loss": round(total / len(Xt), 4),
                        "val_pr_auc": round(float(score), 4)})
        print(f"  epoch {epoch:2d} | train loss {total / len(Xt):.4f} | val pr-auc {score:.4f}")

        if score > best_score:
            best_score, best_state, bad_epochs = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"  early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)  # go back to the best epoch
    return model, history


def main():
    # load the data and split it by year
    df = pd.read_csv(PROCESSED_DIR / "events.csv")
    train_df, val, test = split(df)
    val, test = val.reset_index(drop=True), test.reset_index(drop=True)
    features = CATEGORICAL + NUMERIC
    print(f"train {len(train_df):,} | val {len(val):,} | test {len(test):,}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. preprocessing ---
    # fit on train only, then apply the same recipe to val and test
    prep = make_preprocessor()
    X_train = to_dense(prep.fit_transform(train_df[features]))
    X_val = to_dense(prep.transform(val[features]))
    X_test = to_dense(prep.transform(test[features]))
    y_train = train_df[TARGET].to_numpy()
    print(f"{X_train.shape[1]} input columns after preprocessing")

    # --- 2. training ---
    # train until val pr-auc stops improving, keep the best epoch
    print("\n[1/4] training the network ...")
    model, history = train(X_train, y_train, X_val, val[TARGET].to_numpy())

    # --- 3. evaluation ---
    # pick the cutoff on 2023 only, then use that same cutoff on 2024
    print("\n[2/4] evaluating ...")
    p_val, p_test = predict_proba(model, X_val), predict_proba(model, X_test)
    thr = best_f1_threshold(val[TARGET], p_val)
    results = {
        "val": metrics(val[TARGET], p_val, thr),
        "test": metrics(test[TARGET], p_test, thr),
        "history": history,
    }
    print(pd.DataFrame({k: results[k] for k in ("val", "test")}).T.to_string())

    # --- 4. permutation importance ---
    # shuffle one raw feature at a time and see how much val pr-auc drops
    print("\n[3/4] permutation importance (2023 sample) ...")
    rng = np.random.default_rng(SEED)
    sample = val.sample(n=min(10_000, len(val)), random_state=SEED).reset_index(drop=True)
    y_s = sample[TARGET].to_numpy()
    base = average_precision_score(y_s, predict_proba(model, to_dense(prep.transform(sample[features]))))
    rows = []
    for feat in features:
        drops = []
        for _ in range(3):
            shuffled = sample.copy()
            shuffled[feat] = rng.permutation(shuffled[feat].to_numpy())
            score = average_precision_score(
                y_s, predict_proba(model, to_dense(prep.transform(shuffled[features]))))
            drops.append(base - score)
        rows.append({"feature": feat, "pr_auc_drop": np.mean(drops), "std": np.std(drops)})
    imp = pd.DataFrame(rows).sort_values("pr_auc_drop", ascending=False)
    imp.to_csv(RESULTS_DIR / "nn_permutation_importance.csv", index=False)
    print(imp.round(4).to_string(index=False))

    # --- 5. bias check ---
    # per-state scores on 2024, same table format as the random forest
    print("\n[4/4] per-state metrics (2024) ...")
    breakdown(test, p_test, thr, "state").to_csv(RESULTS_DIR / "nn_by_state.csv", index=False)

    # save everything to results/ so we don't have to copy from the terminal
    with open(RESULTS_DIR / "nn_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved to {RESULTS_DIR / 'nn_metrics.json'}")


if __name__ == "__main__":
    main()