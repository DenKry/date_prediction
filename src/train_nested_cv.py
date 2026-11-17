"""Nested, event-grouped cross-validation for every model.

Outer loop : StratifiedGroupKFold(5) by event, repeated R times -> unbiased test scores.
Inner loop : Optuna TPE search, objective = mean ROC-AUC over a 3-fold
             StratifiedGroupKFold (by event) of the outer-training part.
Grouping by event means every test date comes from an event the model never saw,
which mirrors deployment: predicting matches for an upcoming event.

Usage: python src/train_nested_cv.py [--models LogReg LightGBM ...] [--repeats 2]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import PROCESSED, ROOT, build_pairs, feature_groups  # noqa: E402
from metrics import all_metrics  # noqa: E402
from models import MODELS, make_model, suggest  # noqa: E402

RESULTS = ROOT / "results"
TARGETS = ["y_mutual", "y_m_likes", "y_w_likes"]
N_TRIALS = {"Prior": 0, "LogReg-Attract": 15, "LogReg": 40, "kNN": 40, "DecisionTree": 50,
            "RandomForest": 30, "LightGBM": 60, "LightGBM-Directional": 40, "MLP": 30,
            "MultiTask-NN": 30}
NN = {"MLP", "MultiTask-NN"}
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load():
    f = PROCESSED / "pairs.csv"
    df = pd.read_csv(f) if f.exists() else build_pairs()
    groups = feature_groups(df)
    feats = groups["man"] + groups["woman"] + groups["pair"]
    return df, feats, groups


def inner_splits(Y, g, seed):
    return list(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(Y, Y["y_mutual"], g))


def tune(name, X, Y, g, groups, seed):
    """Return best params (+ epochs for neural nets) and the Optuna study."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if N_TRIALS[name] == 0:
        return {}, None
    splits = inner_splits(Y, g, seed)
    from sklearn.metrics import roc_auc_score

    def objective(trial):
        p = suggest(name, trial)
        aucs, eps = [], []
        for tr, va in splits:
            m = make_model(name, p, groups, seed)
            if name in NN:
                m.fit(X.iloc[tr], Y.iloc[tr], X.iloc[va], Y.iloc[va])
                eps.append(m.best_epoch if hasattr(m, "best_epoch") else 1)
            else:
                m.fit(X.iloc[tr], Y.iloc[tr])
            aucs.append(roc_auc_score(Y.y_mutual.iloc[va], m.predict_proba(X.iloc[va])))
        if eps:
            trial.set_user_attr("epochs", int(np.median(eps)))
        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=N_TRIALS[name])
    best = dict(study.best_params)
    if name in NN:
        best["epochs"] = study.best_trial.user_attrs["epochs"]
    if name in ("LightGBM", "LightGBM-Directional"):
        best["subsample_freq"] = 1
    return best, study


def run_fold(name, df, feats, groups, rep, fold, tr, te):
    t0 = time.time()
    seed = 1000 * rep + fold
    X, Y, g = df[feats], df[TARGETS], df.event_id.values
    best, study = tune(name, X.iloc[tr], Y.iloc[tr], g[tr], groups, seed)
    m = make_model(name, best, groups, seed).fit(X.iloc[tr], Y.iloc[tr])
    p_te, p_tr = m.predict_proba(X.iloc[te]), m.predict_proba(X.iloc[tr])
    oof = pd.DataFrame({"model": name, "repeat": rep, "fold": fold, "row": te,
                        "y": Y.y_mutual.values[te], "p": p_te})
    if name == "MultiTask-NN":
        pm, pw = m.predict_directional(X.iloc[te])
        oof["p_m_likes"], oof["p_w_likes"] = pm, pw
    if name == "LightGBM-Directional":
        oof["p_m_likes"] = m.mm.predict_proba(X.iloc[te])[:, 1]
        oof["p_w_likes"] = m.mw.predict_proba(X.iloc[te])[:, 1]
    row = {"model": name, "repeat": rep, "fold": fold, **all_metrics(Y.y_mutual.values[te], p_te),
           "train_auc": all_metrics(Y.y_mutual.values[tr], p_tr)["roc_auc"],
           "inner_cv_auc": study.best_value if study else np.nan,
           "seconds": time.time() - t0, "params": json.dumps(best)}
    return oof, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS + ["LightGBM-Directional"])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=10)
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    df, feats, groups = load()
    print(f"{len(df)} dates, {len(feats)} features, positives={df.y_mutual.mean():.3f}")
    tasks = []
    for rep in range(a.repeats):
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=rep)
        for fold, (tr, te) in enumerate(cv.split(df, df.y_mutual, df.event_id)):
            tasks.append((rep, fold, tr, te))
    for name in a.models:
        t0 = time.time()
        out = Parallel(n_jobs=a.jobs)(delayed(run_fold)(name, df, feats, groups, *t) for t in tasks)
        oof = pd.concat([o for o, _ in out])
        rows = pd.DataFrame([r for _, r in out])
        oof.to_csv(RESULTS / f"oof_{name}.csv", index=False)
        rows.to_csv(RESULTS / f"folds_{name}.csv", index=False)
        print(f"{name:22s} test AUC {rows.roc_auc.mean():.3f}±{rows.roc_auc.std():.3f}  "
              f"AP {rows.avg_precision.mean():.3f}  train AUC {rows.train_auc.mean():.3f}  "
              f"inner {rows.inner_cv_auc.mean():.3f}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
