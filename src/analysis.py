"""EDA figures, model comparison (event-cluster bootstrap CIs), training dynamics,
learning curves, complexity curves and feature importance.

Run after train_nested_cv.py:  python src/analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import ROOT, load_users  # noqa: E402
from metrics import all_metrics  # noqa: E402
from models import MODELS, make_model  # noqa: E402
from train_nested_cv import RESULTS, TARGETS, load  # noqa: E402

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
ALL = MODELS + ["LightGBM-Directional"]
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})
GREY, BLUE, ORANGE = "#9aa0a6", "#1f6feb", "#e8710a"
rng = np.random.default_rng(0)


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close()


# ============================================================================ EDA
def eda(df):
    out = {}
    # 1) like rate vs partner attractiveness, per direction
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, (who, partner, y, lab) in zip(ax, [("men", "w_attract", "y_m_likes", "Man likes woman"),
                                              ("women", "m_attract", "y_w_likes", "Woman likes man")]):
        b = pd.qcut(df[partner], 5)
        r = df.groupby(b, observed=True)[y].agg(["mean", "count"])
        a.bar(range(5), r["mean"], color=BLUE if who == "men" else ORANGE)
        a.set_xticks(range(5), [f"Q{i+1}\n{iv.left:.1f}-{iv.right:.1f}" for i, iv in enumerate(r.index)], fontsize=8)
        a.set_title(f"{lab}: rate by partner attractiveness quintile")
        a.set_ylabel("share of dates")
        a.set_ylim(0, 1)
        for i, v in enumerate(r["mean"]):
            a.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=8)
        out[f"like_rate_by_partner_attr_{who}"] = r["mean"].round(3).tolist()
    save("eda_like_vs_partner_attractiveness")

    # 2) own attractiveness vs selectivity
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, (own, y, lab, c) in zip(ax, [("m_attract", "y_m_likes", "Men", BLUE),
                                        ("w_attract", "y_w_likes", "Women", ORANGE)]):
        b = pd.qcut(df[own], 5)
        r = df.groupby(b, observed=True)[y].mean()
        a.plot(range(5), r.values, "o-", color=c)
        a.set_xticks(range(5), [f"Q{i+1}" for i in range(5)])
        a.set_title(f"{lab}: share of partners liked vs own attractiveness")
        a.set_ylim(0, 1)
        out[f"selectivity_by_own_attr_{lab.lower()}"] = r.round(3).tolist()
    save("eda_selectivity_vs_own_attractiveness")

    # 3) base rates
    out["n_dates"], out["n_events"] = len(df), df.event_id.nunique()
    out["n_men"], out["n_women"] = df.male_user_id.nunique(), df.female_user_id.nunique()
    out["rates"] = df[TARGETS].mean().round(4).to_dict()
    out["event_mutual_rate_range"] = df.groupby("event_id").y_mutual.mean().agg(["min", "median", "max"]).round(3).tolist()
    out["missing_questionnaire_share_dates"] = {"men": float(df.m_questionnaire_missing.mean()),
                                                "women": float(df.w_questionnaire_missing.mean())}
    # univariate AUCs (single feature, sign-agnostic)
    feats = [c for c in df.columns if c.startswith(("m_", "w_", "absdiff_")) or c in
             ("age_diff", "height_diff", "attract_diff", "attract_sum")]
    uni = {}
    for t in TARGETS:
        s = {}
        for f in feats:
            x = df[f].fillna(df[f].median())
            if x.nunique() > 1:
                a = roc_auc_score(df[t], x)
                s[f] = max(a, 1 - a)
        uni[t] = pd.Series(s).sort_values(ascending=False).head(8).round(3).to_dict()
    out["univariate_auc_top"] = uni
    # attractiveness pairing heatmap: mutual rate by (man quintile x woman quintile)
    piv = df.assign(mq=pd.qcut(df.m_attract, 3, labels=["low", "mid", "high"]),
                    wq=pd.qcut(df.w_attract, 3, labels=["low", "mid", "high"])) \
        .pivot_table(index="mq", columns="wq", values="y_mutual", aggfunc="mean", observed=True)
    fig, a = plt.subplots(figsize=(4.6, 3.8))
    im = a.imshow(piv.values, cmap="Blues", vmin=0, vmax=0.3)
    a.set_xticks(range(3), piv.columns)
    a.set_yticks(range(3), piv.index)
    a.set_xlabel("woman attractiveness tercile")
    a.set_ylabel("man attractiveness tercile")
    a.grid(False)
    for i in range(3):
        for j in range(3):
            a.text(j, i, f"{piv.values[i, j]:.0%}", ha="center", va="center")
    a.set_title("Mutual-match rate")
    plt.colorbar(im, ax=a, fraction=0.046)
    save("eda_mutual_rate_attractiveness_grid")
    out["mutual_grid"] = piv.round(3).to_dict()
    return out


# ============================================================================ comparison
def cluster_boot_auc(oof, events, W):
    """Event-cluster bootstrap of the mean *within-fold* ROC-AUC / AP.
    Resampling whole events keeps the dependence between dates of one event;
    AUC is computed inside each outer test fold (as in the training log) and averaged,
    so fold-to-fold differences in calibration do not leak into the ranking metric."""
    o = oof.assign(ev=events[oof.row.values]).reset_index(drop=True)
    parts = [(g.ev.values, g.y.values, g.p.values) for _, g in o.groupby(["repeat", "fold"])]
    ev_ids = np.unique(events)
    aucs, aps = [], []
    for w in W:  # W[b, e] = how often event e is drawn in bootstrap sample b (shared by all models)
        a_s, p_s = [], []
        for ev, y, p in parts:
            k = w[np.searchsorted(ev_ids, ev)]
            yy, pp = np.repeat(y, k), np.repeat(p, k)
            if 0 < yy.sum() < len(yy):
                a_s.append(roc_auc_score(yy, pp))
                p_s.append(average_precision_score(yy, pp))
        aucs.append(np.mean(a_s))
        aps.append(np.mean(p_s))
    return np.array(aucs), np.array(aps)


def compare(df):
    events = df.event_id.values
    rows, boots, folds = [], {}, []
    ev_ids = np.unique(events)
    W = np.random.default_rng(1).multinomial(len(ev_ids), np.ones(len(ev_ids)) / len(ev_ids), size=1000)
    for m in ALL:
        f = RESULTS / f"oof_{m}.csv"
        if not f.exists():
            continue
        oof = pd.read_csv(f)
        fo = pd.read_csv(RESULTS / f"folds_{m}.csv")
        folds.append(fo)
        pooled = fo[["roc_auc", "avg_precision", "brier", "log_loss", "prec_top10", "lift_top10"]].mean()
        a, ap = cluster_boot_auc(oof, events, W)
        boots[m] = a
        rows.append({"model": m, **pooled.to_dict(),
                     "auc_lo": np.percentile(a, 2.5), "auc_hi": np.percentile(a, 97.5),
                     "ap_lo": np.percentile(ap, 2.5), "ap_hi": np.percentile(ap, 97.5),
                     "fold_auc_mean": fo.roc_auc.mean(), "fold_auc_std": fo.roc_auc.std(),
                     "train_auc": fo.train_auc.mean(), "inner_cv_auc": fo.inner_cv_auc.mean(),
                     "fit_seconds_per_fold": fo.seconds.mean()})
    res = pd.DataFrame(rows)
    ref = "LogReg"
    # same rng stream per model is not shared, so compare via paired re-run on identical resamples
    # paired bootstrap: share of identical event-resamples in which model beats the reference
    res["p_better_than_logreg"] = [np.mean(boots[m] > boots["LogReg"]) for m in res.model]
    res["p_better_than_attract"] = [np.mean(boots[m] > boots["LogReg-Attract"]) for m in res.model]
    res.to_csv(RESULTS / "model_comparison.csv", index=False)
    fold_all = pd.concat(folds)
    fold_all.to_csv(RESULTS / "all_folds.csv", index=False)

    # figure: AUC with CI + train AUC
    r = res.set_index("model").loc[[m for m in ALL if m in set(res.model)]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    y = np.arange(len(r))[::-1]
    ax[0].errorbar(r.roc_auc, y, xerr=[r.roc_auc - r.auc_lo, r.auc_hi - r.roc_auc], fmt="o",
                   color=BLUE, ecolor=GREY, capsize=3, label="test (nested CV)")
    ax[0].scatter(r.train_auc, y, marker="x", color=ORANGE, label="train")
    ax[0].axvline(0.5, color="k", lw=0.8, ls="--")
    ax[0].set_yticks(y, r.index)
    ax[0].set_xlabel("ROC-AUC  (95% event-bootstrap CI)")
    ax[0].legend(loc="lower right", fontsize=8)
    ax[0].set_title("Mutual match: ROC-AUC")
    ax[1].errorbar(r.avg_precision, y, xerr=[r.avg_precision - r.ap_lo, r.ap_hi - r.avg_precision],
                   fmt="o", color=BLUE, ecolor=GREY, capsize=3)
    ax[1].axvline(df.y_mutual.mean(), color="k", lw=0.8, ls="--", label="base rate")
    ax[1].set_yticks(y, [])
    ax[1].set_xlabel("Average precision (PR-AUC)")
    ax[1].legend(fontsize=8)
    ax[1].set_title("Mutual match: PR-AUC")
    save("model_comparison")

    # ROC curves + calibration
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for m, c in [("LogReg-Attract", GREY), ("LogReg", "#5e35b1"), ("RandomForest", "#2e7d32"),
                 ("LightGBM", BLUE), ("MultiTask-NN", ORANGE)]:
        f = RESULTS / f"oof_{m}.csv"
        if not f.exists():
            continue
        o = pd.read_csv(f)
        o = o[o.repeat == 0]
        fpr, tpr, _ = roc_curve(o.y, o.p)
        ax[0].plot(fpr, tpr, color=c, label=f"{m} ({roc_auc_score(o.y, o.p):.3f})")
        pt, pp = calibration_curve(o.y, o.p, n_bins=8, strategy="quantile")
        ax[1].plot(pp, pt, "o-", color=c, label=m, ms=3)
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set_xlabel("False positive rate")
    ax[0].set_ylabel("True positive rate")
    ax[0].legend(fontsize=7)
    ax[0].set_title("ROC (repeat 0, pooled outer folds)")
    ax[1].plot([0, .4], [0, .4], "k--", lw=0.8)
    ax[1].set_xlabel("predicted P(mutual)")
    ax[1].set_ylabel("observed rate")
    ax[1].set_title("Calibration")
    ax[1].legend(fontsize=7)
    save("roc_calibration")

    # directional performance (how predictable is each side)
    d = {}
    for m in ["MultiTask-NN", "LightGBM-Directional"]:
        f = RESULTS / f"oof_{m}.csv"
        if f.exists():
            o = pd.read_csv(f)
            o = o.merge(df[["y_m_likes", "y_w_likes"]], left_on="row", right_index=True)
            d[m] = {"auc_man_likes": np.mean([roc_auc_score(g.y_m_likes, g.p_m_likes) for _, g in o.groupby("repeat")]),
                    "auc_woman_likes": np.mean([roc_auc_score(g.y_w_likes, g.p_w_likes) for _, g in o.groupby("repeat")])}
    return res, d


# ============================================================================ dynamics
def best_params(m):
    fo = pd.read_csv(RESULTS / f"folds_{m}.csv")
    return json.loads(fo.sort_values("inner_cv_auc", ascending=False).params.iloc[0])


def dynamics(df, feats, groups):
    X, Y, g = df[feats], df[TARGETS], df.event_id.values
    cv = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, Y.y_mutual, g))
    out = {}

    # (a) learning curves: AUC vs fraction of training events
    fracs = [0.2, 0.4, 0.6, 0.8, 1.0]
    lc = []
    cached = RESULTS / "learning_curves.csv"
    for m in ([] if cached.exists() else ["LogReg", "RandomForest", "LightGBM", "MLP", "MultiTask-NN"]):
        p = best_params(m)
        for frac in fracs:
            for k, (tr, te) in enumerate(cv):
                evs = np.unique(g[tr])
                keep = rng.choice(evs, max(2, int(round(frac * len(evs)))), replace=False)
                tr2 = tr[np.isin(g[tr], keep)]
                mod = make_model(m, p, groups, k).fit(X.iloc[tr2], Y.iloc[tr2])
                lc.append({"model": m, "frac": frac, "n_train": len(tr2), "fold": k,
                           "train_auc": roc_auc_score(Y.y_mutual.iloc[tr2], mod.predict_proba(X.iloc[tr2])),
                           "test_auc": roc_auc_score(Y.y_mutual.iloc[te], mod.predict_proba(X.iloc[te]))})
    lc = pd.read_csv(cached) if cached.exists() else pd.DataFrame(lc)
    lc.to_csv(RESULTS / "learning_curves.csv", index=False)
    s = lc.groupby(["model", "frac"]).agg(n=("n_train", "mean"), tr=("train_auc", "mean"),
                                          te=("test_auc", "mean"), se=("test_auc", "sem")).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    cols = {"LogReg": "#5e35b1", "RandomForest": "#2e7d32", "LightGBM": BLUE, "MLP": "#c62828", "MultiTask-NN": ORANGE}
    for m, d in s.groupby("model"):
        ax.errorbar(d.n, d.te, yerr=d.se, fmt="o-", color=cols[m], label=f"{m} test", capsize=2)
        ax.plot(d.n, d.tr, ":", color=cols[m], alpha=0.6)
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    ax.set_xlabel("training dates")
    ax.set_ylabel("ROC-AUC (dotted = train)")
    ax.set_title("Learning curves: more data helps only slowly")
    ax.legend(fontsize=8)
    save("learning_curves")
    out["learning_curves"] = s.round(3).to_dict(orient="records")

    # (b) neural-net training curves (one outer split, inner validation by event)
    tr, te = cv[0]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, m in zip(ax, ["MLP", "MultiTask-NN"]):
        p = {**best_params(m), "epochs": 400, "patience": 400}
        mod = make_model(m, p, groups, 0).fit(X.iloc[tr], Y.iloc[tr], X.iloc[te], Y.iloc[te])
        h = mod.history
        a.plot(h.epoch, h.train_loss, color=GREY, label="train loss (objective)")
        a2 = a.twinx()
        a2.plot(h.epoch, h.val_logloss, color=BLUE, label="held-out log-loss P(mutual)")
        a2.axvline(best_params(m)["epochs"], color=ORANGE, ls="--", lw=1, label="epochs chosen by inner CV")
        a.set_xlabel("epoch")
        a.set_title(m)
        h1, l1 = a.get_legend_handles_labels()
        h2, l2 = a2.get_legend_handles_labels()
        a.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=7, framealpha=0.95)
        a.set_ylabel("training objective")
        a2.set_ylabel("held-out log-loss")
        a2.grid(False)
        out[f"nn_curve_{m}"] = {"min_val_epoch": int(h.val_logloss.idxmin() + 1),
                                "min_val": float(h.val_logloss.min()), "final_val": float(h.val_logloss.iloc[-1])}
    save("nn_training_curves")

    # (c) complexity curves: tree depth & boosting rounds
    cc = []
    for depth in [1, 2, 3, 4, 6, 8, 12, 16, None]:
        for k, (tr, te) in enumerate(cv):
            mod = make_model("DecisionTree", {"max_depth": depth, "min_samples_leaf": 1}, groups).fit(X.iloc[tr], Y.iloc[tr])
            cc.append({"depth": depth or 30, "train": roc_auc_score(Y.y_mutual.iloc[tr], mod.predict_proba(X.iloc[tr])),
                       "test": roc_auc_score(Y.y_mutual.iloc[te], mod.predict_proba(X.iloc[te]))})
    cc = pd.DataFrame(cc).groupby("depth").mean().reset_index()
    from lightgbm import LGBMClassifier
    p = best_params("LightGBM")
    p.update({"n_estimators": 1500})
    curves = []
    for k, (tr, te) in enumerate(cv):
        mod = LGBMClassifier(**p, verbose=-1).fit(X.iloc[tr], Y.y_mutual.iloc[tr],
                                                   eval_set=[(X.iloc[tr], Y.y_mutual.iloc[tr]), (X.iloc[te], Y.y_mutual.iloc[te])],
                                                   eval_names=["train", "test"], eval_metric="auc")
        r = mod.evals_result_
        curves.append(pd.DataFrame({"train": r["train"]["auc"], "test": r["test"]["auc"]}))
    bc = sum(curves) / len(curves)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(cc.depth.astype(str), cc.train, "o:", color=GREY, label="train")
    ax[0].plot(cc.depth.astype(str), cc.test, "o-", color=BLUE, label="test")
    ax[0].set_xlabel("max_depth (30 = unlimited)")
    ax[0].set_title("Decision tree: depth vs AUC")
    ax[0].legend()
    ax[1].plot(bc.index + 1, bc.train, color=GREY, label="train")
    ax[1].plot(bc.index + 1, bc.test, color=BLUE, label="test")
    ax[1].axvline(best_params("LightGBM")["n_estimators"], color=ORANGE, ls="--", label="rounds chosen by inner CV")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("boosting rounds")
    ax[1].set_title(f"LightGBM (lr={p['learning_rate']:.3f}): rounds vs AUC")
    ax[1].legend(fontsize=8)
    save("complexity_curves")
    out["tree_depth_curve"] = cc.round(3).to_dict(orient="records")
    out["lgbm_rounds"] = {"best_test_round": int(bc.test.idxmax() + 1), "best_test_auc": float(bc.test.max()),
                          "chosen": best_params("LightGBM")["n_estimators"],
                          "auc_at_chosen": float(bc.test.iloc[min(len(bc) - 1, best_params("LightGBM")["n_estimators"] - 1)]),
                          "auc_at_1500": float(bc.test.iloc[-1])}
    return out


# ============================================================================ importance
def importance(df, feats, groups):
    X, Y, g = df[feats], df[TARGETS], df.event_id.values
    cv = list(StratifiedGroupKFold(5, shuffle=True, random_state=7).split(X, Y.y_mutual, g))
    imp = []
    for m in ["LightGBM", "LogReg"]:
        p = best_params(m)
        for k, (tr, te) in enumerate(cv):
            mod = make_model(m, p, groups, k).fit(X.iloc[tr], Y.iloc[tr])
            r = permutation_importance(mod.est, X.iloc[te], Y.y_mutual.iloc[te], scoring="roc_auc",
                                       n_repeats=10, random_state=k, n_jobs=8)
            imp.append(pd.DataFrame({"model": m, "feature": feats, "fold": k, "drop": r.importances_mean}))
    imp = pd.concat(imp).groupby(["model", "feature"]).drop.mean().reset_index()
    imp.to_csv(RESULTS / "permutation_importance.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    top = {}
    for a, m in zip(ax, ["LightGBM", "LogReg"]):
        d = imp[imp.model == m].sort_values("drop", ascending=False).head(12)[::-1]
        a.barh(d.feature, d["drop"], color=[BLUE if v > 0 else GREY for v in d["drop"]])
        a.set_xlabel("mean drop in test ROC-AUC when permuted")
        a.set_title(f"{m}: permutation importance")
        top[m] = d[::-1].set_index("feature")["drop"].round(4).to_dict()
    save("feature_importance")

    # directional importance: what drives each side's "like"
    from lightgbm import LGBMClassifier
    dirimp = {}
    for t in ["y_m_likes", "y_w_likes"]:
        rs = []
        for k, (tr, te) in enumerate(cv):
            mod = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=20,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.5, verbose=-1)
            mod.fit(X.iloc[tr], Y[t].iloc[tr])
            r = permutation_importance(mod, X.iloc[te], Y[t].iloc[te], scoring="roc_auc", n_repeats=5,
                                       random_state=k, n_jobs=8)
            rs.append(r.importances_mean)
        dirimp[t] = pd.Series(np.mean(rs, 0), index=feats).sort_values(ascending=False).head(8).round(4).to_dict()
    return top, dirimp


def main():
    df, feats, groups = load()
    summary = {"eda": eda(df)}
    res, d = compare(df)
    summary["directional_auc"] = d
    print(res[["model", "roc_auc", "auc_lo", "auc_hi", "avg_precision", "lift_top10", "brier",
               "train_auc", "p_better_than_attract"]].round(3).to_string())
    summary["dynamics"] = dynamics(df, feats, groups)
    summary["importance"], summary["directional_importance"] = importance(df, feats, groups)
    summary["best_params"] = {m: best_params(m) for m in ALL if (RESULTS / f"folds_{m}.csv").exists() and m != "Prior"}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
