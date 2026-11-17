"""Model zoo: from a trivial prior to a multi-task two-tower neural network.

Every model exposes the same interface:
    m = make_model(name, params, groups)   # groups = feature column groups
    m.fit(X_df, Y_df)                      # Y_df has y_mutual, y_m_likes, y_w_likes
    p = m.predict_proba(X_df)              # P(mutual match)
and a search space `suggest(name, trial)` used by Optuna.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

torch.set_num_threads(1)

MODELS = ["Prior", "LogReg-Attract", "LogReg", "kNN", "DecisionTree", "RandomForest",
          "LightGBM", "MLP", "MultiTask-NN"]


def dense_prep():
    """Median imputation + missing-indicators + standardisation (for non-tree models)."""
    return [SimpleImputer(strategy="median", add_indicator=True), StandardScaler()]


# --------------------------------------------------------------------------- sklearn wrappers
class SkModel:
    def __init__(self, est, cols=None):
        self.est, self.cols = est, cols

    def _x(self, X):
        return X[self.cols] if self.cols is not None else X

    def fit(self, X, Y):
        self.est.fit(self._x(X), Y["y_mutual"].values)
        return self

    def predict_proba(self, X):
        return self.est.predict_proba(self._x(X))[:, 1]


class Prior:
    def fit(self, X, Y):
        self.p = Y["y_mutual"].mean()
        return self

    def predict_proba(self, X):
        return np.full(len(X), self.p)


class DirectionalProduct:
    """Ablation: two LightGBMs for each direction; P(mutual) = P(m likes) * P(w likes)."""

    def __init__(self, params):
        self.mm = LGBMClassifier(**params, verbose=-1, n_jobs=1)
        self.mw = LGBMClassifier(**params, verbose=-1, n_jobs=1)

    def fit(self, X, Y):
        self.mm.fit(X, Y["y_m_likes"].values)
        self.mw.fit(X, Y["y_w_likes"].values)
        return self

    def predict_proba(self, X):
        return self.mm.predict_proba(X)[:, 1] * self.mw.predict_proba(X)[:, 1]


# --------------------------------------------------------------------------- neural nets
def mlp(d_in, width, depth, dropout, d_out=1):
    layers, d = [], d_in
    for i in range(depth):
        w = max(8, int(width / (2 ** i)))  # funnel-shaped
        layers += [nn.Linear(d, w), nn.BatchNorm1d(w), nn.ReLU(), nn.Dropout(dropout)]
        d = w
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class TwoTowerNet(nn.Module):
    """Man tower + woman tower (separate weights, genders differ) -> embeddings;
    a joint head sees [emb_man, emb_woman, emb_man*emb_woman, pair features]
    and outputs two logits: P(man likes woman), P(woman likes man)."""

    def __init__(self, d_m, d_w, d_p, emb, width, dropout):
        super().__init__()
        self.tm = nn.Sequential(nn.Linear(d_m, width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, emb))
        self.tw = nn.Sequential(nn.Linear(d_w, width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, emb))
        self.head = nn.Sequential(nn.Linear(3 * emb + d_p, width), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(width, 2))

    def forward(self, xm, xw, xp):
        em, ew = self.tm(xm), self.tw(xw)
        return self.head(torch.cat([em, ew, em * ew, xp], 1))


class TorchModel:
    """Shared training loop. kind='mlp' -> single-task MLP on P(mutual).
    kind='mt' -> two-tower multi-task net, P(mutual)=P(m)*P(w) and loss
    = BCE(m) + BCE(w) + lam * BCE(P(m)P(w), mutual)."""

    def __init__(self, kind, params, groups, seed=0):
        self.kind, self.p, self.groups, self.seed = kind, dict(params), groups, seed
        self.history = None

    def _prep(self, X, fit=False):
        if fit:
            self.prep = make_pipeline(*dense_prep()).fit(X)
            names = self.prep[0].get_feature_names_out(X.columns)
            # map every transformed column (incl. missing indicators) back to its group
            self.idx = {}
            for g, cols in self.groups.items():
                s = set(cols)
                self.idx[g] = [i for i, n in enumerate(names)
                               if n in s or (n.startswith("missingindicator_") and n[17:] in s)]
        return torch.tensor(self.prep.transform(X), dtype=torch.float32)

    def _build(self, d):
        p = self.p
        if self.kind == "mlp":
            return mlp(d, p["width"], p["depth"], p["dropout"])
        return TwoTowerNet(len(self.idx["man"]), len(self.idx["woman"]), len(self.idx["pair"]),
                           p["emb"], p["width"], p["dropout"])

    def _forward(self, net, x):
        if self.kind == "mlp":
            return net(x)[:, 0]
        return net(x[:, self.idx["man"]], x[:, self.idx["woman"]], x[:, self.idx["pair"]])

    def _loss(self, out, y):
        bce = nn.functional.binary_cross_entropy_with_logits
        if self.kind == "mlp":
            return bce(out, y[:, 0], pos_weight=torch.tensor(self.p.get("pos_weight", 1.0)))
        pm, pw = torch.sigmoid(out[:, 0]), torch.sigmoid(out[:, 1])
        pmut = (pm * pw).clamp(1e-6, 1 - 1e-6)
        return (bce(out[:, 0], y[:, 1]) + bce(out[:, 1], y[:, 2])
                + self.p["lam"] * nn.functional.binary_cross_entropy(pmut, y[:, 0]))

    def _proba(self, net, x):
        net.eval()
        with torch.no_grad():
            out = self._forward(net, x)
            if self.kind == "mlp":
                return torch.sigmoid(out).numpy()
            return (torch.sigmoid(out[:, 0]) * torch.sigmoid(out[:, 1])).numpy()

    def fit(self, X, Y, X_val=None, Y_val=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        x = self._prep(X, fit=True)
        y = torch.tensor(Y[["y_mutual", "y_m_likes", "y_w_likes"]].values, dtype=torch.float32)
        net = self._build(x.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=self.p["lr"], weight_decay=self.p["wd"])
        bs, n = self.p.get("batch", 128), len(x)
        epochs = self.p.get("epochs", 300)
        xv = self._prep(X_val) if X_val is not None else None
        hist, best, best_state, wait = [], np.inf, None, 0
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, bs):
                b = perm[i:i + bs]
                if len(b) < 2:
                    continue
                opt.zero_grad()
                loss = self._loss(self._forward(net, x[b]), y[b])
                loss.backward()
                opt.step()
                tot += loss.item() * len(b)
            rec = {"epoch": ep + 1, "train_loss": tot / n}
            if xv is not None:  # early stopping on validation log-loss of P(mutual)
                pv = np.clip(self._proba(net, xv), 1e-6, 1 - 1e-6)
                yv = Y_val["y_mutual"].values
                vl = -np.mean(yv * np.log(pv) + (1 - yv) * np.log(1 - pv))
                rec["val_logloss"] = vl
                if vl < best - 1e-5:
                    best, wait, self.best_epoch = vl, 0, ep + 1
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    wait += 1
                    if wait >= self.p.get("patience", 25):
                        hist.append(rec)
                        break
            hist.append(rec)
        if best_state is not None:
            net.load_state_dict(best_state)
        self.net, self.history = net, pd.DataFrame(hist)
        return self

    def predict_proba(self, X):
        return self._proba(self.net, self._prep(X))

    def predict_directional(self, X):
        assert self.kind == "mt"
        self.net.eval()
        with torch.no_grad():
            out = self._forward(self.net, self._prep(X))
        return torch.sigmoid(out[:, 0]).numpy(), torch.sigmoid(out[:, 1]).numpy()


# --------------------------------------------------------------------------- factory
def make_model(name, params, groups, seed=0):
    p = dict(params)
    if name == "Prior":
        return Prior()
    if name == "LogReg-Attract":
        return SkModel(make_pipeline(*dense_prep(), LogisticRegression(C=p["C"], max_iter=2000)),
                       cols=["m_attract", "w_attract"])
    if name == "LogReg":
        return SkModel(make_pipeline(*dense_prep(), LogisticRegression(
            C=p["C"], l1_ratio=p["l1_ratio"], solver="saga", max_iter=5000,
            class_weight=p["class_weight"])))
    if name == "kNN":
        return SkModel(make_pipeline(*dense_prep(), KNeighborsClassifier(
            n_neighbors=p["k"], weights=p["weights"], p=p["p"])))
    if name == "DecisionTree":
        return SkModel(DecisionTreeClassifier(random_state=seed, **p))
    if name == "RandomForest":
        return SkModel(RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1, **p))
    if name == "LightGBM":
        return SkModel(LGBMClassifier(random_state=seed, verbose=-1, n_jobs=1, **p))
    if name == "LightGBM-Directional":
        return DirectionalProduct({**p, "random_state": seed})
    if name == "MLP":
        return TorchModel("mlp", p, groups, seed)
    if name == "MultiTask-NN":
        return TorchModel("mt", p, groups, seed)
    raise ValueError(name)


def suggest(name, t):
    """Optuna search spaces."""
    if name == "LogReg-Attract":
        return {"C": t.suggest_float("C", 1e-3, 10, log=True)}
    if name == "LogReg":
        return {"C": t.suggest_float("C", 1e-3, 10, log=True),
                "l1_ratio": t.suggest_float("l1_ratio", 0.0, 1.0),
                "class_weight": t.suggest_categorical("class_weight", [None, "balanced"])}
    if name == "kNN":
        return {"k": t.suggest_int("k", 5, 300, log=True),
                "weights": t.suggest_categorical("weights", ["uniform", "distance"]),
                "p": t.suggest_categorical("p", [1, 2])}
    if name == "DecisionTree":
        return {"max_depth": t.suggest_int("max_depth", 2, 12),
                "min_samples_leaf": t.suggest_int("min_samples_leaf", 5, 200, log=True),
                "criterion": t.suggest_categorical("criterion", ["gini", "entropy"]),
                "class_weight": t.suggest_categorical("class_weight", [None, "balanced"])}
    if name == "RandomForest":
        return {"max_depth": t.suggest_categorical("max_depth", [3, 5, 8, 12, None]),
                "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 50, log=True),
                "max_features": t.suggest_categorical("max_features", ["sqrt", 0.3, 0.6]),
                "class_weight": t.suggest_categorical("class_weight", [None, "balanced_subsample"])}
    if name in ("LightGBM", "LightGBM-Directional"):
        return {"n_estimators": t.suggest_int("n_estimators", 50, 800, log=True),
                "learning_rate": t.suggest_float("learning_rate", 0.005, 0.2, log=True),
                "num_leaves": t.suggest_int("num_leaves", 3, 31, log=True),
                "max_depth": t.suggest_int("max_depth", 2, 8),
                "min_child_samples": t.suggest_int("min_child_samples", 5, 100, log=True),
                "subsample": t.suggest_float("subsample", 0.5, 1.0), "subsample_freq": 1,
                "colsample_bytree": t.suggest_float("colsample_bytree", 0.2, 1.0),
                "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 30, log=True)}
    if name == "MLP":
        return {"width": t.suggest_categorical("width", [16, 32, 64, 128, 256]),
                "depth": t.suggest_int("depth", 1, 3),
                "dropout": t.suggest_float("dropout", 0.0, 0.6),
                "lr": t.suggest_float("lr", 1e-4, 1e-2, log=True),
                "wd": t.suggest_float("wd", 1e-5, 1e-1, log=True),
                "batch": t.suggest_categorical("batch", [64, 128, 256]),
                "pos_weight": t.suggest_categorical("pos_weight", [1.0, 3.0, 6.3])}
    if name == "MultiTask-NN":
        return {"emb": t.suggest_categorical("emb", [4, 8, 16, 32]),
                "width": t.suggest_categorical("width", [16, 32, 64, 128]),
                "dropout": t.suggest_float("dropout", 0.0, 0.6),
                "lr": t.suggest_float("lr", 1e-4, 1e-2, log=True),
                "wd": t.suggest_float("wd", 1e-5, 1e-1, log=True),
                "batch": t.suggest_categorical("batch", [64, 128, 256]),
                "lam": t.suggest_float("lam", 0.0, 3.0)}
    return {}
