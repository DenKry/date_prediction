import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def precision_at(y, p, frac=0.10):
    """Share of true mutual matches among the top `frac` pairs ranked by the model."""
    y, p = np.asarray(y), np.asarray(p)
    k = max(1, int(round(frac * len(y))))
    thr = np.sort(p)[::-1][k - 1]
    above, tied = p > thr, p == thr
    # pairs tied at the cut-off are taken at random -> use their expected share
    need = k - above.sum()
    hits = y[above].sum() + need * y[tied].mean()
    return float(hits / k)


def all_metrics(y, p):
    y, p = np.asarray(y), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    base = y.mean()
    return {"roc_auc": roc_auc_score(y, p), "avg_precision": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p),
            "prec_top10": precision_at(y, p), "lift_top10": precision_at(y, p) / base}
