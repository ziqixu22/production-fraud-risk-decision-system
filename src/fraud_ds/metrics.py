from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss


def classification_metrics(y_true, prob) -> dict[str, float]:
    y = np.asarray(y_true)
    p = np.asarray(prob)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "fraud_rate": float(y.mean()),
    }
