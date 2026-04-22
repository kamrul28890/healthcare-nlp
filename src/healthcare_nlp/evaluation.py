from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _score_values(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        if isinstance(scores, list):
            scores = np.array(scores)
        return scores
    return None


def evaluate_binary_model(model, x, y_true) -> dict[str, Any]:
    y_pred = model.predict(x)
    scores = _score_values(model, x)

    output = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }

    if scores is not None:
        output["roc_auc"] = float(roc_auc_score(y_true, scores))
        output["pr_auc"] = float(average_precision_score(y_true, scores))

    return output
