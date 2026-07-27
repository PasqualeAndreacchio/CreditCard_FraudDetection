"""
Centralized metrics computation for Credit Card Fraud Detection.

Provides a single function ``compute_all_metrics()`` that computes every
metric needed for a comprehensive fraud-detection evaluation report.
Both evaluators (reconstruction and classification) call this, ensuring
consistent and comparable results across all model types.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score,
    balanced_accuracy_score,
    precision_recall_curve,
)

logger = logging.getLogger(__name__)


# ─── Core Metrics ───────────────────────────────────────────────────────────

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute a comprehensive set of binary classification metrics.

    Args:
        y_true: Ground-truth binary labels (0 = normal, 1 = fraud), shape ``(N,)``.
        y_pred: Binary predictions (0/1), shape ``(N,)``.
        y_scores: Continuous anomaly scores or fraud probabilities, shape ``(N,)``.
            Higher values indicate higher likelihood of fraud.
        threshold: Decision threshold used to produce ``y_pred``.

    Returns:
        Dictionary with all computed metrics, ready for JSON serialization
        (numpy arrays converted to lists where necessary).
    """
    y_true = np.asarray(y_true).flatten().astype(int)
    y_pred = np.asarray(y_pred).flatten().astype(int)
    y_scores = np.asarray(y_scores, dtype=np.float64).flatten()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # ── Standard metrics ────────────────────────────────────────────
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # ── Extended metrics ────────────────────────────────────────────
    f2 = float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    mcc = float(matthews_corrcoef(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

    # ── Ranking metrics (threshold-independent) ─────────────────────
    try:
        auprc = float(average_precision_score(y_true, y_scores))
    except ValueError:
        auprc = 0.0
    try:
        auroc = float(roc_auc_score(y_true, y_scores))
    except ValueError:
        auroc = 0.0

    # ── Report ──────────────────────────────────────────────────────
    report = classification_report(
        y_true, y_pred, labels=[0, 1],
        target_names=["Normal", "Fraud"], zero_division=0,
    )

    metrics = {
        "threshold": threshold,
        # Standard
        "precision": precision,
        "recall": recall,
        "f1": f1,
        # Extended
        "f2": f2,
        "specificity": specificity,
        "mcc": mcc,
        "cohen_kappa": kappa,
        "balanced_accuracy": balanced_acc,
        # Ranking
        "auprc": auprc,
        "auroc": auroc,
        # Confusion matrix
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        # Report
        "classification_report": report,
    }

    return metrics


def log_metrics(metrics: dict[str, Any]) -> None:
    """Pretty-print a metrics dictionary to the logger.

    Args:
        metrics: Dictionary returned by ``compute_all_metrics()``.
    """
    logger.info("=" * 55)
    logger.info("  EVALUATION RESULTS")
    logger.info("=" * 55)
    logger.info("  Threshold         : %.6f", metrics["threshold"])
    logger.info("  ──────────────────────────────────────")
    logger.info("  Precision          : %.4f", metrics["precision"])
    logger.info("  Recall (Sensitivity): %.4f", metrics["recall"])
    logger.info("  Specificity        : %.4f", metrics["specificity"])
    logger.info("  F1-score           : %.4f", metrics["f1"])
    logger.info("  F2-score           : %.4f", metrics["f2"])
    logger.info("  MCC                : %.4f", metrics["mcc"])
    logger.info("  Cohen's Kappa      : %.4f", metrics["cohen_kappa"])
    logger.info("  Balanced Accuracy  : %.4f", metrics["balanced_accuracy"])
    logger.info("  ──────────────────────────────────────")
    logger.info("  AUPRC              : %.4f", metrics["auprc"])
    logger.info("  AUROC              : %.4f", metrics["auroc"])
    logger.info("  ──────────────────────────────────────")
    logger.info("  TP=%d  FP=%d  FN=%d  TN=%d", metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"])
    logger.info("\n%s", metrics["classification_report"])
    logger.info("=" * 55)


# ─── Threshold Utilities ────────────────────────────────────────────────────

def find_f1_optimal_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Search for the threshold that maximises F1-score.

    Uses the precision-recall curve from scikit-learn to enumerate
    candidate thresholds efficiently.

    Args:
        scores: Per-sample anomaly scores or predicted probabilities.
        labels: Ground-truth binary labels (0=normal, 1=fraud).

    Returns:
        Optimal threshold value.
    """
    labels = (np.asarray(labels).flatten() > 0).astype(np.int32)
    scores = np.asarray(scores, dtype=np.float64).flatten()

    unique = np.unique(labels)
    if len(unique) < 2:
        logger.warning(
            "find_f1_optimal_threshold: labels contain only one class (%s). "
            "Returning threshold=0.0 with F1=0.",
            unique,
        )
        return 0.0

    precisions, recalls, thresholds = precision_recall_curve(labels, scores, pos_label=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = 2 * precisions * recalls / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores)

    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[min(best_idx, len(thresholds) - 1)])
    logger.info(
        "F1-optimal search: best_f1=%.4f at threshold=%.6f",
        f1_scores[best_idx],
        best_threshold,
    )
    return best_threshold
