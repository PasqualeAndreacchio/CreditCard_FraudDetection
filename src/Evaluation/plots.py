"""
Centralized plotting utilities for Credit Card Fraud Detection.

Every plot function follows the same signature pattern:
    - Input data arrays (scores, labels, etc.)
    - ``save_dir`` and ``filename`` for file output
    - Optional ``title`` override
    - Returns ``None`` (saves to disk and logs the path)

Both evaluators (reconstruction and classification) call these functions,
ensuring a consistent visual style across the entire report.
"""

from __future__ import annotations

import os
import logging
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


# Confusion matrix heatmap

def plot_confusion_matrix(
    cm: np.ndarray,
    save_dir: str,
    filename: str = "confusion_matrix.png",
    title: str = "Confusion Matrix",
) -> None:
    """Generate and save a confusion matrix heatmap.

    Args:
        cm: 2x2 confusion matrix (from sklearn.metrics.confusion_matrix).
        save_dir: Directory where the plot will be saved.
        filename: Output filename (default: ``confusion_matrix.png``).
        title: Plot title.
    """
    os.makedirs(save_dir, exist_ok=True)

    if cm.shape != (2, 2):
        cm_2x2 = np.zeros((2, 2), dtype=int)
        cm_2x2[:min(cm.shape[0], 2), :min(cm.shape[1], 2)] = cm
        cm = cm_2x2

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Fraud"])
    ax.set_yticklabels(["Normal", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=14, fontweight="bold",
            )
    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    logger.info("Saved: %s", path)


# Precision-Recall curve

def plot_precision_recall_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    save_dir: str,
    filename: str = "precision_recall_curve.png",
    title: str | None = None,
) -> None:
    """Generate and save a Precision-Recall curve.

    Args:
        labels: Ground-truth binary labels (0=normal, 1=fraud).
        scores: Per-sample anomaly scores or predicted probabilities.
        save_dir: Directory where the plot will be saved.
        filename: Output filename (default: ``precision_recall_curve.png``).
        title: Optional prefix for the plot title; AUPRC is always appended.
    """
    os.makedirs(save_dir, exist_ok=True)

    labels = (np.asarray(labels).flatten() > 0).astype(np.int32)
    scores = np.asarray(scores, dtype=np.float64).flatten()

    precisions, recalls, _ = precision_recall_curve(labels, scores, pos_label=1)
    auprc = average_precision_score(labels, scores, pos_label=1)
    plot_title = (
        f"{title} (AUPRC = {auprc:.4f})"
        if title
        else f"Precision-Recall Curve (AUPRC = {auprc:.4f})"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recalls, precisions, color="#9b59b6", linewidth=2)
    ax.fill_between(recalls, precisions, alpha=0.15, color="#9b59b6")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(plot_title)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    logger.info("Saved: %s", path)


# ROC curve

def plot_roc_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    save_dir: str,
    filename: str = "roc_curve.png",
    title: str | None = None,
) -> None:
    """Generate and save a ROC curve.

    Args:
        labels: Ground-truth binary labels (0=normal, 1=fraud).
        scores: Per-sample anomaly scores or predicted probabilities.
        save_dir: Directory where the plot will be saved.
        filename: Output filename.
        title: Optional title override.
    """
    os.makedirs(save_dir, exist_ok=True)

    labels = (np.asarray(labels).flatten() > 0).astype(np.int32)
    scores = np.asarray(scores, dtype=np.float64).flatten()

    fpr, tpr, _ = roc_curve(labels, scores)
    auc_val = roc_auc_score(labels, scores)

    plot_title = title or f"ROC Curve (AUROC = {auc_val:.4f})"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#e67e22", linewidth=2, label=f"AUROC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], color="#95a5a6", linewidth=1.5, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(plot_title)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    logger.info("Saved: %s", path)


# F1 score vs decision threshold

def plot_f1_vs_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    optimal_threshold: float,
    save_dir: str,
    filename: str = "f1_vs_threshold.png",
    title: str = "F1 Score vs Decision Threshold",
) -> None:
    """Plot F1 score as a function of decision threshold.

    Visualizes the trade-off and marks the optimal threshold with a
    vertical dashed line.

    Args:
        labels: Ground-truth binary labels.
        scores: Continuous anomaly scores or probabilities.
        optimal_threshold: The threshold selected for final predictions.
        save_dir: Directory where the plot will be saved.
        filename: Output filename.
        title: Plot title.
    """
    os.makedirs(save_dir, exist_ok=True)

    labels = (np.asarray(labels).flatten() > 0).astype(np.int32)
    scores = np.asarray(scores, dtype=np.float64).flatten()

    precisions, recalls, thresholds = precision_recall_curve(labels, scores, pos_label=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_values = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1])
    f1_values = np.nan_to_num(f1_values)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, f1_values, color="#2980b9", linewidth=2, label="F1 score")
    ax.axvline(
        optimal_threshold, color="#e74c3c", linestyle="--", linewidth=2,
        label=f"Optimal threshold = {optimal_threshold:.4f}",
    )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    logger.info("Saved: %s", path)


# Score distribution histogram (normal vs fraud)

def plot_score_distribution(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    save_dir: str,
    filename: str = "score_distribution.png",
    xlabel: str = "Anomaly Score",
    title: str = "Score Distribution",
    log_scale: bool = True,
    clip_percentile: float | None = 99.5,
) -> None:
    """Plot overlapping histograms of scores for normal vs fraud samples.

    Args:
        scores: Per-sample scores (reconstruction error or probability).
        labels: Ground-truth binary labels.
        threshold: Decision threshold shown as a vertical line.
        save_dir: Directory where the plot will be saved.
        filename: Output filename.
        xlabel: X-axis label (e.g. "Reconstruction Error (MSE)" or "Fraud Probability").
        title: Plot title.
        log_scale: If True, set Y-axis to log scale (essential for imbalanced data).
        clip_percentile: If provided, clips X-axis to this percentile of scores to avoid
            extreme outliers compressing the distribution.
    """
    os.makedirs(save_dir, exist_ok=True)

    normal_scores = scores[labels == 0]
    fraud_scores = scores[labels == 1]

    # Determine upper limit for X-axis ensuring threshold is ALWAYS visible
    if clip_percentile is not None and len(scores) > 0:
        p_max = float(np.percentile(scores, clip_percentile))
        x_max = max(p_max, threshold * 1.15)
    else:
        max_score = float(scores.max()) if len(scores) > 0 else 1.0
        x_max = max(1.0 if max_score <= 1.0 else max_score, threshold * 1.05)

    # Define 100 bins strictly within the visible range [0, x_max]
    bins = np.linspace(0, x_max, 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    if len(normal_scores) > 0:
        ax.hist(normal_scores, bins=bins, alpha=0.6, label="Normal", color="#3498db", density=True)
    if len(fraud_scores) > 0:
        ax.hist(fraud_scores, bins=bins, alpha=0.6, label="Fraud", color="#e74c3c", density=True)
    ax.axvline(
        threshold, color="#2ecc71", linestyle="--", linewidth=2.5,
        label=f"Threshold = {threshold:.4f}",
    )
    ax.set_xlim(left=0, right=x_max)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density (Log Scale)" if log_scale else "Density")
    ax.set_title(title)
    
    if log_scale:
        ax.set_yscale("log")

    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    logger.info("Saved: %s", path)


# Training history plots (loss, metrics, learning rate)

def plot_training_history(
    history: dict[str, list[float]],
    save_dir: str,
    filename: str = "training_history.png",
    title: str = "Training History",
) -> None:
    """Plot training and validation loss (and optionally F1/AUPRC) over epochs.

    Args:
        history: Dictionary with keys like ``train_loss``, ``val_loss``,
            ``val_f1``, ``val_auprc``, ``lr``.
        save_dir: Directory where the plot will be saved.
        filename: Output filename.
        title: Plot title.
    """
    os.makedirs(save_dir, exist_ok=True)

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_f1 = history.get("val_f1", [])
    val_auprc = history.get("val_auprc", [])
    lr_history = history.get("lr", [])

    # Determine how many subplots are needed
    has_val_metric = any(v > 0 for v in val_f1) or any(v > 0 for v in val_auprc)
    has_lr = len(lr_history) > 0
    n_plots = 1 + int(has_val_metric) + int(has_lr)

    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_loss) + 1)

    # Loss subplot
    ax = axes[0]
    ax.plot(epochs, train_loss, label="Train Loss", color="#3498db", linewidth=2)
    if val_loss:
        ax.plot(epochs, val_loss, label="Val Loss", color="#e74c3c", linewidth=2)
    ax.set_ylabel("Loss")
    ax.set_title(f"{title} — Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    idx = 1

    # Validation metric subplot (F1 or AUPRC)
    if has_val_metric:
        ax = axes[idx]
        if any(v > 0 for v in val_f1):
            ax.plot(epochs, val_f1, label="Val F1", color="#27ae60", linewidth=2)
        if any(v > 0 for v in val_auprc):
            ax.plot(epochs, val_auprc, label="Val AUPRC", color="#8e44ad", linewidth=2)
        ax.set_ylabel("Metric")
        ax.set_title(f"{title} — Validation Metrics")
        ax.legend()
        ax.grid(True, alpha=0.3)
        idx += 1

    # Learning rate subplot
    if has_lr:
        ax = axes[idx]
        ax.plot(epochs, lr_history, label="Learning Rate", color="#f39c12", linewidth=2)
        ax.set_ylabel("LR")
        ax.set_xlabel("Epoch")
        ax.set_title(f"{title} — Learning Rate")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    if not has_lr:
        axes[-1].set_xlabel("Epoch")

    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    logger.info("Saved: %s", path)


# Combined precision, recall and F1 vs threshold

def plot_precision_recall_f1_vs_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    optimal_threshold: float,
    save_dir: str,
    filename: str = "pr_f1_vs_threshold.png",
    title: str = "Precision, Recall & F1 vs Threshold",
) -> None:
    """Plot precision, recall, and F1 curves as a function of threshold.

    Provides a combined view of the trade-offs at different operating points.

    Args:
        labels: Ground-truth binary labels.
        scores: Continuous anomaly scores or probabilities.
        optimal_threshold: Threshold used for final predictions.
        save_dir: Output directory.
        filename: Output filename.
        title: Plot title.
    """
    os.makedirs(save_dir, exist_ok=True)

    labels = (np.asarray(labels).flatten() > 0).astype(np.int32)
    scores = np.asarray(scores, dtype=np.float64).flatten()

    precisions, recalls, thresholds = precision_recall_curve(labels, scores, pos_label=1)
    # precision_recall_curve returns len(thresholds) = len(precisions) - 1
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        f1_values = 2 * precisions * recalls / (precisions + recalls)
    f1_values = np.nan_to_num(f1_values)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, precisions, color="#2980b9", linewidth=1.5, label="Precision", alpha=0.8)
    ax.plot(thresholds, recalls, color="#27ae60", linewidth=1.5, label="Recall", alpha=0.8)
    ax.plot(thresholds, f1_values, color="#e74c3c", linewidth=2, label="F1")
    ax.axvline(
        optimal_threshold, color="#8e44ad", linestyle="--", linewidth=2,
        label=f"Threshold = {optimal_threshold:.4f}",
    )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    logger.info("Saved: %s", path)
