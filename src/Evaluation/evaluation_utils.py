import os
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
)

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that can serialize numpy types.
    Useful to save the metrics into an external JSON file.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return super().default(obj)

# Threshold util
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
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    # F1 = 2 * P * R / (P + R)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = 2 * precisions * recalls / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores)

    best_idx = np.argmax(f1_scores)
    # precision_recall_curve returns len(thresholds) = len(precisions) - 1
    best_threshold = float(thresholds[min(best_idx, len(thresholds) - 1)])
    logger.info(
        "F1-optimal search: best_f1=%.4f at threshold=%.6f",
        f1_scores[best_idx],
        best_threshold,
    )
    return best_threshold


# Plot utils
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

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Normal", "Fraud"],
        yticklabels=["Normal", "Fraud"],
        xlabel="Predicted",
        ylabel="Actual",
        title=title,
    )
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

    precisions, recalls, _ = precision_recall_curve(labels, scores)
    auprc = average_precision_score(labels, scores)
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
