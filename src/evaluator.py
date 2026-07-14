"""
Evaluation and anomaly-detection module for the FFNN Autoencoder.

Provides:
    - Per-sample reconstruction error (anomaly score) computation
    - Threshold determination (percentile, mean+std, F1-optimal)
    - Binary prediction (normal / fraud)
    - Comprehensive metrics (Precision, Recall, F1, AUPRC, AUROC)
    - Diagnostic plots (error distribution, precision-recall curve, confusion matrix)
"""

from __future__ import annotations

import os
import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)

import matplotlib.pyplot as plt

from src.models.ffnn_autoencoder import FFNNAutoencoder

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate an :class:`FFNNAutoencoder` for anomaly-based fraud detection.

    The evaluator computes per-sample reconstruction errors, determines
    an anomaly threshold, produces binary predictions, and generates
    a comprehensive evaluation report.

    Args:
        model: Trained :class:`FFNNAutoencoder`.
        config: Full configuration dictionary (parsed from YAML).
        device: Torch device for computation.

    Example::

        evaluator = Evaluator(model, config, device)
        scores = evaluator.compute_anomaly_scores(test_loader)
        threshold = evaluator.find_optimal_threshold(scores, val_labels)
        preds = evaluator.predict(test_loader, threshold)
        report = evaluator.evaluate(test_loader, test_labels)
    """

    def __init__(
        self,
        model: FFNNAutoencoder,
        config: dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.anomaly_cfg = config.get("anomaly", {})

    # ── Anomaly Scores ──────────────────────────────────────────────────

    @torch.no_grad()
    def compute_anomaly_scores(self, loader: DataLoader) -> np.ndarray:
        """Compute per-sample reconstruction error (MSE) for all data.

        Args:
            loader: DataLoader yielding feature tensors (or (features, labels)
                tuples — labels are ignored).

        Returns:
            1-D numpy array of reconstruction errors, one per sample.
        """
        self.model.eval()
        all_errors: list[torch.Tensor] = []

        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            x = x.to(self.device)
            errors = self.model.compute_reconstruction_error(x, reduction="mean")
            all_errors.append(errors.cpu())

        return torch.cat(all_errors).numpy()

    # ── Threshold Determination ─────────────────────────────────────────

    def find_optimal_threshold(
        self,
        scores: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> float:
        """Determine the anomaly threshold using the configured method.

        Args:
            scores: 1-D array of reconstruction errors (e.g. from validation set).
            labels: Ground-truth labels (required for ``f1_optimal`` method).

        Returns:
            Threshold value: samples with ``score > threshold`` are classified
            as fraudulent.
        """
        method = self.anomaly_cfg.get("threshold_method", "percentile")

        if method == "percentile":
            pct = self.anomaly_cfg.get("percentile", 95)
            threshold = float(np.percentile(scores, pct))
            logger.info("Threshold (percentile=%d%%): %.6f", pct, threshold)

        elif method == "mean_std":
            mult = self.anomaly_cfg.get("std_multiplier", 2.0)
            threshold = float(scores.mean() + mult * scores.std())
            logger.info(
                "Threshold (mean + %.1f×std): %.6f", mult, threshold
            )

        elif method == "f1_optimal":
            if labels is None:
                raise ValueError(
                    "Ground-truth labels are required for 'f1_optimal' threshold."
                )
            threshold = self._find_f1_optimal_threshold(scores, labels)
            logger.info("Threshold (F1-optimal): %.6f", threshold)

        else:
            raise ValueError(f"Unknown threshold method: '{method}'.")

        return threshold

    @staticmethod
    def _find_f1_optimal_threshold(
        scores: np.ndarray, labels: np.ndarray
    ) -> float:
        """Search for the threshold that maximises F1-score.

        Uses the precision-recall curve from scikit-learn to enumerate
        candidate thresholds efficiently.

        Args:
            scores: Per-sample anomaly scores.
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
            f1_scores[best_idx], best_threshold,
        )
        return best_threshold

    # ── Prediction ──────────────────────────────────────────────────────

    def predict(
        self,
        loader: DataLoader,
        threshold: float,
    ) -> np.ndarray:
        """Produce binary predictions (0 = normal, 1 = fraud).

        Args:
            loader: DataLoader yielding feature tensors.
            threshold: Anomaly score threshold.

        Returns:
            1-D numpy array of binary predictions.
        """
        scores = self.compute_anomaly_scores(loader)
        predictions = (scores > threshold).astype(np.int32)
        n_fraud = predictions.sum()
        logger.info(
            "Predictions: %d / %d flagged as fraud (threshold=%.6f).",
            n_fraud, len(predictions), threshold,
        )
        return predictions

    # ── Full Evaluation ─────────────────────────────────────────────────

    def evaluate(
        self,
        loader: DataLoader,
        labels: np.ndarray,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Run a complete evaluation and return a metrics dictionary.

        If *threshold* is ``None``, the optimal threshold is computed
        from the data using the configured method.

        Args:
            loader: DataLoader yielding feature tensors.
            labels: Ground-truth binary labels (0/1).
            threshold: Override anomaly threshold (optional).

        Returns:
            Dictionary with keys: ``threshold``, ``precision``, ``recall``,
            ``f1``, ``auprc``, ``auroc``, ``confusion_matrix``,
            ``classification_report``.
        """
        scores = self.compute_anomaly_scores(loader)

        if threshold is None:
            threshold = self.find_optimal_threshold(scores, labels)

        predictions = (scores > threshold).astype(np.int32)

        precision = float(precision_score(labels, predictions, zero_division=0))
        recall = float(recall_score(labels, predictions, zero_division=0))
        f1 = float(f1_score(labels, predictions, zero_division=0))
        auprc = float(average_precision_score(labels, scores))
        auroc = float(roc_auc_score(labels, scores))
        cm = confusion_matrix(labels, predictions)
        report = classification_report(labels, predictions, target_names=["Normal", "Fraud"])

        metrics = {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auprc": auprc,
            "auroc": auroc,
            "confusion_matrix": cm,
            "classification_report": report,
        }

        logger.info("=" * 50)
        logger.info("  EVALUATION RESULTS")
        logger.info("=" * 50)
        logger.info("  Threshold : %.6f", threshold)
        logger.info("  Precision : %.4f", precision)
        logger.info("  Recall    : %.4f", recall)
        logger.info("  F1-score  : %.4f", f1)
        logger.info("  AUPRC     : %.4f", auprc)
        logger.info("  AUROC     : %.4f", auroc)
        logger.info("  Confusion Matrix:\n%s", cm)
        logger.info("\n%s", report)
        logger.info("=" * 50)

        return metrics

    # ── Plotting ────────────────────────────────────────────────────────

    def plot_results(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        threshold: float,
        save_dir: str = "plots/",
    ) -> None:
        """Generate and save diagnostic plots.

        Produces three plots:
        1. Reconstruction error distribution (normal vs fraud)
        2. Precision-Recall curve
        3. Confusion matrix heatmap

        Args:
            scores: Per-sample anomaly scores.
            labels: Ground-truth binary labels (0/1).
            threshold: Anomaly threshold used for predictions.
            save_dir: Directory to save plots.
        """
        os.makedirs(save_dir, exist_ok=True)

        normal_scores = scores[labels == 0]
        fraud_scores = scores[labels == 1]

        # ── 1. Error Distribution ───────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(normal_scores, bins=100, alpha=0.6, label="Normal", color="#3498db", density=True)
        ax.hist(fraud_scores, bins=100, alpha=0.6, label="Fraud", color="#e74c3c", density=True)
        ax.axvline(threshold, color="#2ecc71", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.4f}")
        ax.set_xlabel("Reconstruction Error (MSE)")
        ax.set_ylabel("Density")
        ax.set_title("FFNN Autoencoder — Reconstruction Error Distribution")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "ffnn_error_distribution.png"), dpi=300)
        plt.close()
        logger.info("Saved: %s", os.path.join(save_dir, "ffnn_error_distribution.png"))

        # ── 2. Precision-Recall Curve ───────────────────────────────
        precisions, recalls, _ = precision_recall_curve(labels, scores)
        auprc = average_precision_score(labels, scores)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recalls, precisions, color="#9b59b6", linewidth=2)
        ax.fill_between(recalls, precisions, alpha=0.15, color="#9b59b6")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"FFNN Autoencoder — Precision-Recall Curve (AUPRC = {auprc:.4f})")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "ffnn_precision_recall_curve.png"), dpi=300)
        plt.close()
        logger.info("Saved: %s", os.path.join(save_dir, "ffnn_precision_recall_curve.png"))

        # ── 3. Confusion Matrix ─────────────────────────────────────
        predictions = (scores > threshold).astype(np.int32)
        cm = confusion_matrix(labels, predictions)

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
            title="FFNN Autoencoder — Confusion Matrix",
        )
        # Annotate cells
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, f"{cm[i, j]:,}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold",
                )
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "ffnn_confusion_matrix.png"), dpi=300)
        plt.close()
        logger.info("Saved: %s", os.path.join(save_dir, "ffnn_confusion_matrix.png"))
