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

import json

from src.Evaluation.evaluation_utils import (
    find_f1_optimal_threshold,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    NumpyEncoder,
)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import matplotlib.pyplot as plt

from src.models.FFNNDecoder import FFNNAutoencoder

logger = logging.getLogger(__name__)


class ReconstructionEvaluator:
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
            threshold = find_f1_optimal_threshold(scores, labels)
            logger.info("Threshold (F1-optimal): %.6f", threshold)

        else:
            raise ValueError(f"Unknown threshold method: '{method}'.")

        return threshold


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
    ) -> None:
        """Run a complete evaluation.

        Computes metrics, saves them to a JSON file and generates diagnostic plots.
        If *threshold* is ``None``, the optimal threshold is computed
        from the data using the configured method.

        Args:
            loader: DataLoader yielding feature tensors.
            labels: Ground-truth binary labels (0/1).
            threshold: Override anomaly threshold (optional).
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

        # Generate and save the plots
        plot_dir = self.anomaly_cfg.get("plots_dir")
        if plot_dir is None:
            logger.warning("plots_dir not found in config!")
            logger.warning("Falling back to default 'plots/reconstruction'")
            plot_dir = "plots/reconstruction"
        self.plot_results(scores, labels, threshold, save_dir=plot_dir)

        # Save the metrics to a file and return them
        results_dir = self.anomaly_cfg.get("results_dir")
        if results_dir is None:
            logger.warning("results_dir not found in config!")
            logger.warning("Falling back to default 'results/reconstruction'")
            results_dir = "results/reconstruction"
        os.makedirs(results_dir, exist_ok=True)

        metrics_path = os.path.join(results_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4, cls=NumpyEncoder)
        logger.info("Saved metrics to: %s", metrics_path)

    # ── Plotting ────────────────────────────────────────────────────────

    def plot_results(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        threshold: float,
        save_dir: str = "plots/reconstruction",
    ) -> None:
        """Generate and save diagnostic plots.

        Produces three plots:
        1. Reconstruction error distribution (normal vs fraud) — model-specific
        2. Precision-Recall curve  — shared helper
        3. Confusion matrix heatmap — shared helper

        Args:
            scores: Per-sample anomaly scores.
            labels: Ground-truth binary labels (0/1).
            threshold: Anomaly threshold used for predictions.
            save_dir: Directory to save plots.
        """
        os.makedirs(save_dir, exist_ok=True)

        # ── 1. Error Distribution (reconstruction-specific) ─────────
        normal_scores = scores[labels == 0]
        fraud_scores = scores[labels == 1]

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

        # ── 2. Precision-Recall Curve (shared helper) ───────────────
        plot_precision_recall_curve(
            labels, scores,
            save_dir=save_dir,
            filename="ffnn_precision_recall_curve.png",
            title="FFNN Autoencoder — Precision-Recall Curve",
        )

        # ── 3. Confusion Matrix (shared helper) ─────────────────────
        predictions = (scores > threshold).astype(np.int32)
        cm = confusion_matrix(labels, predictions)
        plot_confusion_matrix(
            cm,
            save_dir=save_dir,
            filename="ffnn_confusion_matrix.png",
            title="FFNN Autoencoder — Confusion Matrix",
        )
