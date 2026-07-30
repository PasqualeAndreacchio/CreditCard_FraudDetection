"""
Evaluation module for the FraudAutoencoder (reconstruction-based anomaly detection).

Provides:
    - Per-sample reconstruction error (anomaly score) computation
    - Threshold determination (percentile, mean+std, F1-optimal)
    - Binary prediction (normal / fraud)
    - Comprehensive metrics via ``metrics.compute_all_metrics()``
    - Full diagnostic plots via ``plots.*``
"""

from __future__ import annotations

import os
import logging
import json
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.Evaluation.metrics import (
    compute_all_metrics,
    log_metrics,
    find_f1_optimal_threshold,
)
from src.Evaluation.plots import (
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_f1_vs_threshold,
    plot_score_distribution,
    plot_precision_recall_f1_vs_threshold,
)
from src.Evaluation.evaluation_utils import NumpyEncoder

logger = logging.getLogger(__name__)


class ReconstructionEvaluator:
    """Evaluate a reconstruction-based model for anomaly-based fraud detection.

    The evaluator computes per-sample reconstruction errors, determines
    an anomaly threshold, produces binary predictions, and generates
    a comprehensive evaluation report with all metrics and diagnostic plots.

    Args:
        model: Trained autoencoder (e.g. FraudAutoencoder).
        config: Full configuration dictionary (parsed from YAML).
        device: Torch device for computation.

    Example::

        evaluator = ReconstructionEvaluator(model, config, device)
        evaluator.evaluate(test_loader, test_labels)
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.anomaly_cfg = config.get("anomaly", {})

    # Per-sample anomaly score computation
    @torch.no_grad()
    def compute_anomaly_scores(self, loader: DataLoader) -> np.ndarray:
        """Compute per-sample reconstruction error (MSE) for all data."""
        self.model.eval()
        all_errors: list[float] = []

        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            x = x.to(self.device)
            reconstructed = self.model(x)
            mse_per_sample = torch.mean((reconstructed - x) ** 2, dim=tuple(range(1, x.ndim)))
            all_errors.extend(mse_per_sample.cpu().numpy().tolist())

        return np.array(all_errors)

    # Threshold determination methods
    def find_optimal_threshold(
        self,
        scores: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> float:
        """Determine the anomaly threshold using the configured method.

        Args:
            scores: 1-D array of reconstruction errors.
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
            logger.info("Threshold (mean + %.1f×std): %.6f", mult, threshold)

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

    # Binary prediction from scores and threshold
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

    # Full evaluation pipeline
    def evaluate(
        self,
        loader: DataLoader,
        labels: np.ndarray,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Run a complete evaluation.

        Computes all metrics, saves them to a JSON file and generates
        the full suite of diagnostic plots.

        Args:
            loader: DataLoader yielding feature tensors.
            labels: Ground-truth binary labels (0/1).
            threshold: Override anomaly threshold (optional).

        Returns:
            Dictionary with all computed metrics.
        """
        labels = np.asarray(labels).flatten().astype(int)
        scores = self.compute_anomaly_scores(loader)

        if threshold is None:
            threshold = self.find_optimal_threshold(scores, labels)

        predictions = (scores > threshold).astype(np.int32)

        # Compute all metrics
        metrics = compute_all_metrics(labels, predictions, scores, threshold)
        log_metrics(metrics)

        # Generate diagnostic plots
        plot_dir = self.anomaly_cfg.get("plots_dir", "plots/reconstruction")
        self._generate_plots(scores, labels, threshold, save_dir=plot_dir)

        # Save metrics to JSON
        results_dir = self.anomaly_cfg.get("results_dir", "results/reconstruction")
        os.makedirs(results_dir, exist_ok=True)
        metrics_path = os.path.join(results_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4, cls=NumpyEncoder)
        logger.info("Saved metrics to: %s", metrics_path)

        return metrics

    # Plot generation helpers
    def _generate_plots(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        threshold: float,
        save_dir: str = "plots/reconstruction",
    ) -> None:
        """Generate and save the full suite of diagnostic plots.

        Produces:
        1. Reconstruction error distribution (normal vs fraud)
        2. Precision-Recall curve
        3. ROC curve
        4. F1 vs Threshold curve
        5. Precision, Recall & F1 vs Threshold
        6. Confusion matrix heatmap

        Args:
            scores: Per-sample anomaly scores.
            labels: Ground-truth binary labels (0/1).
            threshold: Anomaly threshold used for predictions.
            save_dir: Directory to save plots.
        """
        predictions = (scores > threshold).astype(np.int32)

        # 1. Score distribution
        plot_score_distribution(
            scores, labels, threshold,
            save_dir=save_dir,
            filename="error_distribution.png",
            xlabel="Reconstruction Error (MSE)",
            title="Autoencoder — Reconstruction Error Distribution",
        )

        # 2. Precision-Recall curve
        plot_precision_recall_curve(
            labels, scores,
            save_dir=save_dir,
            filename="precision_recall_curve.png",
            title="Autoencoder — Precision-Recall Curve",
        )

        # 3. ROC curve
        plot_roc_curve(
            labels, scores,
            save_dir=save_dir,
            filename="roc_curve.png",
            title="Autoencoder — ROC Curve",
        )

        # 4. F1 vs Threshold
        plot_f1_vs_threshold(
            labels, scores, threshold,
            save_dir=save_dir,
            filename="f1_vs_threshold.png",
            title="Autoencoder — F1 Score vs Threshold",
        )

        # 5. Precision, Recall & F1 vs Threshold
        plot_precision_recall_f1_vs_threshold(
            labels, scores, threshold,
            save_dir=save_dir,
            filename="pr_f1_vs_threshold.png",
            title="Autoencoder — Precision, Recall & F1 vs Threshold",
        )

        # 6. Confusion matrix
        from sklearn.metrics import confusion_matrix as cm_func
        cm = cm_func(labels, predictions, labels=[0, 1])
        plot_confusion_matrix(
            cm,
            save_dir=save_dir,
            filename="confusion_matrix.png",
            title="Autoencoder — Confusion Matrix",
        )
