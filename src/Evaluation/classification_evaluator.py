"""
Evaluation module for the supervised classifier (FraudDetectionMLP).

Provides:
    - Per-sample fraud probability computation (sigmoid for binary output)
    - Threshold determination (percentile, mean+std, F1-optimal)
    - Binary prediction
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


class ClassificationEvaluator:
    """Evaluate a supervised classifier for direct fraud detection.

    The evaluator computes per-sample fraud probabilities (via sigmoid
    for a single-output model), determines an optimal decision threshold,
    produces binary predictions, and generates a comprehensive evaluation
    report with all metrics and diagnostic plots.

    Args:
        model: Trained classifier (e.g. FraudDetectionMLP, or any nn.Module
            with a single logit output).
        config: Full configuration dictionary (parsed from YAML).
        device: Torch device for computation.

    Example::

        evaluator = ClassificationEvaluator(model, config, device)
        evaluator.evaluate(test_loader, test_labels)
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict[str, Any],
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = model.to(self.device)
        self.classification_cfg = config.get("classification", {})

    # ── Probability Computation ─────────────────────────────────────────

    @torch.no_grad()
    def compute_probabilities(self, dataloader: DataLoader) -> np.ndarray:
        """Compute fraud probability for each sample.

        For a model with a single logit output (BCEWithLogitsLoss),
        applies sigmoid to get the fraud probability. For a 2-class
        softmax model, takes the probability of class 1.

        Args:
            dataloader: DataLoader yielding feature tensors (or (features, labels)
              tuples — labels are ignored).

        Returns:
            1-D array of fraud probabilities, shape ``(N,)``.
        """
        self.model.eval()
        all_probs: list[torch.Tensor] = []

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            x = x.to(self.device)
            outputs = self.model(x)

            if outputs.shape[-1] == 1:
                # Single output logit → sigmoid for probability
                probs = torch.sigmoid(outputs).squeeze(-1)
            else:
                # Multi-class output → softmax, take fraud class (index 1)
                probs = torch.softmax(outputs, dim=1)[:, 1]

            all_probs.append(probs.cpu())

        return torch.cat(all_probs).numpy()

    # ── Prediction ──────────────────────────────────────────────────────

    def predict(self, dataloader: DataLoader, threshold: float = 0.5) -> np.ndarray:
        """Predict the label of each sample.

        Args:
            dataloader: DataLoader yielding feature tensors.
            threshold: Decision threshold (default: 0.5).

        Returns:
            1-D array of binary predictions (0/1).
        """
        probabilities = self.compute_probabilities(dataloader)
        return (probabilities > threshold).astype(np.int32)

    # ── Full Evaluation ─────────────────────────────────────────────────

    def evaluate(
        self,
        dataloader: DataLoader,
        labels: np.ndarray,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Run a complete evaluation.

        Computes all metrics, saves them to a JSON file, and generates
        the full suite of diagnostic plots.

        Args:
            dataloader: DataLoader yielding feature tensors.
            labels: Ground-truth binary labels (0/1).
            threshold: Override threshold (auto-computed via F1-optimal if None).

        Returns:
            Dictionary with all computed metrics.
        """
        labels = np.asarray(labels).flatten().astype(int)
        probs = self.compute_probabilities(dataloader)

        if threshold is None:
            threshold = self._find_optimal_threshold(probs, labels)

        predictions = (probs > threshold).astype(np.int32)

        # ── Compute all metrics ─────────────────────────────────────
        metrics = compute_all_metrics(labels, predictions, probs, threshold)
        log_metrics(metrics)

        # ── Generate plots ──────────────────────────────────────────
        plot_dir = self.classification_cfg.get("plots_dir", "plots/classification")
        self._generate_plots(probs, labels, threshold, save_dir=plot_dir)

        # ── Save metrics to JSON ────────────────────────────────────
        results_dir = self.classification_cfg.get("results_dir", "results/classification")
        os.makedirs(results_dir, exist_ok=True)
        metrics_path = os.path.join(results_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4, cls=NumpyEncoder)
        logger.info("Saved metrics to: %s", metrics_path)

        return metrics

    # ── Plotting ────────────────────────────────────────────────────────

    def _generate_plots(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        threshold: float,
        save_dir: str = "plots/classification",
    ) -> None:
        """Generate and save the full suite of diagnostic plots.

        Produces:
        1. Fraud probability distribution
        2. Precision-Recall curve
        3. ROC curve
        4. F1 vs Threshold
        5. Precision, Recall & F1 vs Threshold
        6. Confusion matrix
        """
        predictions = (scores > threshold).astype(np.int32)

        # 1. Probability distribution (no percentile clipping, full 0.0 to 1.0 probability range)
        plot_score_distribution(
            scores, labels, threshold,
            save_dir=save_dir,
            filename="classifier_prob_distribution.png",
            xlabel="Predicted Probability (Fraud class)",
            title="Classifier — Fraud Probability Distribution",
            clip_percentile=None,
        )

        # 2. Precision-Recall curve
        plot_precision_recall_curve(
            labels, scores,
            save_dir=save_dir,
            filename="classifier_precision_recall_curve.png",
            title="Classifier — Precision-Recall Curve",
        )

        # 3. ROC curve
        plot_roc_curve(
            labels, scores,
            save_dir=save_dir,
            filename="classifier_roc_curve.png",
            title="Classifier — ROC Curve",
        )

        # 4. F1 vs Threshold
        plot_f1_vs_threshold(
            labels, scores, threshold,
            save_dir=save_dir,
            filename="classifier_f1_vs_threshold.png",
            title="Classifier — F1 Score vs Threshold",
        )

        # 5. Precision, Recall & F1 vs Threshold
        plot_precision_recall_f1_vs_threshold(
            labels, scores, threshold,
            save_dir=save_dir,
            filename="classifier_pr_f1_vs_threshold.png",
            title="Classifier — Precision, Recall & F1 vs Threshold",
        )

        # 6. Confusion matrix
        from sklearn.metrics import confusion_matrix as cm_func
        cm = cm_func(labels, predictions, labels=[0, 1])
        plot_confusion_matrix(
            cm,
            save_dir=save_dir,
            filename="classifier_confusion_matrix.png",
            title="Classifier — Confusion Matrix",
        )

    # ── Private Helpers ─────────────────────────────────────────────────

    def _find_optimal_threshold(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Determine the best threshold using the configured method.

        Args:
            scores: Array of fraud probabilities.
            labels: Ground-truth labels.

        Returns:
            Threshold value.
        """
        method = self.classification_cfg.get("threshold_method", "f1_optimal")

        if method == "percentile":
            pct = self.classification_cfg.get("percentile", 95)
            threshold = float(np.percentile(scores, pct))
            logger.info("Threshold (percentile=%d%%): %.6f", pct, threshold)

        elif method == "mean_std":
            mult = self.classification_cfg.get("std_multiplier", 2.0)
            threshold = float(scores.mean() + mult * scores.std())
            logger.info("Threshold (mean + %.1f×std): %.6f", mult, threshold)

        elif method == "f1_optimal":
            threshold = find_f1_optimal_threshold(scores, labels)
            logger.info("Threshold (F1-optimal): %.6f", threshold)

        else:
            raise ValueError(f"Unknown threshold method: '{method}'.")

        return threshold
