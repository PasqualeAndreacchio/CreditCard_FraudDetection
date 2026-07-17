from __future__ import annotations

import os
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
import json
from torch.utils.data import DataLoader

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
from src.models.Model import Complete_Autoencoder
from typing import Any

logger = logging.getLogger(__name__)


class ClassificationEvaluator:
    """
    Evaluate a "Complete_Autoencoder" for direct fraud classification.

    The evaluator computes per-sample fraud probabilities via softmax,
    determines an optimal decision threshold, produces binary predictions,
    and generates a comprehensive evaluation report.

    Args of the constructor:
        - model (Complete_Autoencoder): Trained "Complete_Autoencoder".
        - config (dict): Full configuration dictionary (parsed from YAML).
        - device (str): Torch device for computation.   
    """

    def __init__(
        self,
        model: Complete_Autoencoder,
        config: dict[str, Any],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.config = config
        self.device = device
        self.model = model.to(device)
        self.classification_config = config.get("classification", {})
        
    
    def evaluate(
        self,
        dataloader: DataLoader,
        labels: np.ndarray,
        threshold: float | None = None,
    ) -> None:
        """
        Evaluate the model on the given dataset.
        Computes metrics, saves them to a JSON file and generates diagnostic plots.
        Args:
            - dataloader (DataLoader): DataLoader yielding feature tensors
            - labels (np.ndarray): Ground-truth labels
            - threshold (float | None): Threshold used to predict the label (auto-computed if None)
        """
        # Compute the probabilities and, in case, find the best threshold
        probs = self.compute_probabilities(dataloader)
        if threshold is None:
            threshold = self._find_optimal_threshold(probs[:, 1], labels)

        # Predict the labels
        predictions = (probs[:, 1] > threshold).astype(np.int32)
        
        # Get the metrics and save them into a dict
        precision   = precision_score(labels, predictions, zero_division=0)
        recall      = recall_score(labels, predictions, zero_division=0)
        f1          = f1_score(labels, predictions, zero_division=0)
        pr_auc      = average_precision_score(labels, probs[:, 1])
        roc_auc     = roc_auc_score(labels, probs[:, 1])
        conf_matrix = confusion_matrix(labels, predictions)
        report      = classification_report(labels, predictions, target_names=["Normal", "Fraud"])

        metrics = {
            "threshold"              : threshold,
            "precision"              : precision,
            "recall"                 : recall,
            "f1"                     : f1,
            "pr_auc"                 : pr_auc,
            "roc_auc"                : roc_auc,
            "conf_matrix"            : conf_matrix,
            "classification_report"  : report,
        }

        # Print the metrics
        logger.info("=" * 50)
        logger.info("  EVALUATION RESULTS")
        logger.info("=" * 50)
        logger.info("  Threshold  : %.6f", threshold)
        logger.info("  Precision  : %.4f", precision)
        logger.info("  Recall     : %.4f", recall)
        logger.info("  F1-score   : %.4f", f1)
        logger.info("  PR AUC     : %.4f", pr_auc)
        logger.info("  ROC AUC    : %.4f", roc_auc)
        logger.info("  Confusion Matrix:\n%s", conf_matrix)
        logger.info("\n%s", report)
        logger.info("=" * 50)

        # Generate and save the plots
        plot_dir = self.classification_config.get("plots_dir")
        
        if plot_dir is None:
            logger.warning("plots_dir not found in config!")
            logger.warning("Falling back to default 'plots/classification'")
            plot_dir = "plots/classification"
        self.plot_results(probs[:, 1], labels, threshold, save_dir=plot_dir)

        # Save the metrics to a file and return them
        results_dir = self.classification_config.get("results_dir")
        if results_dir is None:
            logger.warning("results_dir not found in config!")
            logger.warning("Falling back to default 'results/classification'")
            results_dir = "results/classification"
        os.makedirs(results_dir, exist_ok=True)
        
        metrics_path = os.path.join(results_dir, "metrics.json")
        

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4, cls=NumpyEncoder)
        logger.info("Saved metrics to: %s", metrics_path)
    

    def plot_results(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        threshold: float,
        save_dir: str = "plots/classification",
    ) -> None:
        """
        Generate and save diagnostic plots.

        Produces three plots:
            - Fraud probability distribution (normal vs fraud) — model-specific
            - Precision-Recall curve — shared helper
            - Confusion matrix heatmap — shared helper

        Args of the method:
            - scores (np.ndarray): Per-sample fraud-class probabilities (probs[:, 1]).
            - labels (np.ndarray): Ground-truth binary labels (0/1).
            - threshold (float): Classification threshold used for predictions.
            - save_dir (str): Directory to save plots.
        """
        
        # Create the directory to save the plots
        os.makedirs(save_dir, exist_ok=True)

        # Probability distribution
        normal_scores = scores[labels == 0]
        fraud_scores = scores[labels == 1]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(normal_scores, bins=100, alpha=0.6, label="Normal", color="#3498db", density=True)
        ax.hist(fraud_scores, bins=100, alpha=0.6, label="Fraud", color="#e74c3c", density=True)
        ax.axvline(
            threshold, color="#2ecc71", linestyle="--", linewidth=2,
            label=f"Threshold = {threshold:.4f}",
        )
        ax.set_xlabel("Predicted Probability (Fraud class)")
        ax.set_ylabel("Density")
        ax.set_title("Classifier — Fraud Probability Distribution")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "classifier_prob_distribution.png"), dpi=300)
        plt.close()
        logger.info("Saved: %s", os.path.join(save_dir, "classifier_prob_distribution.png"))

        # Precision-recall curve
        plot_precision_recall_curve(
            labels, scores,
            save_dir=save_dir,
            filename="classifier_precision_recall_curve.png",
            title="Classifier — Precision-Recall Curve",
        )

        # Confusion matrix
        predictions = (scores > threshold).astype(np.int32)
        cm = confusion_matrix(labels, predictions)
        plot_confusion_matrix(
            cm,
            save_dir=save_dir,
            filename="classifier_confusion_matrix.png",
            title="Classifier — Confusion Matrix",
        )


    @torch.no_grad()
    def compute_probabilities(self, dataloader: DataLoader) -> np.ndarray:
        """
        Compute probabilities using softmax for each sample in the dataloader.
        Args:
            - dataloader (DataLoader): DataLoader yielding feature tensors (or (features, labels)
              tuples — labels are ignored)
        Returns:
            - np.ndarray: Array of probabilities for each sample
        """
        self.model.eval()
        all_probs: list[torch.Tensor] = []

        # Iterate over the dataloader
        for batch in dataloader:
            # If dataloader contains (features, labels) tuples, ignore the labels
            # Else assume batch is the feature tensor 
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            # Compute the probabilities using softmax function
            x = x.to(self.device)
            outputs = self.model(x)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())

        return torch.cat(all_probs).numpy()


    def predict_label(self, dataloader: DataLoader, threshold: float = 0.5) -> np.ndarray:
        """
        Predict the label of each sample in the dataloader using the "compute_probabilities"
        function: if the probability is higher than the threshold, the sample has label 1 (Fraud).
        Args:
            - dataloader (DataLoader): DataLoader yielding feature tensors (or (features, labels)
            - threshold (float): Threshold used to predict the label
        Returns:
            - np.ndarray: Array of predicted labels
        """

        # Compute probabilities using the defined function and keep only
        # labels with probability higher than the provided threshold
        probabilities = self.compute_probabilities(dataloader)
        return (probabilities[:, 1] > threshold).astype(np.int32)


    def _find_optimal_threshold(
        self,
        scores: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> float:
        """
        Determine the best threshold using the configured method.
        Args:
            - scores (np.ndarray): Array of anomaly scores
            - labels (np.ndarray | None): Ground-truth labels (required for 'f1_optimal' method)
        Returns:
            - float: Threshold value
        """
        # Get the method from the config YAML file
        method = self.classification_config.get("threshold_method", "f1_optimal")

        if method == "percentile":
            pct = self.classification_config.get("percentile", 95)
            threshold = float(np.percentile(scores, pct))
            logger.info("Threshold (percentile=%d%%): %.6f", pct, threshold)

        elif method == "mean_std":
            mult = self.classification_config.get("std_multiplier", 2.0)
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
    

