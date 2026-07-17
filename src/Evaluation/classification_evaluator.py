import os
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.Evaluation.evaluation_utils import (
    find_f1_optimal_threshold,
    plot_confusion_matrix,
    plot_precision_recall_curve,
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
    ) -> dict[str, Any]:
        """
        Evaluate the model on the given dataset and return the results.
        Args:
            - dataloader (DataLoader): DataLoader yielding feature tensors
            - labels (np.ndarray): Ground-truth labels
            - threshold (float | None): Threshold used to predict the label
        Returns:
            - dict[str, Any]: Dictionary containing the evaluation results
        """
        # Compute the probabilities and, in case, find the best threshold
        probs = self.compute_probabilities(dataloader)
        if threshold is None:
            threshold = self._find_optimal_threshold(probs, labels)

        # Predict the labels
        predictions = self.predict_label(dataloader, threshold)
        
        # Get the metrics
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        pr_auc = average_precision_score(labels, probs[:, 1])
        roc_auc = roc_auc_score(labels, probs[:, 1])
        conf_matrix = confusion_matrix(labels, predictions)
        report = classification_report(labels, predictions)

        # Print the metrics
        logger.info("Threshold: %f", threshold)
        logger.info("Precision: %f", precision)
        logger.info("Recall: %f", recall)
        logger.info("F1: %f", f1)
        logger.info("PR AUC: %f", pr_auc)
        logger.info("ROC AUC: %f", roc_auc)
        logger.info("Confusion Matrix: %s", conf_matrix)
        logger.info("Classification Report: %s", report)

        # Return the metrics
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "conf_matrix": conf_matrix,
            "threshold": threshold,
        }
    

    def plot_results(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        threshold: float,
        save_dir: str = "plots/",
    ) -> None:
        """Generate and save diagnostic plots.

        Produces three plots:
        1. Fraud probability distribution (normal vs fraud) — model-specific
        2. Precision-Recall curve — shared helper
        3. Confusion matrix heatmap — shared helper

        Args:
            scores: Per-sample fraud-class probabilities (probs[:, 1]).
            labels: Ground-truth binary labels (0/1).
            threshold: Classification threshold used for predictions.
            save_dir: Directory to save plots.
        """
        os.makedirs(save_dir, exist_ok=True)

        # ── 1. Probability Distribution (classifier-specific) ─────────
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

        # ── 2. Precision-Recall Curve (shared helper) ────────────────
        plot_precision_recall_curve(
            labels, scores,
            save_dir=save_dir,
            filename="classifier_precision_recall_curve.png",
            title="Classifier — Precision-Recall Curve",
        )

        # ── 3. Confusion Matrix (shared helper) ───────────────────────
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
            - dataloader: DataLoader yielding feature tensors (or (features, labels)
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
        return probabilities[:, 1] > threshold


    def _find_optimal_threshold(
        self, 
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        Determine the best threshold using the configured method.
        Args:
            - scores (np.ndarray): Array of anomaly scores
            - labels (np.ndarray): Ground-truth labels
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
    

