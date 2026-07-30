"""
Evaluation sub-package for Credit Card Fraud Detection.

Provides:
    - metrics: Centralized metrics computation (compute_all_metrics, find_f1_optimal_threshold).
    - plots: Centralized plotting (confusion matrix, PR curve, ROC curve, etc.).
    - ReconstructionEvaluator: Full evaluation pipeline for reconstruction-based models.
    - ClassificationEvaluator: Full evaluation pipeline for supervised classifiers.
    - NumpyEncoder: JSON encoder for numpy types.
"""

from src.evaluation.metrics import compute_all_metrics, log_metrics, find_f1_optimal_threshold
from src.evaluation.reconstruction_evaluator import ReconstructionEvaluator
from src.evaluation.classification_evaluator import ClassificationEvaluator
from src.evaluation.evaluation_utils import NumpyEncoder

__all__ = [
    "compute_all_metrics",
    "log_metrics",
    "find_f1_optimal_threshold",
    "ReconstructionEvaluator",
    "ClassificationEvaluator",
    "NumpyEncoder",
]
