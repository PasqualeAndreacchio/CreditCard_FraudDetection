"""
Shared evaluation utilities for Credit Card Fraud Detection.

Provides:
    - NumpyEncoder: JSON encoder that can serialize numpy types.

Note:
    Plotting functions have been moved to ``src.Evaluation.plots``.
    Metrics and threshold utilities have been moved to ``src.Evaluation.metrics``.
"""

import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that can serialize numpy types.

    Useful to save the metrics dictionary into an external JSON file.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return super().default(obj)
