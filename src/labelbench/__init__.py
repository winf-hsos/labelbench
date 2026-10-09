"""labelbench: evaluate interchangeable classifiers against a gold standard."""

__version__ = "0.1.0"

from labelbench.cache import JsonCache  # noqa: E402
from labelbench.classifier import Classifier, Prediction, TaskInfo  # noqa: E402

__all__ = ["Classifier", "JsonCache", "Prediction", "TaskInfo", "__version__"]
