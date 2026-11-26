"""Common contract for interchangeable algorithm plugins."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

BINARY_CLASSIFICATION = "binary_classification"

class BaseAlgorithm(ABC):
    """Build estimators and handle algorithm-specific model logging."""

    name: ClassVar[str]
    supported_tasks: ClassVar[tuple[str, ...]]

    @abstractmethod
    def build(self, params: Mapping[str, Any], seed: int) -> Any:
        """Return a fresh, unfitted, sklearn-compatible estimator.

        Params contains the fixed and candidate parameters from YAML.
        For binary classification, the estimator must support predict_proba.
        """
        raise NotImplementedError

    @abstractmethod
    def log_model(self, model: Any) -> None:
        """Log the fitted estimator inside the engine's active MLflow run."""
        raise NotImplementedError