"""Shared construction and MLflow logging for scikit-learn plugins."""

from abc import abstractmethod
from collections.abc import Mapping
from importlib.metadata import version as package_version
from typing import Any, ClassVar

import mlflow.sklearn
from sklearn.base import BaseEstimator

from .base import BINARY_CLASSIFICATION, BaseAlgorithm


class SklearnAlgorithm(BaseAlgorithm):
    """Common implementation for scikit-learn classifier plugins."""

    supported_tasks = (BINARY_CLASSIFICATION,)

    # Concrete plugins may override this narrow serialization allowlist.
    trusted_types: ClassVar[tuple[str, ...]] = ()

    # Preserve the explicit model dependencies used by the existing trainer.
    required_packages: ClassVar[tuple[str, ...]] = (
        "scikit-learn",
        "numpy",
        "scipy",
        "pandas",
        "cloudpickle",
        "skops",
    )

    @property
    @abstractmethod
    def estimator_class(self) -> type[BaseEstimator]:
        """The estimator class supplied by a concrete plugin."""
        raise NotImplementedError

    def build(
        self, params: Mapping[str, Any], seed: int
    ) -> BaseEstimator:
        """Build a fresh estimator using YAML parameters and the shared seed."""
        return self.estimator_class(
            **dict(params),
            random_state=seed,
        )

    def model_requirements(self) -> list[str]:
        """Pin model dependencies to their installed versions."""
        return [
            f"{package}=={package_version(package)}"
            for package in self.required_packages
        ]

    def log_model(self, model: BaseEstimator) -> None:
        """Log the fitted estimator in the engine's active MLflow run."""
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=list(self.trusted_types),
            pip_requirements=self.model_requirements(),
        )