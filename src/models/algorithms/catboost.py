"""CatBoost binary classifier plugin."""

from collections.abc import Mapping
from typing import Any

import mlflow.catboost
from catboost import CatBoostClassifier

from .sklearn_base import SklearnAlgorithm


class CatBoostAlgorithm(SklearnAlgorithm):
    """Train CatBoost on the same numeric representation as other plugins."""

    name = "catboost"
    registration_order = 50
    estimator_class = CatBoostClassifier
    required_packages = ("catboost", "scikit-learn", "numpy", "scipy", "pandas")

    def build(self, params: Mapping[str, Any], seed: int) -> CatBoostClassifier:
        """Construct a fresh classifier with CatBoost's canonical seed parameter."""
        # CatBoost uses random_seed; avoid mixing aliases in sklearn.clone().
        return self.estimator_class(**dict(params), random_seed=seed)

    def log_model(self, model: CatBoostClassifier) -> None:
        """Log a fitted classifier in CatBoost's native model format."""
        mlflow.catboost.log_model(
            cb_model=model,
            name="model",
            pip_requirements=self.model_requirements(),
        )
