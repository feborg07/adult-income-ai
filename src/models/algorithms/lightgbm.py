"""LightGBM binary classifier plugin."""

import mlflow.lightgbm
from lightgbm import LGBMClassifier

from .sklearn_base import SklearnAlgorithm


class LightGBMAlgorithm(SklearnAlgorithm):
    """Build LightGBM classifiers and retain the probability prediction API."""

    name = "lightgbm"
    estimator_class = LGBMClassifier
    required_packages = (
        "lightgbm", "scikit-learn", "numpy", "scipy", "pandas", "skops",
    )
    # Explicit allowlist documented by MLflow for LGBMClassifier.
    trusted_types = (
        "collections.OrderedDict",
        "lightgbm.basic.Booster",
        "lightgbm.sklearn.LGBMClassifier",
    )

    def log_model(self, model: LGBMClassifier) -> None:
        """Log a fitted sklearn-compatible LightGBM classifier using skops."""
        # Log the wrapper rather than only its Booster, retaining predict_proba.
        mlflow.lightgbm.log_model(
            lgb_model=model,
            name="model",
            serialization_format="skops",
            skops_trusted_types=list(self.trusted_types),
            pip_requirements=self.model_requirements(),
        )
