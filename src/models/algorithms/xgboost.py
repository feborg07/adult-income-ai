"""XGBoost binary classifier plugin."""

import mlflow.xgboost
from xgboost import XGBClassifier

from .sklearn_base import SklearnAlgorithm


class XGBoostAlgorithm(SklearnAlgorithm):
    """Build XGBoost classifiers and log them in native UBJSON format."""

    name = "xgboost"
    estimator_class = XGBClassifier
    required_packages = ("xgboost", "scikit-learn", "numpy", "scipy", "pandas")

    def log_model(self, model: XGBClassifier) -> None:
        """Log a fitted classifier in the active MLflow run."""
        # Preserve the sklearn wrapper using the native XGBoost flavor.
        mlflow.xgboost.log_model(
            xgb_model=model,
            name="model",
            model_format="ubj",
            pip_requirements=self.model_requirements(),
        )
