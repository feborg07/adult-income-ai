"""Logistic Regression algorithm plugin."""

from collections.abc import Mapping
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .sklearn_base import SklearnAlgorithm


class LogisticRegressionAlgorithm(SklearnAlgorithm):
    """Linear classifier with additional feature scaling."""

    name = "logistic_regression"
    registration_order = 0
    estimator_class = LogisticRegression

    def build(
        self, params: Mapping[str, Any], seed: int
    ) -> Pipeline:
        """Preserve the existing additional scaling step."""
        estimator = super().build(params, seed)

        return make_pipeline(StandardScaler(), estimator)