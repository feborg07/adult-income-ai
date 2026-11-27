"""Histogram Gradient Boosting algorithm plugin."""

from sklearn.ensemble import HistGradientBoostingClassifier

from .sklearn_base import SklearnAlgorithm


class HistGradientBoostingAlgorithm(SklearnAlgorithm):
    """Histogram-based gradient-boosted classifier."""

    name = "hist_gradient_boosting"
    registration_order = 20
    estimator_class = HistGradientBoostingClassifier

    trusted_types = (
        "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    )