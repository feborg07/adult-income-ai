"""Random Forest algorithm plugin."""

from sklearn.ensemble import RandomForestClassifier

from .sklearn_base import SklearnAlgorithm


class RandomForestAlgorithm(SklearnAlgorithm):
    """Bagged decision-tree ensemble."""

    name = "random_forest"
    estimator_class = RandomForestClassifier

    trusted_types = (
        "sklearn.tree._tree.Tree",
    )