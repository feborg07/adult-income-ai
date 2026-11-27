"""Extra Trees algorithm plugin."""

from sklearn.ensemble import ExtraTreesClassifier

from .sklearn_base import SklearnAlgorithm


class ExtraTreesAlgorithm(SklearnAlgorithm):
    """Extremely randomized tree ensemble."""

    name = "extra_trees"
    registration_order = 30
    estimator_class = ExtraTreesClassifier

    trusted_types = (
        "sklearn.tree._tree.Tree",
    )