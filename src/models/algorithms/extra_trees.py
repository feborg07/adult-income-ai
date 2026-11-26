"""Extra Trees algorithm plugin."""

from sklearn.ensemble import ExtraTreesClassifier

from .sklearn_base import SklearnAlgorithm


class ExtraTreesAlgorithm(SklearnAlgorithm):
    """Extremely randomized tree ensemble."""

    name = "extra_trees"
    estimator_class = ExtraTreesClassifier

    trusted_types = (
        "sklearn.tree._tree.Tree",
    )