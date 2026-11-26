"""Tests for shared scikit-learn plugin behavior."""

from unittest.mock import Mock

import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

from src.models.algorithms import sklearn_base
from src.models.algorithms.base import BINARY_CLASSIFICATION
from src.models.algorithms.sklearn_base import SklearnAlgorithm


class StubSklearnAlgorithm(SklearnAlgorithm):
    """Test-only concrete plugin; not registered for production."""

    name = "stub_sklearn"
    estimator_class = RandomForestClassifier


class TrustedStubAlgorithm(StubSklearnAlgorithm):
    trusted_types = ("sklearn.tree._tree.Tree",)


def test_shared_class_is_abstract():
    with pytest.raises(TypeError):
        SklearnAlgorithm()


def test_build_preserves_parameters_and_seed():
    plugin = StubSklearnAlgorithm()
    params = {"n_estimators": 7, "max_depth": 3}

    model = plugin.build(params, seed=42)

    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 7
    assert model.max_depth == 3
    assert model.random_state == 42
    assert params == {"n_estimators": 7, "max_depth": 3}
    assert plugin.supported_tasks == (BINARY_CLASSIFICATION,)


def test_build_returns_fresh_cloneable_estimators():
    plugin = StubSklearnAlgorithm()

    first = plugin.build({}, seed=42)
    second = plugin.build({}, seed=42)

    assert first is not second
    assert not hasattr(first, "estimators_")
    assert clone(first).get_params() == first.get_params()


def test_model_requirements_are_pinned(monkeypatch):
    monkeypatch.setattr(
        sklearn_base, "package_version", lambda _: "1.2.3"
    )

    assert StubSklearnAlgorithm().model_requirements() == [
        "scikit-learn==1.2.3",
        "numpy==1.2.3",
        "scipy==1.2.3",
        "pandas==1.2.3",
        "cloudpickle==1.2.3",
        "skops==1.2.3",
    ]


@pytest.mark.parametrize("plugin, trusted", [
    (StubSklearnAlgorithm(), []),
    (TrustedStubAlgorithm(), ["sklearn.tree._tree.Tree"]),
])
def test_log_model_uses_expected_mlflow_arguments(
    monkeypatch, plugin, trusted
):
    model = plugin.build({}, seed=42)
    requirements = ["scikit-learn==1.2.3"]

    monkeypatch.setattr(
        plugin, "model_requirements", lambda: requirements
    )

    logger = Mock()

    monkeypatch.setattr(
        sklearn_base.mlflow.sklearn, "log_model", logger
    )

    plugin.log_model(model)

    logger.assert_called_once_with(
        sk_model=model,
        name="model",
        skops_trusted_types=trusted,
        pip_requirements=requirements,
    )