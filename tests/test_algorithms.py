"""Unit tests for the algorithm contract and registry."""

import pytest

from src.models.algorithms.base import (
    BINARY_CLASSIFICATION,
    BaseAlgorithm,
)
from src.models.algorithms.registry import AlgorithmRegistry


class StubAlgorithm(BaseAlgorithm):
    """Test-only plugin; no real training or MLflow operations."""

    name = "stub"
    supported_tasks = (BINARY_CLASSIFICATION,)

    def build(self, params, seed):
        return object()

    def log_model(self, model):
        pass


class SecondAlgorithm(StubAlgorithm):
    name = "second"


def test_base_contract_is_abstract():
    with pytest.raises(TypeError):
        BaseAlgorithm()


def test_registry_lookup():
    registry = AlgorithmRegistry()
    plugin = StubAlgorithm()

    registry.register(plugin)

    assert registry.get(
        "stub", task_type=BINARY_CLASSIFICATION
    ) is plugin


def test_registry_preserves_order():
    registry = AlgorithmRegistry()

    registry.register(StubAlgorithm())
    registry.register(SecondAlgorithm())

    assert registry.names() == ("stub", "second")


def test_duplicate_registration_is_rejected():
    registry = AlgorithmRegistry()
    registry.register(StubAlgorithm())

    with pytest.raises(ValueError, match="already registered"):
        registry.register(StubAlgorithm())


def test_unknown_algorithm_is_rejected():
    with pytest.raises(KeyError, match="Unknown algorithm"):
        AlgorithmRegistry().get("unknown")


def test_invalid_plugin_is_rejected():
    with pytest.raises(TypeError, match="BaseAlgorithm"):
        AlgorithmRegistry().register(object())


def test_incompatible_task_is_rejected():
    registry = AlgorithmRegistry()
    registry.register(StubAlgorithm())

    with pytest.raises(ValueError, match="does not support"):
        registry.get(
            "stub", task_type="multiclass_classification"
        )


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("name", ""),
        ("supported_tasks", ()),
    ],
)
def test_invalid_metadata_is_rejected(attribute, value):
    plugin = StubAlgorithm()
    setattr(plugin, attribute, value)

    with pytest.raises(ValueError):
        AlgorithmRegistry().register(plugin)
