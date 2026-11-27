"""Test the three new boosting plugins and their deployment contracts."""

from importlib import import_module
from itertools import product
from pathlib import Path
from unittest.mock import Mock

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml
from src.models.configuration import load_resolved_config
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.models.algorithms.base import BINARY_CLASSIFICATION
from src.models.algorithms.registry import ALGORITHM_REGISTRY
from src.models.tasks.binary_classification import positive_probabilities
from src.models.train import build_deployable_pipeline


# Model classes, native MLflow flavor, logging argument, and seed parameter.
FAMILIES = {
    "xgboost": (XGBClassifier, "xgb_model", "random_state"),
    "catboost": (CatBoostClassifier, "cb_model", "random_seed"),
    "lightgbm": (LGBMClassifier, "lgb_model", "random_state"),
}


@pytest.fixture(scope="module")
def config():
    """Read the production YAML without changing it."""
    path = Path(__file__).resolve().parents[1] / "configs/config.yml"
    return load_resolved_config(path)


@pytest.fixture(scope="module")
def sample():
    """Create a named, numeric synthetic binary classification dataset."""
    X, y = make_classification(
        n_samples=160, n_features=6, n_informative=4, random_state=42,
    )
    X = pd.DataFrame(X, columns=[
        "age", "education-num", "hours-per-week", "capital-gain-log1p",
        "workclass_Private", "occupation_Exec-managerial",
    ])
    return X.iloc[:120], X.iloc[120:], pd.Series(y[:120])


def small_model(family, config):
    """Build a cheap candidate with production settings and fewer trees."""
    spec = config["models"][family]
    params = {**spec["fixed"], **{k: v[0] for k, v in spec["grid"].items()}}
    # Limit the synthetic test runtime; production YAML is not altered.
    params["iterations" if family == "catboost" else "n_estimators"] = 12
    plugin = ALGORITHM_REGISTRY.get(family, task_type=BINARY_CLASSIFICATION)
    return plugin, plugin.build(params, config["seed"])


def test_registry_and_config_agree(config):
    """Require the new families and matching registry and YAML order."""
    assert ALGORITHM_REGISTRY.names() == tuple(config["models"])
    assert set(FAMILIES).issubset(config["models"])


@pytest.mark.parametrize("family", FAMILIES)
def test_all_grid_candidates_are_cloneable(family, config):
    """Preserve fixed and swept parameters and the configured seed."""
    plugin = ALGORITHM_REGISTRY.get(family, task_type=BINARY_CLASSIFICATION)
    estimator_class, _, seed_key = FAMILIES[family]
    spec = config["models"][family]
    grid = spec["grid"]

    for values in product(*grid.values()):
        params = {**spec["fixed"], **dict(zip(grid, values))}
        model = plugin.build(params, config["seed"])
        cloned = clone(model)
        assert isinstance(model, estimator_class)
        assert cloned is not model
        for key, value in {**params, seed_key: config["seed"]}.items():
            assert cloned.get_params()[key] == value


@pytest.mark.parametrize("family", FAMILIES)
def test_prediction_and_deployable_joblib_roundtrip(family, config, sample, tmp_path):
    """Fit, predict and reload the complete raw-input deployment pipeline."""
    X_train, X_val, y_train = sample
    # Reproduce the engine's fitted preprocessor and named feature adapter.
    preprocessor = StandardScaler().fit(X_train)
    processed_train = pd.DataFrame(preprocessor.transform(X_train), columns=X_train.columns)
    processed_val = pd.DataFrame(preprocessor.transform(X_val), columns=X_val.columns)
    _, prototype = small_model(family, config)
    model = clone(prototype).fit(processed_train, y_train)
    expected = positive_probabilities(model, processed_val)
    assert expected.shape == (len(X_val),)

    pipeline = build_deployable_pipeline(
        preprocessor, model, list(X_train.columns), processed_train,
    )
    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, path)
    restored = joblib.load(path)
    np.testing.assert_allclose(
        positive_probabilities(restored, X_val), expected, rtol=1e-8, atol=1e-8,
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_native_logging_arguments(family, config, monkeypatch):
    """Dispatch logging to the correct native flavor with pinned dependencies."""
    plugin, model = small_model(family, config)
    flavor = import_module(f"mlflow.{family}")
    requirements = [f"{family}==0.0.0"]
    monkeypatch.setattr(plugin, "model_requirements", lambda: requirements)
    logger = Mock()
    monkeypatch.setattr(flavor, "log_model", logger)
    plugin.log_model(model)

    expected = {FAMILIES[family][1]: model, "name": "model", "pip_requirements": requirements}
    if family == "xgboost":
        expected["model_format"] = "ubj"
    elif family == "lightgbm":
        expected.update(serialization_format="skops", skops_trusted_types=list(plugin.trusted_types))
    logger.assert_called_once_with(**expected)


@pytest.mark.parametrize("family", FAMILIES)
def test_native_mlflow_roundtrip(family, config, sample, tmp_path, monkeypatch):
    """Log and reload a real native MLflow model using an isolated SQLite store."""
    X_train, X_val, y_train = sample
    plugin, model = small_model(family, config)
    model.fit(X_train, y_train)
    expected = positive_probabilities(model, X_val)
    flavor = import_module(f"mlflow.{family}")
    original_logger = flavor.log_model
    logged = []
    previous_uri = mlflow.get_tracking_uri()

    def capture(*args, **kwargs):
        """Capture the actual logged model URI without mocking serialization."""
        info = original_logger(*args, **kwargs)
        logged.append(info)
        return info

    monkeypatch.setattr(flavor, "log_model", capture)
    # Use a fresh backend and artifact directory for every parameterized test.
    try:
        mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}")
        experiment_id = mlflow.create_experiment(
            "native-boosting-test", artifact_location=(tmp_path / "artifacts").as_uri(),
        )
        with mlflow.start_run(experiment_id=experiment_id):
            plugin.log_model(model)
        assert len(logged) == 1
        restored = flavor.load_model(logged[0].model_uri)
        np.testing.assert_allclose(
            positive_probabilities(restored, X_val), expected, rtol=1e-7, atol=1e-8,
        )
    finally:
        # Restore process-global settings for the remainder of the test suite.
        mlflow.set_tracking_uri(previous_uri)
