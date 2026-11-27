"""Verify migrated classifiers against their legacy construction."""
from itertools import product
from pathlib import Path
import joblib
import numpy as np
import pytest
import yaml
from src.models.configuration import load_resolved_config
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.models.algorithms.base import BINARY_CLASSIFICATION
from src.models.algorithms.registry import ALGORITHM_REGISTRY
LEGACY = {
    'logistic_regression': (LogisticRegression, ()),
    'random_forest': (RandomForestClassifier, ('sklearn.tree._tree.Tree',)),
    'hist_gradient_boosting': (HistGradientBoostingClassifier, ('sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor',)),
    'extra_trees': (ExtraTreesClassifier, ('sklearn.tree._tree.Tree',)),
}
@pytest.fixture
def config():
    """Load the production YAML configuration."""
    path = Path(__file__).resolve().parents[1] / 'configs/config.yml'
    return load_resolved_config(path)
def legacy_model(family, params, seed):
    """Reproduce original estimator construction for regression checks."""
    estimator = LEGACY[family][0](**params, random_state=seed)
    if family == 'logistic_regression':
        return make_pipeline(StandardScaler(), estimator)
    return estimator
def test_registry_contains_existing_families():
    """Preserve the original four families and their relative order."""
    # New plugins may be appended without breaking legacy regression checks.
    assert tuple(name for name in ALGORITHM_REGISTRY.names() if name in LEGACY) == tuple(LEGACY)
@pytest.mark.parametrize('family', LEGACY)
def test_plugin_preserves_yaml_candidates(family, config):
    """Compare all original YAML combinations with legacy estimators."""
    plugin = ALGORITHM_REGISTRY.get(family, task_type=BINARY_CLASSIFICATION)
    spec = config['models'][family]
    grid = spec['grid']
    assert plugin.trusted_types == LEGACY[family][1]
    for values in product(*grid.values()):
        params = {**spec['fixed'], **dict(zip(grid, values))}
        actual = plugin.build(params, config['seed'])
        expected = legacy_model(family, params, config['seed'])
        assert type(actual) is type(expected)
        if family == 'logistic_regression':
            assert [name for name, _ in actual.steps] == [name for name, _ in expected.steps]
            for (_, actual_step), (_, expected_step) in zip(actual.steps, expected.steps):
                assert type(actual_step) is type(expected_step)
                assert actual_step.get_params() == expected_step.get_params()
        else:
            assert actual.get_params() == expected.get_params()
@pytest.mark.parametrize('family', LEGACY)
def test_plugin_prediction_and_joblib_roundtrip(family, config, tmp_path):
    """Check legacy prediction equivalence and Joblib roundtrip."""
    plugin = ALGORITHM_REGISTRY.get(family, task_type=BINARY_CLASSIFICATION)
    spec = config['models'][family]
    params = {**spec['fixed'], **{key: values[0] for key, values in spec['grid'].items()}}
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    actual = plugin.build(params, config['seed']).fit(X[:80], y[:80])
    expected = legacy_model(family, params, config['seed']).fit(X[:80], y[:80])
    probabilities = actual.predict_proba(X[80:])
    assert probabilities.shape == (20, 2)
    np.testing.assert_allclose(probabilities, expected.predict_proba(X[80:]), rtol=1e-10, atol=1e-10)
    # Confirm that reloading the fitted model preserves predictions.
    path = tmp_path / 'model.joblib'
    joblib.dump(actual, path)
    np.testing.assert_allclose(joblib.load(path).predict_proba(X[80:]), probabilities, rtol=1e-10, atol=1e-10)
