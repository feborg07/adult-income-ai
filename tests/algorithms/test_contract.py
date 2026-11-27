"""Apply the same binary classifier contract to all discovered production plugins."""
from itertools import product

import joblib
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification

from src.models.algorithms.registry import ALGORITHM_REGISTRY
from src.models.algorithms.base import BINARY_CLASSIFICATION
from src.models.train import load_training_config


@pytest.fixture(scope="module")
def config():
    """Read the production configuration through the real validated loader."""
    return load_training_config()


def test_default_cartridges_match_configuration(config):
    """Reject a production configuration or plugin missing its counterpart."""
    assert set(config['models']) == set(ALGORITHM_REGISTRY.names())


@pytest.mark.parametrize('family', ALGORITHM_REGISTRY.names())
def test_all_grid_candidates_are_fresh_and_cloneable(family, config):
    """Validate constructor compatibility for every configured grid point."""
    plugin = ALGORITHM_REGISTRY.get(family, task_type=BINARY_CLASSIFICATION)
    spec = config['models'][family]
    for values in product(*spec['grid'].values()):
        params = {**spec['fixed'], **dict(zip(spec['grid'], values))}
        model = plugin.build(params, config['seed'])
        assert clone(model) is not model
        assert plugin.build(params, config['seed']) is not model
        assert callable(model.predict_proba)


@pytest.mark.parametrize('family', ALGORITHM_REGISTRY.names())
def test_fit_predict_and_joblib_roundtrip(family, config, tmp_path):
    """Fit a real synthetic model and verify the two-class probability contract."""
    spec = config['models'][family]
    params = {**spec['fixed'], **{name: values[0] for name, values in spec['grid'].items()}}
    # Limit test runtime without altering the production configuration.
    for key in ('n_estimators', 'iterations', 'max_iter'):
        if key in params:
            params[key] = min(params[key], 100 if key == 'max_iter' else 12)
    if 'n_jobs' in params:
        params['n_jobs'] = 2
    X, y = make_classification(n_samples=100, n_features=6, random_state=42)
    plugin = ALGORITHM_REGISTRY.get(family, task_type=BINARY_CLASSIFICATION)
    model = plugin.build(params, config['seed']).fit(X[:80], y[:80])
    probability = model.predict_proba(X[80:])
    assert probability.shape == (20, 2)
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    np.testing.assert_allclose(probability.sum(axis=1), 1, rtol=1e-6, atol=1e-6)
    assert set(model.classes_) == {0, 1}
    path = tmp_path / 'model.joblib'
    joblib.dump(model, path)
    np.testing.assert_allclose(joblib.load(path).predict_proba(X[80:]), probability)
