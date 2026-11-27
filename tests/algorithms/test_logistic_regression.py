"""Specialized constructor checks for logistic_regression."""
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.models.algorithms.registry import ALGORITHM_REGISTRY
from src.models.train import load_training_config


def test_estimator_type_and_parameters():
    """Verify the expected implementation using the configured first grid candidate."""
    config = load_training_config()
    spec = config['models']['logistic_regression']
    params = {**spec['fixed'], **{key: values[0] for key, values in spec['grid'].items()}}
    plugin = ALGORITHM_REGISTRY.get('logistic_regression')
    model = plugin.build(params, config['seed'])
    assert isinstance(model, Pipeline)
    assert isinstance(model.steps[0][1], StandardScaler)
    assert isinstance(model.steps[-1][1], LogisticRegression)
