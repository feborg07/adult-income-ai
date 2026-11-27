"""Specialized constructor checks for catboost."""
from catboost import CatBoostClassifier
from src.models.algorithms.registry import ALGORITHM_REGISTRY
from src.models.train import load_training_config


def test_estimator_type_and_parameters():
    """Verify the expected implementation using the configured first grid candidate."""
    config = load_training_config()
    spec = config['models']['catboost']
    params = {**spec['fixed'], **{key: values[0] for key, values in spec['grid'].items()}}
    plugin = ALGORITHM_REGISTRY.get('catboost')
    model = plugin.build(params, config['seed'])
    assert isinstance(model, CatBoostClassifier)
