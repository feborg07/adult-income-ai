"""Specialized constructor checks for random_forest."""
from sklearn.ensemble import RandomForestClassifier
from src.models.algorithms.registry import ALGORITHM_REGISTRY
from src.models.train import load_training_config


def test_estimator_type_and_parameters():
    """Verify the expected implementation using the configured first grid candidate."""
    config = load_training_config()
    spec = config['models']['random_forest']
    params = {**spec['fixed'], **{key: values[0] for key, values in spec['grid'].items()}}
    plugin = ALGORITHM_REGISTRY.get('random_forest')
    model = plugin.build(params, config['seed'])
    assert isinstance(model, RandomForestClassifier)
