"""Verify the CLI and plugin discovery without evaluating any held-out data."""
from copy import deepcopy
from importlib import import_module

import pytest
import yaml

from src.models import train as training
from src.models.algorithms.discovery import discover_plugins
from src.models.configuration import configuration_fingerprints
from tests.test_train_cv import cv_workspace, tracked_mlflow
from tests.test_train_optuna import optuna_workspace


def test_experiment_override_through_cli(tmp_path, monkeypatch):
    """Resolve one override without changing the base YAML or MLflow name."""
    base = training.load_training_config()
    model = 'xgboost'
    override = tmp_path / 'override.yml'
    override.write_text(yaml.safe_dump({'models': {model: {'grid': {'max_depth': [4]}}}}), encoding='utf-8')
    monkeypatch.setattr('sys.argv', ['train', '--quick', '--models', model,
        '--experiment-config', str(override), '--experiment', 'synthetic-test'])
    args = training.parse_args()
    assert args.experiment == 'synthetic-test'
    resolved = training.load_training_config(args.config, experiment_path=args.experiment_config)
    assert resolved['models'][model]['grid']['max_depth'] == [4]
    assert training.load_training_config() == base
    assert all(resolved['models'][name] == spec for name, spec in base['models'].items() if name != model)


def test_new_plugin_discovered_without_registry_changes(tmp_path, monkeypatch):
    """Discover an extra local cartridge through the existing generic interface."""
    package = tmp_path / 'synthetic_cartridges'
    package.mkdir()
    (package / '__init__.py').write_text('"""Temporary synthetic plugin package."""\n', encoding='utf-8')
    (package / 'probe.py').write_text('''"""Synthetic plugin for automatic discovery."""
from src.models.algorithms.base import BaseAlgorithm, BINARY_CLASSIFICATION

class Probe(BaseAlgorithm):
    """Implement the plugin contract without a registry change."""
    name = 'probe'
    supported_tasks = (BINARY_CLASSIFICATION,)

    def build(self, params, seed):
        """Return a placeholder; this test verifies discovery only."""
        return object()

    def log_model(self, model):
        """Reject unintended logging during discovery."""
        raise AssertionError('unexpected logging')
''', encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))
    plugins = discover_plugins(import_module('synthetic_cartridges'))
    assert [plugin.name for plugin in plugins] == ['probe']


def test_new_model_does_not_change_existing_configuration_ids():
    """Adding an unrelated cartridge leaves existing configuration IDs stable."""
    base = training.load_training_config()
    extra = deepcopy(base)
    extra['models']['probe'] = {'fixed': {}, 'grid': {'C': [1]}}
    original, updated = map(configuration_fingerprints, (base, extra))
    assert original['complete_sha256'] != updated['complete_sha256']
    assert all(original['study_configuration_sha256'][name] == updated['study_configuration_sha256'][name]
               for name in base['models'])


def test_real_fit_with_all_new_optuna_cartridges(optuna_workspace, tracked_mlflow, tmp_path):
    """Fit each remaining family through the real CV/SQLite lifecycle on synthetic data."""
    import json
    from tests.test_optuna_persistence import run_and_read

    args, _, reference = optuna_workspace
    args.models = ["random_forest", "extra_trees", "logistic_regression"]
    args.search = "optuna"
    args.trials = 1
    args.variants = ["baseline", "robust", "interactions"]
    config = training.load_training_config()
    for name in ("random_forest", "extra_trees"):
        config["models"][name]["fixed"]["n_jobs"] = 2
    args.config = tmp_path / "synthetic.yml"
    args.config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    sentinel = reference.read_bytes()
    summary, current, _ = run_and_read(args)
    assert set(current.model) == set(args.models)
    assert summary["candidate_count"] == 3
    assert len(summary["optuna_contexts"]) == 3
    assert reference.read_bytes() == sentinel
    run_dir = args.output_dir / summary["execution_id"]
    fingerprints = json.loads((run_dir / "configuration_fingerprints.json").read_text())
    assert fingerprints["complete_sha256"] == summary["config_sha256"]
    assert fingerprints["scope"]["models"] == ["logistic_regression", "random_forest", "extra_trees"]
    assert yaml.safe_load((run_dir / "resolved_config.yml").read_text()) == config
