"""Validate the Optuna YAML contract without executing model optimization."""

import copy
from pathlib import Path

import optuna
import pytest
import yaml

from src.models.search.configuration import validate_distribution, validate_optuna_config


@pytest.fixture
def optuna_config():
    """Load a fresh copy of the checked-in training configuration."""
    path = Path(__file__).resolve().parents[1] / "configs/config.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_default_spaces_and_budgets(optuna_config):
    """Keep the approved per-family budget and the existing grid unchanged."""
    before = copy.deepcopy(optuna_config)
    validate_optuna_config(optuna_config)
    budgets = {name: spec["optuna"]["trials"] for name, spec in optuna_config["models"].items()
               if "optuna" in spec}
    assert budgets == {"hist_gradient_boosting": 50, "xgboost": 100,
                       "catboost": 100, "lightgbm": 100}
    assert sum(budgets.values()) == 350
    assert optuna_config == before


def test_legacy_yaml_is_supported(optuna_config):
    """Accept old configuration files without Optuna settings."""
    optuna_config.pop("optuna")
    for spec in optuna_config["models"].values():
        spec.pop("optuna", None)
    validate_optuna_config(optuna_config)


def test_missing_global_settings(optuna_config):
    """Reject orphan per-family search spaces."""
    optuna_config.pop("optuna")
    with pytest.raises(ValueError, match="global settings"):
        validate_optuna_config(optuna_config)


def test_narrowed_scope_preserves_search_config(optuna_config):
    """Permit a YAML scope containing only one family and a subset of variants."""
    optuna_config["models"] = {"lightgbm": optuna_config["models"]["lightgbm"]}
    optuna_config["variants"] = ["baseline"]
    validate_optuna_config(optuna_config)


@pytest.mark.parametrize("spec", [
    {"type": "int", "low": 1, "high": 5},
    {"type": "float", "low": 0.005, "high": 0.2, "log": True},
    {"type": "categorical", "choices": [None, "balanced"]},
])
def test_valid_distributions(spec):
    """Accept the three supported distribution types."""
    validate_distribution(spec, "space.parameter")


@pytest.mark.parametrize("spec", [
    None, [], {}, {"type": "unknown"},
    {"type": "int", "low": True, "high": 5},
    {"type": "int", "low": 1.5, "high": 5},
    {"type": "int", "low": 5, "high": 2},
    {"type": "int", "low": 0, "high": 5, "log": True},
    {"type": "float", "low": 0.1, "high": float("inf")},
    {"type": "float", "low": float("nan"), "high": 1.0},
    {"type": "float", "low": 0.1, "high": 1.0, "log": "true"},
    {"type": "float", "low": 0.1, "high": 1.0, "step": 0.1},
    {"type": "categorical", "choices": []},
    {"type": "categorical", "choices": [True, 1]},
    {"type": "categorical", "choices": ["balanced", "balanced"]},
    {"type": "categorical", "choices": [[1, 2]]},
    {"type": "categorical", "choices": [float("nan")]},
])
def test_invalid_distributions(spec):
    """Reject invalid types, bounds, log flags and ambiguous choices."""
    with pytest.raises(ValueError, match="space.parameter"):
        validate_distribution(spec, "space.parameter")


@pytest.mark.parametrize("key,value", [
    ("sampler", "random"), ("n_startup_trials", -1),
    ("n_startup_trials", True), ("storage_path", ""),
    ("storage_path", "file.json"), ("unknown", True),
])
def test_invalid_global_settings(optuna_config, key, value):
    """Reject unsupported sampler, invalid storage and malformed startup settings."""
    optuna_config["optuna"][key] = value
    with pytest.raises(ValueError, match="optuna"):
        validate_optuna_config(optuna_config)


@pytest.mark.parametrize("key,value", [
    ("trials", 0), ("trials", True), ("space", {}),
    ("space", {"random_state": {"type": "int", "low": 1, "high": 10}}),
    ("space", {"variant": {"type": "categorical", "choices": ["baseline"]}}),
    ("unknown", True), ("enqueue", "invalid"),
])
def test_invalid_family_settings(optuna_config, key, value):
    """Reject malformed per-family settings and reserved estimator parameters."""
    optuna_config["models"]["lightgbm"]["optuna"][key] = value
    with pytest.raises(ValueError, match="models.lightgbm.optuna"):
        validate_optuna_config(optuna_config)


@pytest.mark.parametrize("change", ["missing", "extra", "bounds", "category", "variant", "duplicate", "too_many"])
def test_invalid_queued_configuration(optuna_config, change):
    """Reject partial, duplicate or incompatible warm-start configurations."""
    section = optuna_config["models"]["lightgbm"]["optuna"]
    entry = section["enqueue"][0]
    if change == "missing":
        entry["params"].pop("learning_rate")
    elif change == "extra":
        entry["params"]["unexpected"] = 1
    elif change == "bounds":
        entry["params"]["learning_rate"] = 1.0
    elif change == "category":
        entry["params"]["class_weight"] = "unsupported"
    elif change == "variant":
        entry["variant"] = "unsupported"
    elif change == "duplicate":
        section["enqueue"].append(copy.deepcopy(entry))
    elif change == "too_many":
        section["trials"] = 1
        section["enqueue"].append(copy.deepcopy(entry))
    with pytest.raises(ValueError, match="enqueue"):
        validate_optuna_config(optuna_config)


@pytest.mark.parametrize("family", ["lightgbm", "xgboost", "catboost", "hist_gradient_boosting"])
def test_distributions_and_queued_values_work_with_optuna(optuna_config, tmp_path, family):
    """Exercise real Optuna distributions, queued starts and SQLite persistence."""
    spec = optuna_config["models"][family]["optuna"]
    settings = optuna_config["optuna"]
    storage = f"sqlite:///{(tmp_path / 'study.db').as_posix()}"
    study = optuna.create_study(study_name=family, direction="maximize", storage=storage,
        sampler=optuna.samplers.TPESampler(seed=optuna_config["seed"],
                                          n_startup_trials=settings["n_startup_trials"]))
    queued = spec["enqueue"][0]
    study.enqueue_trial({"variant": queued["variant"], **queued["params"]})
    trial = study.ask()
    assert trial.suggest_categorical("variant", optuna_config["variants"]) == queued["variant"]
    for name, distribution in spec["space"].items():
        # Match the declared distributions with the actual installed Optuna API.
        kwargs = {key: value for key, value in distribution.items() if key != "type"}
        if distribution["type"] == "int":
            value = trial.suggest_int(name, **kwargs)
        elif distribution["type"] == "float":
            value = trial.suggest_float(name, **kwargs)
        else:
            value = trial.suggest_categorical(name, **kwargs)
        assert value == queued["params"][name]
    study.tell(trial, 0.5)  # Synthetic objective; no model fits or data access.
    reloaded = optuna.load_study(study_name=family, storage=storage)
    assert reloaded.best_trial.params == {"variant": queued["variant"], **queued["params"]}


def test_training_loader_validates_optuna_schema(optuna_config, tmp_path):
    """Check training YAML integration and preserve the current grid count."""
    from src.models import train as training

    path = tmp_path / "training.yml"
    path.write_text(yaml.safe_dump(optuna_config, sort_keys=False), encoding="utf-8")
    config = training.load_training_config(path)
    assert len(training.candidates(True, config=config)) == 7
    assert len(training.candidates(False, config=config)) == 42
    optuna_config["models"]["lightgbm"]["optuna"]["trials"] = -1
    path.write_text(yaml.safe_dump(optuna_config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="models.lightgbm.optuna.trials"):
        training.load_training_config(path)
