"""Verify YAML-controlled CV settings and backward-compatible CLI modes."""

import copy
import json
from unittest.mock import Mock

import pytest
import yaml

from src.models import train as training
from tests.test_train_cv import cv_workspace, tracked_mlflow


def write_config(tmp_path, settings):
    """Write an isolated YAML configuration with the requested CV section."""
    config = training.load_training_config()
    if settings is None:
        config.pop("cross_validation")
    else:
        config["cross_validation"] = settings
    path = tmp_path / "training.yml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.parametrize("settings, expected", [
    (None, {"n_splits": 3, "calibration_size": 0.20}),
    ({"n_splits": 4}, {"n_splits": 4, "calibration_size": 0.20}),
    ({"n_splits": 5, "calibration_size": 0.25}, {"n_splits": 5, "calibration_size": 0.25}),
])
def test_yaml_cv_settings(tmp_path, settings, expected):
    """Read custom YAML values and preserve defaults for older configurations."""
    path = write_config(tmp_path, settings)
    assert training.load_training_config(path)["cross_validation"] == expected


@pytest.mark.parametrize("section", [
    [], None, {"n_split": 3}, {"n_splits": True}, {"n_splits": 1},
    {"n_splits": 3.5}, {"n_splits": "3"}, {"calibration_size": True},
    {"calibration_size": 0}, {"calibration_size": 1},
    {"calibration_size": float("nan")}, {"calibration_size": float("inf")},
    {"calibration_size": "0.2"},
])
def test_invalid_yaml_cv_settings(tmp_path, section):
    """Reject invalid types, unknown keys, and out-of-range values."""
    path = write_config(tmp_path, {})
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["cross_validation"] = section
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="cross_validation"):
        training.load_training_config(path)


@pytest.mark.parametrize("options, mode, evaluation, search", [
    ([], "full", "holdout", "grid"),
    (["--mode", "full"], "full", "holdout", "grid"),
    (["--quick"], "validation", "holdout", "first"),
    (["--mode", "validation"], "validation", "holdout", "first"),
    (["--mode", "validation", "--evaluation", "cv", "--search", "grid"], "validation", "cv", "grid"),
])
def test_lifecycle_compatibility(monkeypatch, options, mode, evaluation, search):
    """Preserve existing defaults while allowing explicit lifecycle selection."""
    monkeypatch.setattr("sys.argv", ["train", *options])
    args = training.parse_args()
    assert args.mode == mode and args.quick == (mode == "validation")
    assert args.evaluation == evaluation
    assert training.resolve_training_scope(training.load_training_config(), args)[1] == search


@pytest.mark.parametrize("options", [
    ["--quick", "--mode", "full"],
    ["--quick", "--mode", "validation"],
    ["--mode", "full", "--evaluation", "cv"],
    ["--evaluation", "cv"],
    ["--quick", "--cv-folds", "4"],
    ["--calibration-size", "0.2"],
    ["--quick", "--evaluation", "cv", "--cv-folds", "1"],
    ["--quick", "--evaluation", "cv", "--calibration-size", "nan"],
])
def test_invalid_cli_combinations(monkeypatch, options):
    """Reject unsafe lifecycle combinations and meaningless CV overrides."""
    monkeypatch.setattr("sys.argv", ["train", *options])
    with pytest.raises(SystemExit) as caught:
        training.parse_args()
    assert caught.value.code == 2


def test_cli_override_precedence(monkeypatch, tmp_path):
    """Apply CLI overrides without mutating YAML or its loaded configuration."""
    path = write_config(tmp_path, {"n_splits": 4, "calibration_size": 0.25})
    config = training.load_training_config(path)
    original = copy.deepcopy(config)
    monkeypatch.setattr("sys.argv", ["train", "--config", str(path), "--mode", "validation",
                                     "--evaluation", "cv", "--cv-folds", "2"])
    args = training.parse_args()
    assert training.resolve_cv_settings(config, args) == {"n_splits": 2, "calibration_size": 0.25}
    assert config == original
    assert training.load_training_config(path) == original


@pytest.mark.parametrize("overrides, expected", [
    ({}, {"n_splits": 2, "calibration_size": 0.25}),
    ({"cv_folds": 3, "calibration_size": 0.30}, {"n_splits": 3, "calibration_size": 0.30}),
])
def test_effective_settings_reach_training_and_tracking(
    cv_workspace, tracked_mlflow, monkeypatch, tmp_path, overrides, expected,
):
    """Fit actual synthetic CV folds and verify effective settings in artifacts."""
    args, _, _ = cv_workspace
    tracker, artifacts = tracked_mlflow
    args.config = write_config(tmp_path, {"n_splits": 2, "calibration_size": 0.25})
    for key, value in overrides.items():
        setattr(args, key, value)
    prepare = Mock(wraps=training.cv.prepare_cv_folds)
    monkeypatch.setattr(training.cv, "prepare_cv_folds", prepare)
    training.train(args)
    for key, value in expected.items():
        assert prepare.call_args.kwargs[key] == value
        assert tracker.log_params.call_args.args[0][key] == value
    run_dir = next(args.output_dir.iterdir())
    summary = json.loads((run_dir / "cv_training_summary.json").read_text())
    assert {key: summary[key] for key in expected} == expected
    assert summary["config_sha256"] == training.configuration_digest(training.load_training_config(args.config))
    assert len(list((artifacts / "cv").glob("fold_*"))) == expected["n_splits"]
    assert not (run_dir / "best_model_pipeline.joblib").exists()


def test_invalid_direct_training_override_fails_before_output(cv_workspace):
    """Apply validation to programmatic callers before tracking or artifact creation."""
    args, _, _ = cv_workspace
    args.cv_folds = 1
    with pytest.raises(ValueError, match="n_splits"):
        training.train(args)
    assert not args.output_dir.exists()
