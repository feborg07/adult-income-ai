"""Exercise Optuna CLI dispatch and the shared CV training lifecycle."""

import json
from argparse import Namespace
from unittest.mock import Mock

import pandas as pd
import pytest

from src.models import train as training
from tests.test_train_cv import cv_workspace, tracked_mlflow


@pytest.fixture
def optuna_workspace(cv_workspace, tmp_path):
    """Keep persistent study databases in pytest temporary directories."""
    args, _, _ = cv_workspace
    args.optuna_storage = tmp_path / "optuna.db"
    return cv_workspace


@pytest.mark.parametrize("options", [
    ["--mode", "validation", "--evaluation", "cv", "--search", "optuna"],
    ["--quick", "--evaluation", "cv", "--search", "optuna", "--trials", "2"],
])
def test_optuna_cli_accepts_validation(monkeypatch, options):
    """Enable adaptive search only in the CV validation lifecycle."""
    monkeypatch.setattr("sys.argv", ["train", *options])
    args = training.parse_args()
    config, search = training.resolve_training_scope(training.load_training_config(), args)
    assert args.quick and args.evaluation == "cv" and search == "optuna"
    assert set(config["models"]) == {"lightgbm", "xgboost", "catboost", "hist_gradient_boosting"}


@pytest.mark.parametrize("options", [
    ["--mode", "full", "--evaluation", "cv", "--search", "optuna"],
    ["--quick", "--search", "optuna"],
    ["--search", "optuna"],
    ["--quick", "--evaluation", "cv", "--search", "grid", "--trials", "2"],
    ["--quick", "--evaluation", "cv", "--search", "optuna", "--trials", "0"],
    ["--quick", "--evaluation", "cv", "--search", "optuna", "--trials", "-1"],
    ["--quick", "--evaluation", "cv", "--search", "optuna", "--models", "random_forest"],
    ["--quick", "--evaluation", "cv", "--search", "optuna", "--models", "lightgbm", "extra_trees"],
])
def test_invalid_optuna_cli(monkeypatch, options):
    """Reject unsupported modes, budgets and algorithm selections before fitting."""
    monkeypatch.setattr("sys.argv", ["train", *options])
    with pytest.raises(SystemExit) as caught:
        training.parse_args()
    assert caught.value.code == 2


def test_optuna_yaml_scope_and_budget_defaults(optuna_workspace, tracked_mlflow, monkeypatch):
    """Pass per-family YAML budgets and the narrowed scope into the CV runner."""
    args, _, _ = optuna_workspace
    args.search = "optuna"
    args.models = ["lightgbm", "hist_gradient_boosting"]
    runner = Mock()
    monkeypatch.setattr(training, "run_cv_search", runner)
    training.train(args)
    scope = runner.call_args.args[1]
    assert scope["trial_budgets"] == {"hist_gradient_boosting": 50, "lightgbm": 100}
    assert scope["variants"] == ["baseline"]
    args.trials = 2
    training.train(args)
    assert runner.call_args.args[1]["trial_budgets"] == {"hist_gradient_boosting": 2, "lightgbm": 2}


def test_optuna_legacy_config_rejected_without_space():
    """Keep legacy grids valid while rejecting adaptive search without its settings."""
    config = training.load_training_config()
    config.pop("optuna")
    for model in config["models"].values():
        model.pop("optuna", None)
    with pytest.raises(ValueError, match="missing from YAML"):
        training.resolve_training_scope(config, Namespace(quick=True, evaluation="cv", search="optuna"))


def test_optuna_training_uses_shared_cv(optuna_workspace, tracked_mlflow, monkeypatch):
    """Execute real LightGBM trials on synthetic raw data with isolated tracking."""
    args, _, reference = optuna_workspace
    tracker, _ = tracked_mlflow
    args.search, args.trials = "optuna", 2
    args.models, args.variants = ["lightgbm"], ["interactions"]
    before = reference.read_bytes()
    prepare = Mock(wraps=training.cv.prepare_cv_folds)
    monkeypatch.setattr(training.cv, "prepare_cv_folds", prepare)
    monkeypatch.setattr(training, "load_variant", Mock(side_effect=AssertionError("holdout accessed")))
    monkeypatch.setattr(training, "candidates", Mock(side_effect=AssertionError("grid enumerated")))
    training.train(args)
    prepare.assert_called_once()
    run_dir = next(args.output_dir.iterdir())
    rows = pd.read_csv(run_dir / "cv_leaderboard.csv")
    summary = json.loads((run_dir / "cv_training_summary.json").read_text())
    assert len(rows) == summary["candidate_count"] == summary["successful_candidates"] == 2
    assert summary["optuna_storage"] == str(args.optuna_storage)
    assert args.optuna_storage.is_file()
    assert summary["optuna_studies"]["lightgbm"]["completed"] == 2
    assert summary["scope"]["trial_budgets"] == {"lightgbm": 2}
    assert set(rows.optuna_trial_number) == {0, 1}
    assert rows.optuna_study_name.nunique() == 1
    assert set(rows.variant) == {"interactions"}
    assert rows.cv_f1.between(0, 1).all()
    assert rows.cv_f1.is_monotonic_decreasing
    assert "val_f1" not in rows and "optimal_threshold" not in rows
    warm = rows.loc[rows.optuna_trial_number == 0].iloc[0]
    assert json.loads(warm.parameters)["n_estimators"] == 500
    assert json.loads(warm.effective_parameters)["subsample_freq"] == 1
    assert tracker.start_run.call_count == 2
    assert tracker.log_artifact.call_count == 12
    for call in tracker.log_params.call_args_list:
        params = call.args[0]
        assert "model_min_child_samples" in params and "fixed_min_child_samples" not in params
        assert params["threshold_policy"] == "fold_fit_inner_calibration"
    for call in tracker.set_tags.call_args_list:
        tags = call.args[0]
        assert tags["search_strategy"] == "optuna"
        assert "optuna_study_name" in tags and "optuna_trial_number" in tags
    assert not (run_dir / "best_model_pipeline.joblib").exists()
    assert reference.read_bytes() == before


def test_optuna_candidate_error_is_not_hidden(optuna_workspace, tracked_mlflow, monkeypatch):
    """Stop on unexpected CV errors rather than publish a false success summary."""
    args, _, _ = optuna_workspace
    args.search, args.trials, args.models = "optuna", 1, ["hist_gradient_boosting"]
    monkeypatch.setattr(training.cv, "evaluate_cv_candidate", Mock(side_effect=RuntimeError("fit failed")))
    with pytest.raises(RuntimeError, match="fit failed"):
        training.train(args)
    assert not list(args.output_dir.rglob("cv_training_summary.json"))


def test_optuna_direct_call_rejects_holdout(optuna_workspace, tracked_mlflow):
    """Block unsafe programmatic calls before creating any training artifacts."""
    args, _, _ = optuna_workspace
    args.search, args.models, args.evaluation = "optuna", ["lightgbm"], "holdout"
    with pytest.raises(ValueError, match="Optuna requires"):
        training.train(args)
    assert not args.output_dir.exists()


def test_optuna_budget_is_per_family(optuna_workspace, tracked_mlflow):
    """Execute two independent studies with a one-trial budget for each family."""
    args, _, _ = optuna_workspace
    args.search, args.trials = "optuna", 1
    args.models, args.variants = ["lightgbm", "hist_gradient_boosting"], ["baseline", "interactions"]
    training.train(args)
    run_dir = next(args.output_dir.iterdir())
    summary = json.loads((run_dir / "cv_training_summary.json").read_text())
    rows = pd.read_csv(run_dir / "cv_leaderboard.csv")
    assert summary["candidate_count"] == 2
    assert rows.optuna_study_name.nunique() == 2
    assert set(rows.optuna_trial_number) == {0}
    assert set(rows.variant) == {"baseline", "interactions"}
    assert all(study["completed"] == 1 for study in summary["optuna_studies"].values())
