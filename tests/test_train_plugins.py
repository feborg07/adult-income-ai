"""Integration tests for registry-driven model training."""

import copy
import logging
from argparse import Namespace
from unittest.mock import Mock

import mlflow
import pytest
import yaml
from sklearn.linear_model import LogisticRegression

from src.models import train as training
from src.models.algorithms.base import BaseAlgorithm, BINARY_CLASSIFICATION
from src.models.algorithms.registry import AlgorithmRegistry
from src.models.tasks import binary_classification as binary_task

# Reuse the existing synthetic workspace and isolated MLflow fixtures.
from tests.test_train import (
    isolated_mlflow,
    run_training,
    training_workspace,
)


class ProbeAlgorithm(BaseAlgorithm):
    """Test-only plugin with a previously unknown registry identifier."""

    name = "probe"
    supported_tasks = (BINARY_CLASSIFICATION,)

    def build(self, params, seed):
        """Construct a cloneable estimator from supplied parameters."""
        return LogisticRegression(**dict(params), random_state=seed)

    def log_model(self, model):
        """Provide the logging contract without creating test artifacts."""
        raise NotImplementedError("The probe is used for construction only.")


def test_registry_drives_configuration_cli_and_candidates(monkeypatch, tmp_path):
    """Verify that a registered plugin works without engine-specific branches."""
    config = copy.deepcopy(training.load_training_config())
    config["models"] = {
        "probe": {"fixed": {"max_iter": 50}, "grid": {"C": [0.5, 2.0]}}
    }
    path = tmp_path / "probe.yml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    # Use a temporary registry so production registration is not modified.
    registry = AlgorithmRegistry()
    plugin = ProbeAlgorithm()
    registry.register(plugin)
    build = Mock(wraps=plugin.build)
    monkeypatch.setattr(plugin, "build", build)
    monkeypatch.setattr(training, "ALGORITHM_REGISTRY", registry)
    monkeypatch.setattr(
        "sys.argv", ["train", "--config", str(path), "--quick",
                     "--models", "probe", "--search", "grid"]
    )

    args = training.parse_args()
    effective, search = training.resolve_training_scope(
        training.load_training_config(path), args
    )
    candidates = training.candidates(search == "first", config=effective)

    assert [family for family, _, _ in candidates] == ["probe", "probe"]
    assert [params for _, params, _ in candidates] == [{"C": 0.5}, {"C": 2.0}]
    assert [model.C for _, _, model in candidates] == [0.5, 2.0]
    assert all(model.max_iter == 50 for _, _, model in candidates)
    assert all(model.random_state == config["seed"] for _, _, model in candidates)
    assert build.call_count == 2
    build.assert_any_call({"max_iter": 50, "C": 0.5}, config["seed"])
    build.assert_any_call({"max_iter": 50, "C": 2.0}, config["seed"])


def test_existing_evaluation_interfaces_are_preserved():
    """Verify that legacy callers use the extracted task implementations."""
    assert training.evaluate is binary_task.evaluate
    assert training.optimal_threshold is binary_task.optimal_threshold
    assert training.rank is binary_task.rank


def test_quick_training_dispatches_all_registered_plugins(
    training_workspace, isolated_mlflow, monkeypatch
):
    """Train each plugin and exercise its actual MLflow logging interface."""
    args = training_workspace["args"]
    args.variants = ["baseline"]
    args.search = "first"

    # Spy on the real plugin methods without replacing their behavior.
    families = list(training.load_training_config()["models"])
    loggers = {}
    for family in families:
        plugin = training.ALGORITHM_REGISTRY.get(
            family, task_type=BINARY_CLASSIFICATION
        )
        logger = Mock(wraps=plugin.log_model)
        monkeypatch.setattr(plugin, "log_model", logger)
        loggers[family] = logger

    run_dir, summary = run_training(training_workspace, quick=True)

    assert summary["candidate_count"] == len(families)
    assert summary["successful_candidates"] == len(families)
    assert set(summary["winners"]) == {"baseline"}
    assert "test" not in summary["winners"]["baseline"]
    assert not (run_dir / "best_model_pipeline.joblib").exists()
    assert not (run_dir / "baseline").exists()

    for logger in loggers.values():
        logger.assert_called_once()
        fitted_model = logger.call_args.args[0]
        assert set(fitted_model.classes_) == {0, 1}

    # Confirm that every plugin created a tracked candidate run.
    experiment = mlflow.get_experiment_by_name(args.experiment)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == len(families)
    assert set(runs["tags.model_family"]) == set(families)
    assert set(runs["tags.training_execution_id"]) == {summary["execution_id"]}
    assert runs["metrics.val_f1"].notna().all()


@pytest.mark.parametrize("quick, expected", [
    (True, "Validation Training Successful"),
    (False, "Full Training Completed"),
])
def test_cli_reports_training_completion(monkeypatch, caplog, quick, expected):
    """Report completion only after the training workflow returns successfully."""
    args = Namespace(quick=quick, show_mlflow_command=False)
    runner = Mock()
    monkeypatch.setattr(training, "parse_args", lambda: args)
    monkeypatch.setattr(training, "train", runner)

    with caplog.at_level(logging.INFO, logger="training"):
        training.main()

    runner.assert_called_once_with(args)
    assert expected in caplog.messages
    assert "Quick Training Successful" not in caplog.messages
    assert not any("Training Failed" in message for message in caplog.messages)


@pytest.mark.parametrize("quick, mode", [(True, "Validation"), (False, "Full")])
def test_cli_reports_training_failure(monkeypatch, caplog, quick, mode):
    """Report failure and propagate the exception so the CLI exits nonzero."""
    args = Namespace(quick=quick, show_mlflow_command=False)
    error = RuntimeError("intentional training failure")
    runner = Mock(side_effect=error)
    monkeypatch.setattr(training, "parse_args", lambda: args)
    monkeypatch.setattr(training, "train", runner)

    with caplog.at_level(logging.INFO, logger="training"):
        with pytest.raises(RuntimeError, match="intentional training failure") as caught:
            training.main()

    assert caught.value is error
    runner.assert_called_once_with(args)
    assert f"{mode} Training Failed: intentional training failure" in caplog.messages
    assert "Validation Training Successful" not in caplog.messages
    assert "Full Training Completed" not in caplog.messages


@pytest.mark.parametrize("options", [
    ["--quick"],
    ["--mode", "validation", "--search", "first"],
    ["--mode", "validation", "--search", "grid"],
    ["--mode", "validation", "--evaluation", "cv", "--search", "optuna",
     "--models", "xgboost", "--trials", "100"],
])
def test_validation_status_independent_of_search(monkeypatch, caplog, options):
    """Exercise real CLI parsing and verify lifecycle wording for all searches."""
    monkeypatch.setattr("sys.argv", ["train", *options])
    runner = Mock()
    monkeypatch.setattr(training, "train", runner)

    # The real parser preserves mode and search options; no models are fitted.
    with caplog.at_level(logging.INFO, logger="training"):
        training.main()

    runner.assert_called_once()
    args = runner.call_args.args[0]
    assert args.mode == "validation"
    assert args.quick is True
    if "--search" in options:
        assert args.search == options[options.index("--search") + 1]
    if "--trials" in options:
        assert args.trials == 100
    assert "Validation Training Successful" in caplog.messages
    assert "Quick Training Successful" not in caplog.messages


def test_cli_show_mlflow_command_does_not_report_training(
    monkeypatch, caplog, capsys, tmp_path
):
    """Print the MLflow command without starting or reporting training."""
    args = Namespace(
        quick=True, show_mlflow_command=True, mlflow_db=tmp_path / "mlflow.db"
    )
    runner = Mock()
    monkeypatch.setattr(training, "parse_args", lambda: args)
    monkeypatch.setattr(training, "train", runner)

    with caplog.at_level(logging.INFO, logger="training"):
        training.main()

    runner.assert_not_called()
    assert "mlflow ui --backend-store-uri" in capsys.readouterr().out
    assert not any("Training" in message for message in caplog.messages)
