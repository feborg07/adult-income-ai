"""Test training-engine cross-validation dispatch and held-out isolation."""

import json
import shutil
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data.preprocess import (
    PreprocessingConfig, encode_target, save_config, split_features_target,
)
from src.models import train as training


@pytest.fixture
def cv_workspace(tmp_path, sample_adult_df):
    """Create only raw training inputs and a reference sentinel, not test data."""
    batches = []
    for number in range(12):
        batch = sample_adult_df.copy()
        batch["fnlwgt"] += number * 1_000_000
        # Keep groups distinct even when census weight and demographics are excluded.
        batch["age"] += number
        batches.append(batch)
    X, y = split_features_target(encode_target(pd.concat(batches, ignore_index=True)))
    args = Namespace(
        processed_dir=tmp_path / "processed", preprocessor_dir=tmp_path / "preprocessors",
        output_dir=tmp_path / "models/runs", mlflow_db=tmp_path / "tracking.db",
        mlflow_artifacts=tmp_path / "mlartifacts", experiment="cv-test",
        quick=True, evaluation="cv", models=["logistic_regression"],
        variants=["baseline"], search="first",
    )
    for variant in ("baseline", "robust", "interactions"):
        folder = args.processed_dir / variant
        folder.mkdir(parents=True)
        X.assign(income=y).to_csv(folder / "raw_train.csv", index=False)
        save_config(PreprocessingConfig(interaction_features=variant == "interactions",
                                        numerical_scaling="robust" if variant == "robust" else "standard",
                                        capital_transform="both" if variant == "robust" else "log1p"),
                    folder / "preprocessing_config.json")
    reference = tmp_path / "models/reference/best_model_pipeline.joblib"
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b"approved reference sentinel")
    return args, X, reference


@pytest.fixture
def tracked_mlflow(monkeypatch, tmp_path):
    """Replace the tracking boundary while copying actual serialized artifacts."""
    tracking = MagicMock()
    run = SimpleNamespace(info=SimpleNamespace(run_id="synthetic-run"))
    tracking.start_run.return_value.__enter__.return_value = run
    destination = tmp_path / "logged_artifacts"

    def copy_artifact(filename, artifact_path):
        """Copy the artifact while the temporary source file is still available."""
        folder = destination / artifact_path
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(filename, folder / Path(filename).name)

    tracking.log_artifact.side_effect = copy_artifact
    monkeypatch.setattr(training, "mlflow", tracking)
    monkeypatch.setattr(training, "configure_mlflow", Mock())
    return tracking, destination


def test_cv_training_logs_folds_without_opening_holdout(cv_workspace, tracked_mlflow, monkeypatch):
    """Run three fitted folds through the real registry and verify research artifacts."""
    args, X, reference = cv_workspace
    tracker, artifacts = tracked_mlflow
    before = reference.read_bytes()
    # Any attempt to load legacy processed matrices is an error in CV mode.
    monkeypatch.setattr(training, "load_variant", Mock(side_effect=AssertionError("holdout accessed")))
    training.train(args)
    run_dir = next(args.output_dir.iterdir())
    summary = json.loads((run_dir / "cv_training_summary.json").read_text())
    row = pd.read_csv(run_dir / "cv_leaderboard.csv").iloc[0]
    assert summary["evaluation_protocol"] == "cv"
    assert summary["candidate_count"] == summary["successful_candidates"] == 1
    assert summary["preferred_variant_by_cv"] == "baseline"
    assert 0 <= row["cv_f1"] <= 1 and row["cv_f1_std"] >= 0
    assert "val_f1" not in row.index and "optimal_threshold" not in row.index
    assert "val_f1" not in tracker.log_metrics.call_args.args[0]
    assert not (run_dir / "best_model_pipeline.joblib").exists()
    assert reference.read_bytes() == before
    assert tracker.log_artifact.call_count == 6
    # Check that each fold artifact accepts raw features after serialization.
    for number in range(1, 4):
        folder = artifacts / "cv" / f"fold_{number}"
        pipeline = joblib.load(folder / "model_pipeline.joblib")
        probabilities = pipeline.predict_proba(X.iloc[:5])
        assert probabilities.shape == (5, 2)
        assert np.isfinite(probabilities).all()
        result = json.loads((folder / "fold_result.json").read_text())
        assert result["number"] == number and 0 <= result["threshold"] <= 1


def test_cv_grid_prepares_folds_once_per_variant(cv_workspace, tracked_mlflow, monkeypatch):
    """Reuse preprocessed folds across the complete selected family grid."""
    args, _, _ = cv_workspace
    args.search = "grid"
    prepare = Mock(wraps=training.cv.prepare_cv_folds)
    monkeypatch.setattr(training.cv, "prepare_cv_folds", prepare)
    training.train(args)
    prepare.assert_called_once()
    summary = json.loads((next(args.output_dir.iterdir()) / "cv_training_summary.json").read_text())
    assert summary["candidate_count"] == 6


def test_cv_rejects_full_before_creating_artifacts(cv_workspace):
    """Prevent a CV request from entering the test-evaluation lifecycle."""
    args, _, _ = cv_workspace
    args.quick = False
    with pytest.raises(ValueError, match="requires --quick"):
        training.train(args)
    assert not args.output_dir.exists()


def test_cv_failure_does_not_publish_success_summary(cv_workspace, tracked_mlflow, monkeypatch):
    """Propagate candidate failures instead of presenting a successful CV result."""
    args, _, _ = cv_workspace
    monkeypatch.setattr(training.cv, "evaluate_cv_candidate", Mock(side_effect=RuntimeError("fit failed")))
    with pytest.raises(RuntimeError, match="fit failed"):
        training.train(args)
    assert not list(args.output_dir.rglob("cv_training_summary.json"))


def test_holdout_quick_load_does_not_require_test_csvs(tmp_path):
    """Load only processed train and validation data when test access is forbidden."""
    processed, preprocessors = tmp_path / "processed", tmp_path / "preprocessors"
    folder = processed / "baseline"
    folder.mkdir(parents=True)
    preprocessors.mkdir()
    X = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    for split in ("train", "val"):
        X.to_csv(folder / f"X_{split}.csv", index=False)
        pd.DataFrame({"income": [0, 1, 0, 1]}).to_csv(folder / f"y_{split}.csv", index=False)
    joblib.dump(StandardScaler().fit(X), preprocessors / "baseline.joblib")
    result = training.load_variant(processed, preprocessors, "baseline", include_test=False)
    assert result.X_test is result.y_test is None
    with pytest.raises(FileNotFoundError, match="X_test.csv"):
        training.load_variant(processed, preprocessors, "baseline")


def test_cli_accepts_cv_only_in_quick_mode(monkeypatch):
    """Keep the existing CLI default and reject CV without validation-only mode."""
    monkeypatch.setattr("sys.argv", ["train", "--quick", "--evaluation", "cv"])
    assert training.parse_args().evaluation == "cv"
    monkeypatch.setattr("sys.argv", ["train", "--evaluation", "cv"])
    with pytest.raises(SystemExit) as caught:
        training.parse_args()
    assert caught.value.code == 2
    monkeypatch.setattr("sys.argv", ["train", "--quick"])
    assert training.parse_args().evaluation == "holdout"


def test_cv_real_mlflow_tracking(cv_workspace):
    """Verify actual SQLite tracking, CV metrics and fold artifacts."""
    import mlflow
    from mlflow.tracking import MlflowClient

    args, _, _ = cv_workspace
    previous_uri = mlflow.get_tracking_uri()
    try:
        training.train(args)
        experiment = mlflow.get_experiment_by_name(args.experiment)
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        assert len(runs) == 1
        row = runs.iloc[0]
        assert row["status"] == "FINISHED"
        assert row["tags.evaluation_protocol"] == "cv"
        assert 0 <= row["metrics.cv_f1"] <= 1
        assert "metrics.val_f1" not in runs.columns
        client = MlflowClient()
        for number in range(1, 4):
            paths = {artifact.path for artifact in client.list_artifacts(row["run_id"], f"cv/fold_{number}")}
            assert paths == {
                f"cv/fold_{number}/model_pipeline.joblib", f"cv/fold_{number}/fold_result.json",
            }
    finally:
        if mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_tracking_uri(previous_uri)


def test_cv_rejects_inconsistent_raw_partitions(cv_workspace, tracked_mlflow):
    """Reject different raw training rows across variants before any model fit."""
    args, _, _ = cv_workspace
    tracker, _ = tracked_mlflow
    args.variants = ["baseline", "interactions"]
    path = args.processed_dir / "interactions/raw_train.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Raw training partitions differ"):
        training.train(args)
    tracker.start_run.assert_not_called()
