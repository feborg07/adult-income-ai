"""Verify persistent study identity, cumulative reporting and tracking links."""

import json
from pathlib import Path
from unittest.mock import Mock

import optuna
import pandas as pd
import pytest

from src.models import train as training
from src.models.search.persistence import exclusive_study_lock
from tests.test_train_cv import cv_workspace, tracked_mlflow
from tests.test_train_optuna import optuna_workspace


def run_and_read(args):
    """Execute training and read only the newly generated execution reports."""
    before = set(args.output_dir.glob("*"))
    training.train(args)
    run_dir, = set(args.output_dir.glob("*")) - before
    summary = json.loads((run_dir / "cv_training_summary.json").read_text())
    current = pd.read_csv(run_dir / "cv_leaderboard.csv")
    cumulative = pd.read_csv(run_dir / "optuna_study_leaderboard.csv")
    return summary, current, cumulative


def use_hgb(args):
    """Select a fast, single-family CV experiment for persistence tests."""
    args.search, args.models, args.trials = "optuna", ["hist_gradient_boosting"], 1


def test_resume_and_no_op(optuna_workspace, tracked_mlflow, monkeypatch):
    """Preserve original trial identity and fit only the remaining cumulative budget."""
    args, _, reference = optuna_workspace
    use_hgb(args)
    before = reference.read_bytes()
    spy = Mock(wraps=training.cv.evaluate_cv_candidate)
    monkeypatch.setattr(training.cv, "evaluate_cv_candidate", spy)
    first, current, cumulative = run_and_read(args)
    assert len(current) == len(cumulative) == 1
    assert first["optuna_cumulative_successful_candidates"] == 1
    assert first["optuna_best_ranked_candidate"]["training_execution_id"] == first["execution_id"]
    assert spy.call_count == 1
    repeated, current, cumulative = run_and_read(args)
    assert len(current) == repeated["candidate_count"] == repeated["successful_candidates"] == 0
    assert repeated["preferred_variant_by_cv"] is None and repeated["winners"] == {}
    assert len(cumulative) == 1 and spy.call_count == 1
    assert repeated["optuna_context_id"] == first["optuna_context_id"]
    args.trials = 3
    resumed, current, cumulative = run_and_read(args)
    assert len(current) == resumed["candidate_count"] == 2 and spy.call_count == 3
    assert len(cumulative) == resumed["optuna_cumulative_successful_candidates"] == 3
    assert set(cumulative.training_execution_id) == {first["execution_id"], resumed["execution_id"]}
    assert set(cumulative.optuna_trial_number) == {0, 1, 2}
    study = resumed["optuna_studies"]["hist_gradient_boosting"]
    assert study["attempted_this_run"] == 2 and study["completed"] == 3
    assert study["best_ranked_candidate"] == resumed["optuna_best_ranked_candidate"]
    assert reference.read_bytes() == before


@pytest.mark.parametrize("change", ["cv", "tracking", "source", "variant"])
def test_incompatible_context_gets_new_study(optuna_workspace, tracked_mlflow, monkeypatch, change):
    """Never reuse completed trials when their evaluation or tracking context changes."""
    args, _, _ = optuna_workspace
    use_hgb(args)
    first, _, _ = run_and_read(args)
    if change == "cv":
        args.calibration_size = 0.25
    elif change == "tracking":
        args.experiment = "other-experiment"
    elif change == "source":
        def changed_identity():
            """Simulate a changed evaluator source fingerprint."""
            return {"source": "different"}
        monkeypatch.setattr(training, "optimization_code_identity", changed_identity)
    else:
        args.variants = ["interactions"]
    second, current, cumulative = run_and_read(args)
    assert len(current) == len(cumulative) == 1
    assert second["optuna_context_id"] != first["optuna_context_id"]
    assert second["optuna_studies"]["hist_gradient_boosting"]["study_name"] != first["optuna_studies"]["hist_gradient_boosting"]["study_name"]


def test_storage_cannot_overlap_tracking_or_reference(optuna_workspace):
    """Reject SQLite output paths that could alter approved or tracking artifacts."""
    args, _, _ = optuna_workspace
    use_hgb(args)
    for path in (args.mlflow_db, training.ROOT / "models/reference/optuna.db", Path("not-a-db.json")):
        args.optuna_storage = path
        with pytest.raises(ValueError, match="SQLite|separate|different"):
            training.train(args)
        assert not args.output_dir.exists()


def test_storage_override_requires_optuna(monkeypatch):
    """Reject a study storage override for a non-adaptive search."""
    monkeypatch.setattr("sys.argv", ["train", "--quick", "--optuna-storage", "study.db"])
    with pytest.raises(SystemExit) as caught:
        training.parse_args()
    assert caught.value.code == 2


def test_exclusive_lock_released_on_error(tmp_path):
    """Prevent concurrent access and release the lock on an ordinary exception."""
    database = tmp_path / "study.db"
    with pytest.raises(ValueError, match="forced"):
        with exclusive_study_lock(database, "study"):
            with pytest.raises(RuntimeError, match="locked"):
                with exclusive_study_lock(database, "study"):
                    pytest.fail("Concurrent study acquired the lock")
            raise ValueError("forced")
    with exclusive_study_lock(database, "study"):
        assert len(list(tmp_path.glob("*.lock"))) == 1
    assert not list(tmp_path.glob("*.lock"))


def test_missing_trial_provenance_is_rejected(tmp_path):
    """Do not silently reconstruct rankings from completed trials without run links."""
    study = optuna.create_study(direction="maximize", storage=f"sqlite:///{tmp_path / 'orphan.db'}")
    trial = study.ask()
    study.tell(trial, 0.7)
    with pytest.raises(RuntimeError, match="tracking provenance"):
        training.completed_study_rows(study)


def test_optuna_real_mlflow_persistence(optuna_workspace):
    """Confirm SQLite trial records link to real finished MLflow runs and fold artifacts."""
    import mlflow
    from mlflow.tracking import MlflowClient

    args, _, reference = optuna_workspace
    use_hgb(args)
    before = reference.read_bytes()
    previous = mlflow.get_tracking_uri()
    try:
        first, _, _ = run_and_read(args)
        args.trials = 2
        second, current, cumulative = run_and_read(args)
        assert len(current) == 1 and len(cumulative) == 2
        name = second["optuna_studies"]["hist_gradient_boosting"]["study_name"]
        study = optuna.load_study(study_name=name, storage=f"sqlite:///{args.optuna_storage.as_posix()}")
        client = MlflowClient()
        assert len(study.trials) == 2
        for trial in study.trials:
            metadata = trial.user_attrs["evaluation_metadata"]
            run = client.get_run(metadata["run_id"])
            assert run.info.status == "FINISHED"
            assert run.data.tags["optuna_study_name"] == name
            assert run.data.tags["optuna_trial_number"] == str(trial.number)
            assert run.data.metrics["cv_f1"] == pytest.approx(trial.value)
            assert run.data.tags["training_execution_id"] == metadata["training_execution_id"]
            for fold in range(1, 4):
                artifacts = client.list_artifacts(run.info.run_id, f"cv/fold_{fold}")
                assert {Path(item.path).name for item in artifacts} == {"model_pipeline.joblib", "fold_result.json"}
        assert set(cumulative.training_execution_id) == {first["execution_id"], second["execution_id"]}
        assert reference.read_bytes() == before
    finally:
        if mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_tracking_uri(previous)
