"""Unit tests for the MLflow experiment comparison helper."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import compare_experiments as comparison


def test_rank_rejects_failed_and_final_test_runs():
    """Rank rejects failed and final test runs."""
    runs = pd.DataFrame([
        {"run_id": "candidate", "status": "FINISHED", "tags.model_family": "random_forest", "tags.preprocessing_variant": "baseline", "tags.mode": "quick", "metrics.val_f1": 0.72},
        {"run_id": "failed", "status": "FAILED", "tags.model_family": "random_forest", "tags.preprocessing_variant": "baseline", "tags.mode": "quick", "metrics.val_f1": 0.99},
        {"run_id": "final-test", "status": "FINISHED", "tags.model_family": None, "tags.preprocessing_variant": "baseline", "tags.mode": None, "metrics.val_f1": 0.98},
    ])
    ranked = comparison.rank_candidates(runs)
    assert ranked["run_id"].tolist() == ["candidate"]


def test_rank_uses_validation_f1_and_tie_breakers():
    """Rank uses validation f1 and tie breakers."""
    runs = pd.DataFrame([
        {"run_id": "a", "status": "FINISHED", "tags.model_family": "extra_trees", "tags.preprocessing_variant": "baseline", "metrics.val_f1": "0.71", "metrics.val_pr_auc": 0.8, "metrics.val_roc_auc": 0.9},
        {"run_id": "b", "status": "FINISHED", "tags.model_family": "random_forest", "tags.preprocessing_variant": "baseline", "metrics.val_f1": 0.72, "metrics.val_pr_auc": 0.75, "metrics.val_roc_auc": 0.9},
        {"run_id": "c", "status": "FINISHED", "tags.model_family": "logistic_regression", "tags.preprocessing_variant": "baseline", "metrics.val_f1": 0.72, "metrics.val_pr_auc": 0.82, "metrics.val_roc_auc": 0.9},
    ])
    assert comparison.rank_candidates(runs)["run_id"].tolist() == ["c", "b", "a"]


def test_rank_filters_training_mode_and_variant():
    """Rank filters training mode and variant."""
    runs = pd.DataFrame([
        {"run_id": "a", "status": "FINISHED", "tags.model_family": "extra_trees", "tags.preprocessing_variant": "baseline", "tags.mode": "quick", "metrics.val_f1": 0.6},
        {"run_id": "b", "status": "FINISHED", "tags.model_family": "random_forest", "tags.preprocessing_variant": "robust", "tags.mode": "full", "metrics.val_f1": 0.7},
    ])
    assert comparison.rank_candidates(runs, mode="full", variant="robust")["run_id"].tolist() == ["b"]


def test_rank_skips_nan_and_missing_metrics():
    """Rank skips nan and missing metrics."""
    runs = pd.DataFrame([
        {"run_id": "bad", "status": "FINISHED", "tags.model_family": "extra_trees", "tags.preprocessing_variant": "baseline", "metrics.val_f1": "not-a-number"},
        {"run_id": "good", "status": "FINISHED", "tags.model_family": "extra_trees", "tags.preprocessing_variant": "baseline", "metrics.val_f1": 0.65},
    ])
    assert comparison.rank_candidates(runs)["run_id"].tolist() == ["good"]


def test_load_settings_from_yaml(tmp_path):
    """Load settings from yaml."""
    config = tmp_path / "config.yml"
    config.write_text("variants:\n  - baseline\n  - robust\n  - interactions\nmlflow:\n  experiment: some-experiment\npaths:\n  mlflow_db: mlflow.db\n", encoding="utf-8")
    name, db, variants = comparison.load_settings(config)
    assert name == "some-experiment"
    assert db == (comparison.ROOT / "mlflow.db").resolve()
    assert variants == ["baseline", "robust", "interactions"]


def test_compare_queries_mlflow_and_writes_csv(tmp_path, monkeypatch):
    """Compare queries mlflow and writes csv."""
    db = tmp_path / "mlflow.db"
    db.touch()
    config = tmp_path / "config.yml"
    config.write_text(f"variants:\n  - baseline\n  - robust\n  - interactions\nmlflow:\n  experiment: example\npaths:\n  mlflow_db: {db.as_posix()}\n", encoding="utf-8")
    calls = {}
    monkeypatch.setattr(comparison.mlflow, "set_tracking_uri", lambda uri: calls.setdefault("uri", uri))
    monkeypatch.setattr(comparison.mlflow, "get_experiment_by_name", lambda name: SimpleNamespace(experiment_id="42"))

    def fake_search_runs(**kwargs):
        """Provide deterministic MLflow search results."""
        calls["query"] = kwargs
        return pd.DataFrame([{
            "run_id": "winner", "status": "FINISHED", "tags.model_family": "hist_gradient_boosting",
            "tags.preprocessing_variant": "baseline", "tags.mode": "full",
            "metrics.val_f1": 0.74, "params.model_learning_rate": "0.1",
        }])
    monkeypatch.setattr(comparison.mlflow, "search_runs", fake_search_runs)
    exported = tmp_path / "leaderboard.csv"
    result = comparison.compare(config_path=config, output=exported)
    assert calls["query"]["experiment_ids"] == ["42"]
    assert result.iloc[0]["run_id"] == "winner"
    assert pd.read_csv(exported).iloc[0]["run_id"] == "winner"


def test_missing_mlflow_database_fails_clearly(tmp_path):
    """Missing mlflow database fails clearly."""
    config = tmp_path / "config.yml"
    config.write_text(f"variants:\n  - baseline\n  - robust\n  - interactions\nmlflow:\n  experiment: example\npaths:\n  mlflow_db: {(tmp_path / 'missing.db').as_posix()}\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Run training first"):
        comparison.compare(config_path=config)


def test_compare_defaults_to_full_runs(tmp_path, monkeypatch):
    """Quick smoke-check records must not compete with full experiments."""
    db = tmp_path / "mlflow.db"
    db.touch()
    config = tmp_path / "config.yml"
    config.write_text(
        f"variants:\n  - baseline\n  - robust\n  - interactions\nmlflow:\n  experiment: example\npaths:\n  mlflow_db: {db.as_posix()}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(comparison.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(
        comparison.mlflow, "get_experiment_by_name",
        lambda name: SimpleNamespace(experiment_id="42"),
    )
    monkeypatch.setattr(comparison.mlflow, "search_runs", lambda **kwargs: pd.DataFrame([
        {"run_id": "quick-high", "status": "FINISHED", "tags.model_family": "random_forest",
         "tags.preprocessing_variant": "baseline", "tags.mode": "quick", "metrics.val_f1": 0.99},
        {"run_id": "full-good", "status": "FINISHED", "tags.model_family": "extra_trees",
         "tags.preprocessing_variant": "baseline", "tags.mode": "full", "metrics.val_f1": 0.73},
    ]))
    assert comparison.compare(config_path=config)["run_id"].tolist() == ["full-good"]
    assert comparison.compare(config_path=config, mode="all")["run_id"].tolist() == ["quick-high", "full-good"]


def test_variant_ties_follow_yaml_order():
    """Variant ties follow yaml order."""
    runs = pd.DataFrame([
        {"run_id": "robust-first", "status": "FINISHED",
         "tags.model_family": "hist_gradient_boosting", "tags.preprocessing_variant": "robust",
         "metrics.val_f1": 0.7337, "metrics.val_pr_auc": 0.8296, "metrics.val_roc_auc": 0.9304},
        {"run_id": "baseline-second", "status": "FINISHED",
         "tags.model_family": "hist_gradient_boosting", "tags.preprocessing_variant": "baseline",
         "metrics.val_f1": 0.7337, "metrics.val_pr_auc": 0.8296, "metrics.val_roc_auc": 0.9304},
    ])
    result = comparison.rank_candidates(
        runs, variant_order=["baseline", "robust", "interactions"]
    )
    assert result["run_id"].tolist() == ["baseline-second", "robust-first"]


def test_variant_order_is_from_yaml_not_hardcoded():
    """Variant order is from yaml not hardcoded."""
    runs = pd.DataFrame([
        {"run_id": "baseline", "status": "FINISHED",
         "tags.model_family": "random_forest", "tags.preprocessing_variant": "baseline",
         "metrics.val_f1": 0.7},
        {"run_id": "robust", "status": "FINISHED",
         "tags.model_family": "random_forest", "tags.preprocessing_variant": "robust",
         "metrics.val_f1": 0.7},
    ])
    result = comparison.rank_candidates(runs, variant_order=["robust", "baseline"])
    assert result["run_id"].tolist() == ["robust", "baseline"]


def test_execution_filter_keeps_only_requested_runs():
    """Execution filter keeps only requested runs."""
    chosen = "a" * 32
    runs = pd.DataFrame([
        {"run_id": "old", "status": "FINISHED", "tags.model_family": "random_forest",
         "tags.preprocessing_variant": "baseline", "tags.mode": "full",
         "tags.training_execution_id": "b" * 32, "metrics.val_f1": 0.99},
        {"run_id": "current", "status": "FINISHED", "tags.model_family": "random_forest",
         "tags.preprocessing_variant": "baseline", "tags.mode": "full",
         "tags.training_execution_id": chosen, "metrics.val_f1": 0.72},
    ])
    assert comparison.rank_candidates(
        runs, mode="full", execution_id=chosen
    )["run_id"].tolist() == ["current"]


def test_candidate_index_breaks_same_variant_ties():
    """Candidate index breaks same variant ties."""
    runs = pd.DataFrame([
        {"run_id": "later", "status": "FINISHED", "tags.model_family": "random_forest",
         "tags.preprocessing_variant": "baseline", "tags.candidate_index": "5",
         "metrics.val_f1": 0.7},
        {"run_id": "earlier", "status": "FINISHED", "tags.model_family": "random_forest",
         "tags.preprocessing_variant": "baseline", "tags.candidate_index": "2",
         "metrics.val_f1": 0.7},
    ])
    assert comparison.rank_candidates(runs)["run_id"].tolist() == ["earlier", "later"]


def test_compare_rejects_invalid_execution_id(tmp_path):
    """Compare rejects invalid execution id."""
    config = tmp_path / "config.yml"
    with pytest.raises(ValueError, match="32-character"):
        comparison.compare(config_path=config, execution_id="not-an-id")


def cv_run(run_id="cv", **overrides):
    """Construct a realistic CV record using the training engine's tags and parameters."""
    row = {
        "run_id": run_id, "status": "FINISHED", "tags.model_family": "lightgbm",
        "tags.preprocessing_variant": "baseline", "tags.mode": "quick",
        "tags.evaluation_protocol": "cv", "tags.training_execution_id": "a" * 32,
        "tags.candidate_index": "0", "metrics.cv_f1": 0.72,
        "metrics.cv_pr_auc": 0.82, "metrics.cv_roc_auc": 0.93,
        "metrics.cv_f1_std": 0.005, "params.data_version": "data-hash",
        "params.seed": "42", "params.n_splits": "3", "params.calibration_size": "0.2",
        "params.threshold_policy": "fold_fit_inner_calibration",
    }
    row.update(overrides)
    return row


def test_protocols_are_never_mixed():
    """Filter explicitly tagged CV runs even if both metric types exist."""
    holdout = cv_run("legacy", **{"tags.evaluation_protocol": None, "metrics.val_f1": 0.99})
    cv = cv_run("cv", **{"metrics.val_f1": 1.0})
    runs = pd.DataFrame([holdout, cv])
    assert comparison.rank_candidates(runs, evaluation="cv")["run_id"].tolist() == ["cv"]
    assert comparison.rank_candidates(runs)["run_id"].tolist() == ["legacy"]


def test_cv_rank_matches_trainer_tie_breakers():
    """Rank by CV F1, average precision, ROC-AUC and candidate order."""
    runs = pd.DataFrame([
        cv_run("lower-f1", **{"metrics.cv_f1": 0.71, "metrics.cv_pr_auc": 0.99}),
        cv_run("lower-ap", **{"metrics.cv_pr_auc": 0.81}),
        cv_run("lower-roc", **{"metrics.cv_roc_auc": 0.92}),
        cv_run("later", **{"tags.candidate_index": "2"}),
        cv_run("winner", **{"tags.candidate_index": "1"}),
    ])
    assert comparison.rank_candidates(runs, evaluation="cv")["run_id"].tolist() == [
        "winner", "later", "lower-roc", "lower-ap", "lower-f1",
    ]


@pytest.mark.parametrize("column", comparison.CV_COMPARISON_PARAMS)
def test_cv_rejects_incompatible_protocol_metadata(column):
    """Prevent ranking candidates evaluated on incompatible folds or data."""
    runs = pd.DataFrame([cv_run("a"), cv_run("b", **{column: "different"})])
    with pytest.raises(ValueError, match="Incompatible CV comparison metadata"):
        comparison.rank_candidates(runs, evaluation="cv")


@pytest.mark.parametrize("column", comparison.CV_COMPARISON_PARAMS)
def test_cv_requires_comparison_metadata(column):
    """Reject runs whose comparability cannot be established."""
    runs = pd.DataFrame([cv_run()]).drop(columns=[column])
    with pytest.raises(ValueError, match="Missing CV comparison metadata"):
        comparison.rank_candidates(runs, evaluation="cv")


def test_cv_execution_scope_precedes_comparability_check():
    """Allow comparison of one execution from mixed study histories."""
    runs = pd.DataFrame([
        cv_run("chosen"),
        cv_run("other", **{"tags.training_execution_id": "b" * 32, "params.n_splits": "5"}),
    ])
    ranked = comparison.rank_candidates(runs, evaluation="cv", execution_id="a" * 32)
    assert ranked["run_id"].tolist() == ["chosen"]


def test_cv_excludes_failed_and_incomplete_records():
    """Ignore failed runs, caught candidate errors, final tests and invalid scores."""
    runs = pd.DataFrame([
        cv_run("good"), cv_run("failed", status="FAILED"),
        cv_run("caught", **{"tags.candidate_error": "serialization error"}),
        cv_run("test", **{"tags.stage": "final_test"}),
        cv_run("bad-f1", **{"metrics.cv_f1": float("inf")}),
        cv_run("bad-ap", **{"metrics.cv_pr_auc": None}),
        cv_run("bad-roc", **{"metrics.cv_roc_auc": 1.1}),
    ])
    assert comparison.rank_candidates(runs, evaluation="cv")["run_id"].tolist() == ["good"]


def test_holdout_excludes_caught_candidate_error():
    """Do not select a holdout candidate whose artifact logging failed."""
    good = cv_run("good", **{"tags.evaluation_protocol": None, "metrics.val_f1": 0.70})
    failed = cv_run("bad", **{"tags.evaluation_protocol": None, "metrics.val_f1": 0.99,
                              "tags.candidate_error": "artifact error"})
    assert comparison.rank_candidates(pd.DataFrame([good, failed]))["run_id"].tolist() == ["good"]


def test_cv_compare_exports_protocol_specific_leaderboard(tmp_path, monkeypatch, capsys):
    """Query, rank and export only CV metrics while preserving legacy CLI tags."""
    db = tmp_path / "mlflow.db"
    db.touch()
    config = tmp_path / "config.yml"
    config.write_text(
        f"variants: [baseline]\nmlflow:\n  experiment: example\npaths:\n  mlflow_db: {db.as_posix()}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(comparison.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(comparison.mlflow, "get_experiment_by_name",
                        lambda name: SimpleNamespace(experiment_id="42"))
    monkeypatch.setattr(comparison.mlflow, "search_runs", lambda **kwargs: pd.DataFrame([
        cv_run("cv", **{"metrics.val_f1": None}),
        cv_run("holdout", **{"tags.evaluation_protocol": None, "metrics.val_f1": 0.99}),
    ]))
    output = tmp_path / "cv.csv"
    result = comparison.compare(config_path=config, evaluation="cv", output=output)
    assert result["run_id"].tolist() == ["cv"]
    assert "metrics.val_f1" not in pd.read_csv(output).columns
    assert "Best mean CV F1: 0.7200" in capsys.readouterr().out
    assert comparison.compare(config_path=config, evaluation="cv", mode="validation")["run_id"].tolist() == ["cv"]


def test_cv_compare_rejects_full_mode(tmp_path):
    """Cross-validation comparisons cannot use the full lifecycle filter."""
    with pytest.raises(ValueError, match="validation-only"):
        comparison.compare(config_path=tmp_path / "unused.yml", evaluation="cv", mode="full")


def test_cv_cli_passes_protocol_and_execution(monkeypatch):
    """Forward explicit CV selection and execution filtering through the CLI."""
    import sys
    calls = {}

    def fake_compare(**kwargs):
        """Record the arguments forwarded by main."""
        calls.update(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(comparison, "compare", fake_compare)
    monkeypatch.setattr(sys, "argv", ["compare_experiments", "--evaluation", "cv",
                                      "--mode", "validation", "--execution-id", "a" * 32])
    assert comparison.main() == 0
    assert calls["evaluation"] == "cv"
    assert calls["mode"] == "validation"
    assert calls["execution_id"] == "a" * 32


def test_cv_comparison_real_mlflow(tmp_path):
    """Verify mixed-protocol ranking through a real isolated SQLite tracking store."""
    tracking = pytest.importorskip("mlflow.tracking")
    db = tmp_path / "comparison.db"
    uri = f"sqlite:///{db.as_posix()}"
    config = tmp_path / "config.yml"
    config.write_text(
        f"variants: [baseline]\nmlflow:\n  experiment: cv-comparison\npaths:\n  mlflow_db: {db.as_posix()}\n",
        encoding="utf-8",
    )
    previous_uri = comparison.mlflow.get_tracking_uri()
    try:
        client = tracking.MlflowClient(tracking_uri=uri)
        experiment_id = client.create_experiment("cv-comparison", artifact_location=(tmp_path / "artifacts").as_uri())
        cv_ids = []
        # Store records using the same tag and parameter names as train.py.
        for record in (cv_run("weaker"), cv_run("stronger", **{"metrics.cv_f1": 0.74}),
                       cv_run("holdout", **{"tags.evaluation_protocol": None, "metrics.val_f1": 0.99})):
            run = client.create_run(experiment_id, tags={
                k.removeprefix("tags."): str(v) for k, v in record.items()
                if k.startswith("tags.") and v is not None
            })
            run_id = run.info.run_id
            for key, value in record.items():
                if key.startswith("params."):
                    client.log_param(run_id, key.removeprefix("params."), value)
                elif key.startswith("metrics."):
                    client.log_metric(run_id, key.removeprefix("metrics."), value)
            client.set_terminated(run_id, status="FINISHED")
            if record.get("tags.evaluation_protocol") == "cv":
                cv_ids.append(run_id)
        result = comparison.compare(config_path=config, evaluation="cv", execution_id="a" * 32)
        assert result["run_id"].tolist() == cv_ids[::-1]
        assert result.iloc[0]["metrics.cv_f1"] == pytest.approx(0.74)
    finally:
        # Do not leak the temporary tracking URI into other integration tests.
        comparison.mlflow.set_tracking_uri(previous_uri)
