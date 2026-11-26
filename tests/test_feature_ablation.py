"""Verify fixed paired ablation experiments and their isolation from held-out data."""
import json
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts import run_feature_ablation as ablation
from tests.test_train_cv import cv_workspace, tracked_mlflow


def test_default_plan():
    """Freeze the two historical configurations and default initial representation pair."""
    args = ablation.parse_args([])
    plan = ablation.load_plan(args.plan)
    assert args.representations == ["control", "no_fnlwgt"]
    assert plan["candidates"]["hgb_cv_record"]["parameters"] == {
        "max_iter": 150, "learning_rate": 0.05, "max_leaf_nodes": 31}
    assert plan["candidates"]["lightgbm_pilot_record"]["parameters"]["subsample_freq"] == 1
    assert len(plan["representations"]) == 5


@pytest.mark.parametrize("change", ["missing_control", "duplicate", "unsupported_feature", "unknown_variant", "invalid_score"])
def test_invalid_plan(tmp_path, change):
    """Reject invalid experiment definitions before model fitting."""
    plan = ablation.load_plan(ablation.parse_args([]).plan)
    if change == "missing_control":
        del plan["representations"]["control"]
    elif change == "duplicate":
        plan["representations"]["copy"] = ["fnlwgt"]
    elif change == "unsupported_feature":
        plan["representations"]["no_age"] = ["age"]
    elif change == "unknown_variant":
        plan["candidates"]["hgb_cv_record"]["variant"] = "unknown"
    else:
        plan["candidates"]["hgb_cv_record"]["historical_cv_f1"] = 1.1
    path = tmp_path / "plan.yml"
    path.write_text(yaml.safe_dump(plan), encoding="utf-8")
    with pytest.raises(ValueError):
        ablation.load_plan(path)


def test_duplicate_overlap():
    """Identify newly identical raw vectors without altering original memberships."""
    X = pd.DataFrame({"age": [20, 20, 30, 20], "fnlwgt": [1, 2, 3, 4]})
    parts = [(np.array([0, 2]), np.array([1]), np.array([3]))]
    assert all(r["shared_groups"] == 0 for r in ablation.audit_overlap(X, parts, ()))
    rows = ablation.audit_overlap(X, parts, ("fnlwgt",))
    assert [r["shared_groups"] for r in rows] == [1, 1, 1]
    assert rows[0]["left_rows"] == rows[0]["right_rows"] == 1


def test_paired_comparison():
    """Compare F1 fold-by-fold, not against an unrelated historical score."""
    rows = [{"candidate": "a", "representation": r, "cv_f1": v}
            for r, v in [("control", .7), ("no_fnlwgt", .72)]]
    fold_rows = [{"candidate": "a", "representation": r, "fold": f, "f1": v}
                 for r, values in [("control", [.6, .8]), ("no_fnlwgt", [.64, .8])]
                 for f, v in enumerate(values, 1)]
    result = ablation.paired_comparison(rows, fold_rows)[0]
    assert result["delta_cv_f1"] == pytest.approx(.02)
    assert result["fold_deltas"] == pytest.approx([.04, 0])
    assert result["delta_fold_std"] == pytest.approx(.02)
    with pytest.raises(ValueError, match="identical fold numbers"):
        ablation.paired_comparison(rows, fold_rows[:-1])


def workspace_args(cv_workspace):
    """Map the existing synthetic training-only workspace into ablation CLI arguments."""
    args, _, _ = cv_workspace
    return ablation.parse_args(["--processed-dir", str(args.processed_dir),
                                "--output-dir", str(args.output_dir),
                                "--mlflow-db", str(args.mlflow_db),
                                "--mlflow-artifacts", str(args.mlflow_artifacts)])


@pytest.mark.parametrize("selected", [["no_fnlwgt"], ["control"], ["control", "unknown"], ["control", "control"]])
def test_invalid_scope(cv_workspace, selected):
    """Require a unique control and at least one known ablation."""
    args = workspace_args(cv_workspace)
    args.representations = selected
    with pytest.raises(ValueError, match="Select control"):
        ablation.run_ablation(args)
    assert not args.output_dir.exists()


def test_mismatched_raw_partitions(cv_workspace):
    """Prevent paired comparisons using different raw training rows."""
    args = workspace_args(cv_workspace)
    path = args.processed_dir / "interactions/raw_train.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="differ across variants"):
        ablation.run_ablation(args)
    assert not args.output_dir.exists()


@pytest.fixture
def representation_workspace(cv_workspace):
    """Add retained-feature diversity while preserving reduced duplicate groups."""
    args, X, reference = cv_workspace
    # Each pair of original batches shares retained values but differs in weight.
    for variant in ("baseline", "interactions"):
        path = args.processed_dir / variant / "raw_train.csv"
        frame = pd.read_csv(path)
        batch = np.arange(len(frame)) // 12
        frame["age"] += batch // 2
        frame.to_csv(path, index=False)
    return args, X, reference


def test_ablation_integration(representation_workspace, tracked_mlflow, monkeypatch):
    """Fit the four real candidates on synthetic data with paired folds and mocked tracking."""
    args = workspace_args(representation_workspace)
    tracker, _ = tracked_mlflow
    _, _, reference = representation_workspace
    before = reference.read_bytes()
    monkeypatch.setattr(ablation.training, "load_variant", Mock(side_effect=AssertionError("holdout accessed")))
    loader = Mock(wraps=ablation.load_untransformed_partition)
    monkeypatch.setattr(ablation, "load_untransformed_partition", loader)
    output = ablation.run_ablation(args)
    loader.assert_called_once()
    assert loader.call_args.args[1] == "train"
    summary = json.loads((output / "ablation_summary.json").read_text())
    rows = pd.read_csv(output / "ablation_leaderboard.csv")
    folds = pd.read_csv(output / "ablation_fold_metrics.csv")
    audit = pd.read_csv(output / "ablation_duplicate_audit.csv")
    assert summary["candidate_count"] == len(rows) == 4
    assert len(folds) == 12 and len(audit) == 18
    assert len(summary["comparisons"]) == len(summary["historical_control_comparison"]) == 2
    assert len(summary["fold_sha256"]) == 64
    assert tracker.start_run.call_count == 4 and tracker.log_artifact.call_count == 24
    assert all(c.args[0]["evaluation_protocol"] == "cv_ablation" for c in tracker.set_tags.call_args_list)
    assert set(rows.representation) == {"control", "no_fnlwgt"}
    assert rows.groupby("candidate").parameters.nunique().eq(1).all()
    assert audit.loc[audit.representation == "control", "shared_groups"].eq(0).all()
    assert audit.shared_groups.eq(0).all()
    assert all(row["shared_groups"] == 0 for row in summary["common_group_overlap"])
    assert summary["settings"]["grouping_excluded_features"] == ["fnlwgt", "race", "sex"]
    assert all(not row["directly_comparable"] for row in summary["historical_control_comparison"])
    assert len(summary["fold_balance"]) == 9
    assert rows.fold_sha256.nunique() == 1
    assert rows.fold_sha256.iloc[0] == summary["fold_sha256"]
    assert summary["mlflow_experiment"].endswith("-ablation-representation-safe")
    assert rows.cv_f1.between(0, 1).all()
    assert not (output / "best_model_pipeline.joblib").exists()
    assert reference.read_bytes() == before


def test_failed_candidate_has_no_success_summary(representation_workspace, tracked_mlflow, monkeypatch):
    """Propagate training failures rather than reporting a completed experiment."""
    args = workspace_args(representation_workspace)
    monkeypatch.setattr(ablation.cv, "evaluate_cv_candidate", Mock(side_effect=RuntimeError("fit failed")))
    with pytest.raises(RuntimeError, match="fit failed"):
        ablation.run_ablation(args)
    assert not list(args.output_dir.rglob("ablation_summary.json"))


def test_common_scope_independence(representation_workspace, tracked_mlflow):
    """All five representations reuse the same partitions as the initial pair."""
    args = workspace_args(representation_workspace)
    first = json.loads((ablation.run_ablation(args) / "ablation_summary.json").read_text())
    args.representations = list(ablation.load_plan(args.plan)["representations"])
    output = ablation.run_ablation(args)
    second = json.loads((output / "ablation_summary.json").read_text())
    assert first["fold_sha256"] == second["fold_sha256"]
    assert first["fold_balance"] == second["fold_balance"]
    assert second["candidate_count"] == 10
    assert pd.read_csv(output / "ablation_duplicate_audit.csv").shared_groups.eq(0).all()


def test_overlap_fails_before_training(representation_workspace, tracked_mlflow, monkeypatch):
    """Reject an unsafe partition before creating tracking runs or output files."""
    args = workspace_args(representation_workspace)
    tracker, _ = tracked_mlflow
    monkeypatch.setattr(ablation, "audit_overlap", Mock(return_value=[{"shared_groups": 1}]))
    with pytest.raises(RuntimeError, match="Common representation"):
        ablation.run_ablation(args)
    tracker.start_run.assert_not_called()
    assert not args.output_dir.exists()
