"""Verify selected model inputs, shared partition identity and study isolation."""
import hashlib
import json
from unittest.mock import Mock

import joblib
import pandas as pd
import pytest
import yaml

from scripts import compare_experiments as comparison
from src.data.preprocess import create_cv_folds, predictor_group_ids
from src.models import train as training
from tests.test_compare_experiments import cv_run
from tests.test_train_cv import cv_workspace, tracked_mlflow
from tests.test_train_optuna import optuna_workspace
from tests.test_optuna_persistence import run_and_read


def test_selected_policy():
    """Separate model input exclusion from the stricter common grouping basis."""
    policy = training.resolve_cv_representation(training.load_training_config())
    assert policy["representation"] == "no_fnlwgt"
    assert policy["excluded_features"] == ("fnlwgt",)
    assert policy["grouping_excluded_features"] == ("fnlwgt", "race", "sex")
    assert training.resolve_cv_representation({})["excluded_features"] == ()


@pytest.mark.parametrize("value", [None, [], {}, "unknown", "representation_a"])
def test_invalid_policy(value, tmp_path):
    """Reject unapproved representations when loading YAML."""
    config = training.load_training_config()
    config["cv_representation"] = value
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="cv_representation"):
        training.load_training_config(path)


@pytest.mark.parametrize("name,excluded", [
    ("no_fnlwgt", ("fnlwgt",)),
    ("no_fnlwgt_race", ("fnlwgt", "race")),
    ("no_fnlwgt_sex", ("fnlwgt", "sex")),
    ("no_fnlwgt_demographics", ("fnlwgt", "race", "sex")),
])
def test_reduced_policies_share_grouping(name, excluded):
    """Resolve each approved representation with the same conservative grouping."""
    policy = training.resolve_cv_representation({"cv_representation": name})
    assert policy["excluded_features"] == excluded
    assert policy["grouping_excluded_features"] == ("fnlwgt", "race", "sex")
    assert policy["grouping_policy"] == "common_retained_raw_predictors_v1"


@pytest.mark.parametrize("name,excluded", [
    ("no_fnlwgt", ("fnlwgt",)),
    ("no_fnlwgt_race", ("fnlwgt", "race")),
])
def test_selected_cv_uses_shared_indices_and_reduced_inputs(cv_workspace, tracked_mlflow, monkeypatch, tmp_path, name, excluded):
    """Fit all three variants with the selected inputs and verify logged artifacts."""
    args, X, reference = cv_workspace
    tracker, artifacts = tracked_mlflow
    args.variants = ["baseline", "robust", "interactions"]
    config = training.load_training_config()
    config["cv_representation"] = name
    args.config = tmp_path / "config.yml"
    args.config.write_text(yaml.safe_dump(config), encoding="utf-8")
    before = reference.read_bytes()
    raw_before = (args.processed_dir / "baseline/raw_train.csv").read_bytes()
    prepare = Mock(wraps=training.cv.prepare_cv_folds)
    monkeypatch.setattr(training.cv, "prepare_cv_folds", prepare)
    monkeypatch.setattr(training, "load_variant", Mock(side_effect=AssertionError("holdout accessed")))
    training.train(args)
    assert prepare.call_count == 3
    raw, y = training.load_untransformed_partition(args.processed_dir / "baseline", "train")
    indices = create_cv_folds(raw, y, grouping_excluded_features=("fnlwgt", "race", "sex"))
    expected_hash = hashlib.sha256(json.dumps([[p.tolist() for p in f] for f in indices]).encode()).hexdigest()
    groups = predictor_group_ids(raw.drop(columns=["fnlwgt", "race", "sex"]))
    for fit, cal, score in indices:
        assert set(groups[fit]).isdisjoint(groups[cal])
        assert set(groups[fit]).isdisjoint(groups[score])
        assert set(groups[cal]).isdisjoint(groups[score])
    for call in prepare.call_args_list:
        assert call.args[2].excluded_features == excluded
        assert call.kwargs["grouping_excluded_features"] == ("fnlwgt", "race", "sex")
    output = next(args.output_dir.iterdir())
    summary = json.loads((output / "cv_training_summary.json").read_text())
    rows = pd.read_csv(output / "cv_leaderboard.csv")
    assert summary["fold_sha256"] == expected_hash
    assert rows.fold_sha256.tolist() == [expected_hash] * 3
    assert summary["excluded_features"] == list(excluded)
    assert set(summary["effective_preprocessing_sha256"]) == set(args.variants)
    for call in tracker.log_params.call_args_list:
        assert call.args[0]["fold_sha256"] == expected_hash
        assert json.loads(call.args[0]["excluded_features"]) == list(excluded)
    pipeline = joblib.load(artifacts / "cv/fold_1/model_pipeline.joblib")
    reduced = X.iloc[:5].drop(columns=list(excluded))
    assert pipeline.predict_proba(reduced).shape == (5, 2)
    with pytest.raises(ValueError, match="sex"):
        pipeline.predict_proba(reduced.drop(columns=["sex"]))
    if "race" not in excluded:
        with pytest.raises(ValueError, match="race"):
            pipeline.predict_proba(reduced.drop(columns=["race"]))
    assert reference.read_bytes() == before
    assert (args.processed_dir / "baseline/raw_train.csv").read_bytes() == raw_before


def test_optuna_representation_change_isolated(optuna_workspace, tracked_mlflow, tmp_path):
    """Create distinct persistent studies for A, B and the legacy representation."""
    args, _, _ = optuna_workspace
    args.models, args.search, args.trials = ["hist_gradient_boosting"], "optuna", 1
    config = training.load_training_config()
    args.config = tmp_path / "config.yml"
    args.config.write_text(yaml.safe_dump(config), encoding="utf-8")
    first, _, cumulative = run_and_read(args)
    assert first["optuna_context"]["representation"]["representation"] == "no_fnlwgt"
    assert cumulative.iloc[0].representation == "no_fnlwgt"
    config["cv_representation"] = "no_fnlwgt_race"
    args.config.write_text(yaml.safe_dump(config), encoding="utf-8")
    switched, _, _ = run_and_read(args)
    assert switched["optuna_context_id"] != first["optuna_context_id"]
    assert switched["fold_sha256"] == first["fold_sha256"]
    assert switched["excluded_features"] == ["fnlwgt", "race"]
    config["cv_representation"] = "control"
    args.config.write_text(yaml.safe_dump(config), encoding="utf-8")
    second, _, _ = run_and_read(args)
    assert first["optuna_context_id"] != second["optuna_context_id"]
    assert second["excluded_features"] == []
    assert second["optuna_cumulative_successful_candidates"] == 1


def identity_fields():
    """Return the comparison metadata for one corrected CV execution."""
    return {"params.representation": "no_fnlwgt_race", "params.excluded_features": '["fnlwgt", "race"]',
            "params.grouping_excluded_features": '["fnlwgt", "race", "sex"]',
            "params.grouping_policy": "common_retained_raw_predictors_v1", "params.fold_sha256": "folds"}


@pytest.mark.parametrize("column", comparison.CV_REPRESENTATION_PARAMS)
def test_comparison_rejects_mixed_identity(column):
    """Reject any discrepancy in grouping, model inputs or actual fold membership."""
    first = identity_fields()
    second = {**first, column: "different"}
    with pytest.raises(ValueError, match="Incompatible CV comparison metadata"):
        comparison.rank_candidates(pd.DataFrame([cv_run("a", **first), cv_run("b", **second)]), evaluation="cv")


def test_comparison_rejects_legacy_mixed_with_corrected():
    """Preserve old-only reports but reject mixing historical and fingerprinted folds."""
    old = cv_run("old")
    new = cv_run("new", **identity_fields())
    with pytest.raises(ValueError, match="Missing CV comparison metadata"):
        comparison.rank_candidates(pd.DataFrame([old, new]), evaluation="cv")
    assert len(comparison.rank_candidates(pd.DataFrame([old]), evaluation="cv")) == 1
    assert len(comparison.rank_candidates(pd.DataFrame([new]), evaluation="cv")) == 1
