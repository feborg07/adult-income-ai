"""Verify approved feature exclusions and preservation of CV row identities."""

import json
from dataclasses import asdict, replace
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.preprocess import (
    ABLATION_FEATURES, AdultFeatureEngineer, PreprocessingConfig,
    build_preprocessor, get_feature_groups, predictor_group_ids, save_config,
)
from src.models.cross_validation import prepare_cv_folds

EXCLUSIONS = [subset for size in range(1, 4) for subset in combinations(ABLATION_FEATURES, size)]
VARIANTS = [
    PreprocessingConfig(),
    PreprocessingConfig(numerical_scaling="robust", capital_transform="both"),
    PreprocessingConfig(interaction_features=True),
]


@pytest.mark.parametrize("value", [None, "fnlwgt", 1, {"fnlwgt"}, [None],
                                  ["income"], ["occupation"], ["unknown"],
                                  ["fnlwgt", "fnlwgt"]])
def test_invalid_exclusions(value):
    """Reject invalid or unapproved exclusions before fitting."""
    with pytest.raises(ValueError, match="excluded_features"):
        PreprocessingConfig(excluded_features=value)


def test_configuration_round_trip(tmp_path):
    """Normalize JSON lists and keep legacy configuration files readable."""
    config = PreprocessingConfig(excluded_features=["sex", "fnlwgt", "race"])
    assert config.excluded_features == ABLATION_FEATURES
    path = tmp_path / "preprocessing_config.json"
    save_config(config, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["excluded_features"] == list(ABLATION_FEATURES)
    assert PreprocessingConfig(**payload) == config
    # Previous configurations did not contain the exclusion field.
    legacy = asdict(PreprocessingConfig())
    del legacy["excluded_features"]
    assert PreprocessingConfig(**legacy).excluded_features == ()


@pytest.mark.parametrize("config", VARIANTS, ids=["baseline", "robust", "interactions"])
@pytest.mark.parametrize("excluded", EXCLUSIONS)
def test_exclusions_preserve_other_features(features_target, config, excluded):
    """Remove only selected features and accept reduced raw inputs at inference."""
    X, y = features_target
    original = X.copy(deep=True)
    control = build_preprocessor(config).set_output(transform="pandas")
    expected = control.fit_transform(X, y)
    ablated_config = replace(config, excluded_features=excluded)
    model = clone(build_preprocessor(ablated_config)).set_output(transform="pandas")
    actual = model.fit_transform(X, y)
    # Direct numerical features and encoded categorical columns must disappear.
    removed = [name for name in expected.columns if any(
        name == raw or name.startswith(f"{raw}_") for raw in excluded
    )]
    assert removed
    pd.testing.assert_frame_equal(actual, expected.drop(columns=removed))
    engineer = model.named_steps["feature_engineering"]
    assert set(engineer.required_features_) == set(X.columns) - set(excluded)
    assert list(engineer.feature_names_in_) == list(X.columns)
    numerical, categorical = get_feature_groups(ablated_config)
    assert not set(excluded).intersection(numerical + categorical)
    reduced = X.drop(columns=list(excluded))
    pd.testing.assert_frame_equal(model.transform(reduced), actual)
    pd.testing.assert_frame_equal(model.transform(reduced.iloc[:, ::-1]), actual)
    # Ignored columns cannot affect the output even when their values change.
    modified = X.copy()
    for name in excluded:
        modified[name] = "ignored-value"
    pd.testing.assert_frame_equal(model.transform(modified), actual)
    pd.testing.assert_frame_equal(X, original)
    np.testing.assert_array_equal(model.get_feature_names_out(), actual.columns)


def test_fit_accepts_reduced_schema(features_target):
    """Allow fitting on data already missing the excluded fields."""
    X, y = features_target
    config = PreprocessingConfig(excluded_features=ABLATION_FEATURES)
    reduced = X.drop(columns=list(ABLATION_FEATURES))
    full_model = build_preprocessor(config)
    reduced_model = build_preprocessor(config)
    np.testing.assert_allclose(full_model.fit_transform(X, y), reduced_model.fit_transform(reduced, y))
    np.testing.assert_allclose(reduced_model.transform(X), reduced_model.transform(reduced))


def test_missing_retained_column_fails(features_target):
    """Require retained inputs during both fitting and prediction."""
    X, y = features_target
    pipeline = build_preprocessor(PreprocessingConfig(excluded_features=("fnlwgt",)))
    with pytest.raises(ValueError, match="age"):
        pipeline.fit(X.drop(columns="age"), y)
    pipeline.fit(X, y)
    with pytest.raises(ValueError, match="race"):
        pipeline.transform(X.drop(columns="race"))


def test_direct_transformer_validation(features_target):
    """Apply exclusion validation even when the transformer is used directly."""
    X, _ = features_target
    with pytest.raises(ValueError, match="excluded_features"):
        AdultFeatureEngineer(excluded_features=("education-num",)).fit(X)


@pytest.mark.parametrize("interactions", [False, True])
def test_serialized_classifier_accepts_reduced_inputs(features_target, tmp_path, interactions):
    """Preserve reduced-schema predictions in a serialized synthetic classifier."""
    X, y = features_target
    config = PreprocessingConfig(excluded_features=ABLATION_FEATURES, interaction_features=interactions)
    model = Pipeline([
        ("preprocess", build_preprocessor(config)),
        ("classifier", LogisticRegression(max_iter=500, random_state=42)),
    ]).fit(X, y)
    expected = model.predict_proba(X)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    reduced = X.drop(columns=list(ABLATION_FEATURES))
    np.testing.assert_allclose(restored.predict_proba(reduced), expected)
    assert list(restored.named_steps["preprocess"].named_steps["feature_engineering"].required_features_) == list(reduced.columns)


@pytest.mark.parametrize("interactions", [False, True])
def test_legacy_fitted_artifact_compatibility(features_target, tmp_path, interactions):
    """Simulate older fitted artifact state and preserve its original input contract."""
    X, y = features_target
    model = build_preprocessor(PreprocessingConfig(interaction_features=interactions))
    expected = model.fit_transform(X, y)
    engineer = model.named_steps["feature_engineering"]
    # Older joblib artifacts lack these attributes.
    del engineer.required_features_
    del engineer.excluded_features
    path = tmp_path / "legacy.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    np.testing.assert_allclose(restored.transform(X), expected)
    with pytest.raises(ValueError, match="fnlwgt"):
        restored.transform(X.drop(columns="fnlwgt"))
    assert clone(restored).named_steps["feature_engineering"].excluded_features == ()


def test_control_and_ablation_share_original_cv_rows(features_target):
    """Keep original CV assignments when exclusions create duplicate vectors."""
    X, y = features_target
    frames = [X.assign(fnlwgt=X["fnlwgt"] + batch * 1_000_000) for batch in range(12)]
    X = pd.concat(frames, ignore_index=True)
    y = pd.concat([y] * 12, ignore_index=True)
    before = X.copy(deep=True)
    assert len(np.unique(predictor_group_ids(X))) > len(np.unique(predictor_group_ids(X.drop(columns="fnlwgt"))))
    control = prepare_cv_folds(X, y, PreprocessingConfig())
    ablated = prepare_cv_folds(X, y, PreprocessingConfig(excluded_features=("fnlwgt",)))
    for left, right in zip(control, ablated):
        # Compare the exact fitting, calibration and scoring rows.
        for part in ("fit", "calibration", "scoring"):
            full = getattr(left, f"X_{part}")
            reduced = getattr(right, f"X_{part}")
            pd.testing.assert_frame_equal(full.drop(columns="fnlwgt"), reduced)
            pd.testing.assert_series_equal(getattr(left, f"y_{part}"), getattr(right, f"y_{part}"))
    pd.testing.assert_frame_equal(X, before)
