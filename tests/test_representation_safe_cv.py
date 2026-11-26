"""Check common reduced raw group isolation without changing model feature inputs."""
from dataclasses import replace
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    ABLATION_FEATURES, PreprocessingConfig, create_cv_folds,
    encode_target, predictor_group_ids, split_features_target,
)
from src.models.cross_validation import prepare_cv_folds


@pytest.fixture
def grouped_data(sample_adult_df):
    """Create reduced duplicate pairs including missing predictors and mixed labels."""
    batches = []
    for batch in range(12):
        frame = sample_adult_df.copy()
        frame["age"] += batch // 2
        frame["fnlwgt"] += batch * 1_000_000
        frame.loc[0, "occupation"] = np.nan
        if batch % 2:
            frame["race"] = "Other"
            frame["sex"] = frame["sex"].map({"Male": "Female", "Female": "Male"})
            frame.loc[0, "income"] = ">50K"
        batches.append(frame)
    X, y = split_features_target(encode_target(pd.concat(batches, ignore_index=True)))
    X.index = y.index = pd.Index(np.arange(len(X)) * 3 + 5)
    return X, y


def test_common_group_isolation(grouped_data):
    """Every retained representation is disjoint across all three fold subsets."""
    X, y = grouped_data
    original_X, original_y = X.copy(deep=True), y.copy(deep=True)
    folds = create_cv_folds(X, y, grouping_excluded_features=ABLATION_FEATURES)
    common = predictor_group_ids(X.drop(columns=list(ABLATION_FEATURES)))
    assert common[0] == common[12] and y.iloc[0] != y.iloc[12]
    for parts in folds:
        np.testing.assert_array_equal(np.sort(np.concatenate(parts)), np.arange(len(X)))
        for part in parts:
            assert set(y.iloc[part]) == {0, 1}
        for size in range(4):
            for excluded in combinations(ABLATION_FEATURES, size):
                groups = predictor_group_ids(X.drop(columns=list(excluded)))
                for left, right in combinations(parts, 2):
                    assert not np.intersect1d(groups[left], groups[right]).size
    np.testing.assert_array_equal(np.sort(np.concatenate([p[2] for p in folds])), np.arange(len(X)))
    pd.testing.assert_frame_equal(X, original_X)
    pd.testing.assert_series_equal(y, original_y)


def test_grouping_reproducibility_and_default(grouped_data):
    """Canonical exclusions reproduce partitions while the legacy default is unchanged."""
    X, y = grouped_data
    first = create_cv_folds(X, y, grouping_excluded_features=ABLATION_FEATURES)
    second = create_cv_folds(X, y, grouping_excluded_features=list(reversed(ABLATION_FEATURES)))
    old = create_cv_folds(X, y)
    explicit = create_cv_folds(X, y, grouping_excluded_features=())
    for lhs, rhs in zip(first, second):
        for left, right in zip(lhs, rhs):
            np.testing.assert_array_equal(left, right)
    for lhs, rhs in zip(old, explicit):
        for left, right in zip(lhs, rhs):
            np.testing.assert_array_equal(left, right)
    assert any(not np.array_equal(a[2], b[2]) for a, b in zip(first, old))


@pytest.mark.parametrize("excluded", [None, "fnlwgt", ("age",), ("sex", "sex")])
def test_invalid_grouping(grouped_data, excluded):
    """Reject malformed or unapproved grouping exclusions."""
    with pytest.raises(ValueError, match="excluded_features"):
        create_cv_folds(*grouped_data, grouping_excluded_features=excluded)


@pytest.mark.parametrize("excluded", [(), ("fnlwgt",), ("fnlwgt", "sex"), ABLATION_FEATURES])
def test_preprocessing_grouping_independence(grouped_data, excluded):
    """Use identical memberships while keeping each candidate's own feature schema."""
    X, y = grouped_data
    indices = create_cv_folds(X, y, grouping_excluded_features=ABLATION_FEATURES)
    config = replace(PreprocessingConfig(), excluded_features=excluded)
    folds = prepare_cv_folds(X, y, config, grouping_excluded_features=ABLATION_FEATURES)
    for fold, parts in zip(folds, indices):
        for actual, part in zip((fold.X_fit, fold.X_calibration, fold.X_scoring), parts):
            assert actual.index.equals(X.iloc[part].index)
        engineer = fold.preprocessor.named_steps["feature_engineering"]
        assert set(engineer.required_features_) == set(X.columns) - set(excluded)
