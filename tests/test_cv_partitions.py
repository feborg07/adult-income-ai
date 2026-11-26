"""Verify raw partition persistence and leakage-safe CV index generation."""

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    COLUMN_NAMES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    PreprocessingConfig,
    clean_data,
    create_cv_folds,
    create_train_val_test_split,
    encode_target,
    load_and_concatenate_data,
    load_untransformed_partition,
    predictor_group_ids,
    run_preprocessing_pipeline,
    split_features_target,
)


@pytest.fixture
def cv_data(sample_adult_df):
    """Build balanced synthetic groups with duplicates and conflicting labels."""
    batches = []
    for batch in range(12):
        frame = sample_adult_df.copy()
        # Vary a numerical predictor so batches contain distinct groups.
        frame["fnlwgt"] += batch * 1_000_000
        frame.loc[0, "occupation"] = np.nan
        batches.append(frame)
    source = pd.concat(batches, ignore_index=True)
    conflict = source.iloc[[0]].copy()
    conflict[TARGET_COLUMN] = ">50K"
    source = pd.concat([source, source.iloc[:2], conflict], ignore_index=True)
    X, y = split_features_target(encode_target(source))
    # Nonconsecutive indices detect accidental label-based rather than positional access.
    X.index = y.index = pd.Index(np.arange(len(X)) * 2 + 10)
    return X, y


def test_group_ids_ignore_labels_and_treat_missing_as_equal(cv_data):
    """Equal predictor vectors share a group despite missing values or labels."""
    X, y = cv_data
    groups = predictor_group_ids(X)
    assert groups[0] == groups[-3] == groups[-1]
    assert y.iloc[0] != y.iloc[-1]
    assert groups[1] == groups[-2]
    assert groups[0] != groups[1]


def test_folds_preserve_rows_and_prevent_group_overlap(cv_data):
    """Fit, calibration and scoring groups are disjoint and cover all rows."""
    X, y = cv_data
    groups = predictor_group_ids(X)
    folds = create_cv_folds(X, y)
    assert len(folds) == 3
    for fit, calibration, scoring in folds:
        # Each fold must partition all original training observations exactly once.
        parts = (fit, calibration, scoring)
        np.testing.assert_array_equal(np.sort(np.concatenate(parts)), np.arange(len(X)))
        group_sets = [set(groups[part]) for part in parts]
        assert group_sets[0].isdisjoint(group_sets[1])
        assert group_sets[0].isdisjoint(group_sets[2])
        assert group_sets[1].isdisjoint(group_sets[2])
        for part in parts:
            assert set(y.iloc[part]) == {0, 1}
            assert abs(y.iloc[part].mean() - y.mean()) < 0.20
    # Across outer folds, every row receives exactly one scoring assignment.
    np.testing.assert_array_equal(
        np.sort(np.concatenate([scoring for _, _, scoring in folds])), np.arange(len(X)),
    )


def test_folds_reproducible_and_inputs_unchanged(cv_data):
    """A fixed seed reproduces fold assignments without modifying input data."""
    X, y = cv_data
    X_before, y_before = X.copy(deep=True), y.copy(deep=True)
    for first, second in zip(create_cv_folds(X, y), create_cv_folds(X, y)):
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
    pd.testing.assert_frame_equal(X, X_before)
    pd.testing.assert_series_equal(y, y_before)


@pytest.mark.parametrize("n_splits", [True, 1, 2.5, 500])
def test_invalid_fold_counts(cv_data, n_splits):
    """Invalid or infeasible fold counts fail before fitting any model."""
    X, y = cv_data
    with pytest.raises(ValueError):
        create_cv_folds(X, y, n_splits=n_splits)


def test_invalid_cv_inputs(cv_data):
    """Reject transformed schemas, misaligned targets and invalid labels."""
    X, y = cv_data
    with pytest.raises(ValueError, match="raw predictor"):
        create_cv_folds(X.drop(columns="fnlwgt"), y)
    with pytest.raises(ValueError, match="matching"):
        create_cv_folds(X, y.reset_index(drop=True))
    with pytest.raises(ValueError, match="binary"):
        create_cv_folds(X, y * 0)
    with pytest.raises(ValueError, match="test_size"):
        create_cv_folds(X, y, calibration_size=0)


@pytest.mark.parametrize("interaction_features", [False, True])
def test_raw_partition_round_trip(cv_data, tmp_path, interaction_features):
    """Persist the exact existing train/val rows and reproduce processed values."""
    X, y = cv_data
    raw_file = tmp_path / "adult.data"
    raw = X.assign(income=y.map({0: "<=50K", 1: ">50K"}))
    raw[COLUMN_NAMES].to_csv(raw_file, index=False, header=False, na_rep="?")
    config = PreprocessingConfig(interaction_features=interaction_features)
    output_dir = tmp_path / "processed"
    preprocessor_path = tmp_path / "preprocessor.joblib"
    run_preprocessing_pipeline([raw_file], output_dir, preprocessor_path, config)

    # Independently reproduce the established split using the same cleaned source.
    cleaned = clean_data(load_and_concatenate_data([raw_file]))
    X_all, y_all = split_features_target(encode_target(cleaned))
    X_train, X_val, _, y_train, y_val, _ = create_train_val_test_split(X_all, y_all)
    preprocessor = joblib.load(preprocessor_path)
    for partition, expected_X, expected_y in (("train", X_train, y_train), ("val", X_val, y_val)):
        actual_X, actual_y = load_untransformed_partition(output_dir, partition)
        pd.testing.assert_frame_equal(actual_X, expected_X.reset_index(drop=True), check_dtype=False)
        pd.testing.assert_series_equal(actual_y, expected_y.reset_index(drop=True))
        # Reapply the saved preprocessor to confirm exact alignment with legacy CSVs.
        processed = pd.read_csv(output_dir / f"X_{partition}.csv")
        assert list(processed.columns) == list(preprocessor.get_feature_names_out())
        np.testing.assert_allclose(preprocessor.transform(actual_X), processed.to_numpy(), atol=1e-12)
        pd.testing.assert_series_equal(actual_y, pd.read_csv(output_dir / f"y_{partition}.csv")[TARGET_COLUMN])
    assert not (output_dir / "raw_test.csv").exists()


def test_loader_never_accepts_test_or_missing_file(tmp_path):
    """Restrict development loading to train/val and report absent artifacts."""
    with pytest.raises(ValueError, match="train.*val"):
        load_untransformed_partition(tmp_path, "test")
    with pytest.raises(FileNotFoundError, match="rerun preprocessing"):
        load_untransformed_partition(tmp_path)


def test_loader_rejects_invalid_schema_and_labels(cv_data, tmp_path):
    """Reject malformed development artifacts rather than silently proceeding."""
    X, y = cv_data
    path = tmp_path / "raw_train.csv"
    X.to_csv(path, index=False)
    with pytest.raises(ValueError, match="schema"):
        load_untransformed_partition(tmp_path)
    X.assign(income=y * 0).to_csv(path, index=False)
    with pytest.raises(ValueError, match="binary"):
        load_untransformed_partition(tmp_path)
