"""
Adult Income AI
Module: src/data/preprocess.py

Purpose
-------
Provide a configurable, reproducible preprocessing pipeline for
the UCI Adult Income dataset.

Architecture
------------
1. Load and concatenate raw datasets.
2. Clean and validate the data.
3. Encode the binary target.
4. Split observations into training and testing partitions.
5. Build a configurable scikit-learn preprocessing transformer.
6. Fit transformations using training data only.
7. Save processed datasets and preprocessing artifacts.

EDA-driven options
------------------
- Categorical missing-value imputation
- Rare-category grouping
- Numerical scaling
- Capital gain/loss transformations
- Capital presence indicators
- Education feature representation
- Selected feature interactions
- Duplicate-aware train/test splitting

Important
---------
The preprocessor is designed to be embedded inside a scikit-learn
Pipeline during model cross-validation.

Do not fit a preprocessor on the complete training dataset before
performing cross-validation. Doing so would allow validation-fold
information to influence fitted transformations.

Duplicate records are preserved by default.
"""

# ============================================================
# 1. Imports
# ============================================================

# Standard library
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path

# Model serialization
import joblib

# Data manipulation
import numpy as np
import pandas as pd

# Scikit-learn transformers and pipeline tools
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.utils.validation import check_is_fitted

# ============================================================
# 2. Global constants
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

TARGET_COLUMN = "income"

NUMERICAL_FEATURES = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

TARGET_MAPPING = {
    "<=50K": 0,
    ">50K": 1,
}

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Only the approved ablation candidates have no engineered-feature dependencies.
ABLATION_FEATURES = ("fnlwgt", "race", "sex")

CAPITAL_COLUMNS = [
    "capital-gain",
    "capital-loss",
]


# ============================================================
# 3. Preprocessing configuration
# ============================================================


def _check_choice(name: str, value, choices: tuple[str, ...]) -> None:
    """
    Raise ValueError when a setting is not one of its allowed values.

    The error message lists the options in order, for example:

        numerical_scaling must be 'standard', 'robust', or 'none'.

    Parameters
    ----------
    name:
        Name of the configuration field being checked.

    value:
        Value supplied for that field.

    choices:
        Allowed values, in the order shown in the error message.
    """

    if value in choices:
        return

    quoted = [f"'{choice}'" for choice in choices]

    # Two options read "'a' or 'b'"; three or more read "'a', 'b', or 'c'".
    if len(quoted) == 2:
        options = " or ".join(quoted)
    else:
        options = ", ".join(quoted[:-1]) + ", or " + quoted[-1]

    raise ValueError(f"{name} must be {options}.")


def _normalize_excluded_features(value) -> tuple[str, ...]:
    """Validate approved raw exclusions and normalize their order for reproducibility."""
    if not isinstance(value, (tuple, list)):
        raise ValueError("excluded_features must be a tuple or list of feature names.")
    if any(not isinstance(name, str) or name not in ABLATION_FEATURES for name in value):
        raise ValueError(f"excluded_features must contain only {ABLATION_FEATURES}.")
    if len(set(value)) != len(value):
        raise ValueError("excluded_features must not contain duplicate names.")
    # Canonical order ensures equivalent sets yield the same configuration.
    return tuple(name for name in ABLATION_FEATURES if name in value)


@dataclass(frozen=True)
class PreprocessingConfig:
    """
    Configuration controlling the preprocessing workflow.

    Each configuration can be used as a separate MLflow
    experiment during model development.

    Parameters
    ----------
    categorical_imputation:
        "constant" or "most_frequent".

    missing_category:
        Label used when categorical_imputation="constant".

    rare_category_frequency:
        Minimum frequency required for a category to remain
        separate in OneHotEncoder.

        None disables infrequent-category grouping.

        A float between 0 and 1 represents a proportion of
        the fitting dataset.

    numerical_imputation:
        Numerical missing-value strategy.

    numerical_scaling:
        "standard", "robust", or "none".

    capital_transform:
        "original", "log1p", or "both".

    capital_indicators:
        Add binary features indicating positive capital
        gain and capital loss.

    education_representation:
        "education-num", "education", or "both".

    interaction_features:
        Add selected EDA-motivated interaction features.

    duplicate_aware_split:
        Keep identical predictor vectors in the same partition
        without deleting observations.

    test_size:
        Proportion reserved for final testing.

    random_state:
        Random seed used for reproducibility.

    excluded_features:
        Approved raw predictors omitted from model inputs. Full raw rows
        are still used for duplicate-aware partitioning.
    """

    categorical_imputation: str = "constant"
    missing_category: str = "Unknown"

    rare_category_frequency: float | None = 0.01

    numerical_imputation: str = "median"
    numerical_scaling: str = "standard"

    capital_transform: str = "log1p"
    capital_indicators: bool = True

    education_representation: str = "education-num"

    interaction_features: bool = False

    duplicate_aware_split: bool = True

    test_size: float = TEST_SIZE
    validation_size: float = 0.20  # Fraction of the development (non-test) partition.
    random_state: int = RANDOM_STATE
    excluded_features: tuple[str, ...] = ()

    def __post_init__(self):
        """
        Validate configuration immediately after creation.
        """

        _check_choice(
            "categorical_imputation",
            self.categorical_imputation,
            ("constant", "most_frequent"),
        )

        if (
            not isinstance(self.missing_category, str)
            or not self.missing_category.strip()
        ):
            raise ValueError("missing_category must be a nonempty string.")

        frequency = self.rare_category_frequency

        if frequency is not None and (
            not isinstance(frequency, Real)
            or isinstance(frequency, bool)
            or not 0 < frequency < 1
        ):
            raise ValueError(
                "rare_category_frequency must be None or a number between 0 and 1."
            )

        _check_choice(
            "numerical_imputation",
            self.numerical_imputation,
            ("median", "mean"),
        )
        _check_choice(
            "numerical_scaling",
            self.numerical_scaling,
            ("standard", "robust", "none"),
        )
        _check_choice(
            "capital_transform",
            self.capital_transform,
            ("original", "log1p", "both"),
        )
        _check_choice(
            "education_representation",
            self.education_representation,
            ("education-num", "education", "both"),
        )

        if (
            isinstance(self.test_size, bool)
            or not isinstance(self.test_size, Real)
            or not 0 < self.test_size < 1
        ):
            raise ValueError("test_size must be between 0 and 1.")

        if (
            isinstance(self.validation_size, bool)
            or not isinstance(self.validation_size, Real)
            or not 0 < self.validation_size < 1
        ):
            raise ValueError("validation_size must be between 0 and 1.")

        if (
            isinstance(self.random_state, bool)
            or not isinstance(self.random_state, Integral)
            or not 0 <= self.random_state <= 2**32 - 1
        ):
            raise ValueError(
                "random_state must be an integer between 0 and "
                "2**32 - 1, excluding bool."
            )

        # JSON arrays are accepted and converted to the immutable canonical tuple.
        object.__setattr__(self, "excluded_features", _normalize_excluded_features(self.excluded_features))

        # Normalize NumPy scalar values for JSON serialization.
        object.__setattr__(self, "random_state", int(self.random_state))
        object.__setattr__(self, "test_size", float(self.test_size))
        object.__setattr__(self, "validation_size", float(self.validation_size))
        if frequency is not None:
            object.__setattr__(self, "rare_category_frequency", float(frequency))


# ============================================================
# 4. Load and concatenate raw datasets
# ============================================================


def load_and_concatenate_data(
    paths: Sequence[str | Path],
) -> pd.DataFrame:
    """
    Load multiple Adult Income files and concatenate them.

    Handles:
    - adult.data
    - adult.test
    - The metadata line beginning with "|" in adult.test
    - Missing values represented by "?"
    - Source-file tracking

    Returns
    -------
    pd.DataFrame
        Combined dataset containing all observations.
    """

    if not paths:
        raise ValueError("The paths list cannot be empty.")

    dataframes = []

    for path in paths:
        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")

        df = pd.read_csv(
            path,
            header=None,
            names=COLUMN_NAMES,
            sep=",",
            skipinitialspace=True,
            na_values=["?", ""],
            comment="|",
        )

        # Remove completely empty records.
        df = df.dropna(how="all")

        # Preserve the origin of each observation.
        df["source_file"] = path.name

        dataframes.append(df)

    combined_df = pd.concat(
        dataframes,
        ignore_index=True,
    )

    return combined_df


# ============================================================
# 5. Clean dataset
# ============================================================


def clean_data(
    df: pd.DataFrame,
    remove_duplicates: bool = False,
) -> pd.DataFrame:
    """
    Apply deterministic cleaning operations.

    Operations
    ----------
    1. Copy the input DataFrame.
    2. Remove surrounding whitespace.
    3. Normalize missing-value markers.
    4. Remove trailing periods from target labels.
    5. Convert numerical columns.
    6. Remove records without a valid target value.
    7. Optionally remove duplicate observations.

    Duplicate records are preserved by default.
    """

    cleaned = df.copy(deep=True)

    # Normalize string columns.
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    # Normalize textual missing-value markers.
    cleaned = cleaned.replace(
        ["?", "", "nan"],
        np.nan,
    )

    # Normalize income labels from adult.test.
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype("string").str.rstrip(".")

    # Convert all configured numerical features.
    for column in NUMERICAL_FEATURES:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    # A supervised observation requires a target label.
    cleaned = cleaned.dropna(subset=[TARGET_COLUMN])

    # Optional duplicate removal.
    # Source-file metadata must not define uniqueness.
    if remove_duplicates:
        cleaned = cleaned.drop_duplicates(
            subset=COLUMN_NAMES,
        )

    return cleaned.reset_index(drop=True)


# ============================================================
# 6. Validate dataset
# ============================================================


def validate_data(
    df: pd.DataFrame,
) -> None:
    """
    Validate the dataset before modeling.

    Checks:
    - Required columns exist.
    - Dataset is not empty.
    - Target contains only supported labels.
    - Both target classes are represented.
    """

    missing_columns = set(COLUMN_NAMES) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if df.empty:
        raise ValueError("The dataset is empty.")

    target_values = set(df[TARGET_COLUMN].dropna().unique())

    invalid_labels = target_values - set(TARGET_MAPPING)

    if invalid_labels:
        raise ValueError(f"Unexpected target labels: {sorted(invalid_labels)}")

    if target_values != set(TARGET_MAPPING):
        raise ValueError("The dataset must contain both income classes.")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("The target column contains missing values.")


# ============================================================
# 7. Encode binary target
# ============================================================


def encode_target(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Encode income as a binary numerical target.

    <=50K -> 0
    >50K  -> 1
    """

    encoded = df.copy(deep=True)

    if TARGET_COLUMN not in encoded.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    target = encoded[TARGET_COLUMN]
    if target.isna().any():
        raise ValueError("Target contains missing values.")

    invalid = target.loc[~target.isin(TARGET_MAPPING)]
    if not invalid.empty:
        raise ValueError(f"Unexpected target labels: {invalid.unique().tolist()}")

    encoded[TARGET_COLUMN] = encoded[TARGET_COLUMN].map(TARGET_MAPPING).astype("int64")

    return encoded


# ============================================================
# 8. Separate features and target
# ============================================================


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate predictor features and target labels.

    The source_file metadata column is excluded from
    the model input.
    """

    X = df[FEATURE_COLUMNS].copy()

    y = df[TARGET_COLUMN].copy()

    return X, y


# ============================================================
# 9. Duplicate-aware train/test splitting
# ============================================================


def predictor_group_ids(X: pd.DataFrame) -> np.ndarray:
    """Assign identical complete predictor rows the same positional group ID."""
    # Preserve the original splitter's grouping semantics, including missing values.
    group_index = pd.MultiIndex.from_frame(X.reset_index(drop=True))
    group_codes, _ = pd.factorize(group_index, sort=False)
    return group_codes


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    duplicate_aware: bool = True,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Split data into training and testing partitions.

    When duplicate_aware=True, identical complete feature
    records are assigned to the same partition.

    Duplicate records are not deleted.

    The split operates on unique feature groups and then
    expands those groups back into individual observations.

    Limitations
    -----------
    Group-level stratification approximates, but does not
    guarantee, observation-level class proportions.

    Groups are defined using complete predictor values.
    Identical feature vectors with different target labels
    are still assigned to the same partition.
    """

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")

    if not X.index.equals(y.index):
        raise ValueError("X and y must have matching indices.")

    # Reuse the same validation for direct calls and the full workflow.
    split_config = PreprocessingConfig(
        test_size=test_size,
        random_state=random_state,
    )
    test_size = split_config.test_size
    random_state = split_config.random_state

    if not duplicate_aware:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

    # Reset positional indices for reliable group expansion.
    X_reset = X.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)

    # Factorize complete feature rows into group IDs.
    #
    # Missing values are treated as equivalent within
    # the grouping operation.
    group_codes = predictor_group_ids(X_reset)

    # Construct a table linking each observation to
    # its feature-group ID and target label.
    grouping_df = pd.DataFrame(
        {
            "group": group_codes,
            "target": y_reset.to_numpy(),
        }
    )

    # Calculate group sizes and positive-class rates.
    group_summary = (
        grouping_df.groupby("group", sort=False)
        .agg(
            group_size=("target", "size"),
            positive_rate=("target", "mean"),
        )
        .reset_index()
    )

    # Assign each group to a class for approximate
    # group-level stratification.
    group_summary["stratum"] = (group_summary["positive_rate"] >= 0.5).astype(int)

    # Stratification requires at least two groups
    # in each stratum.
    stratum_counts = group_summary["stratum"].value_counts()

    can_stratify = (
        len(stratum_counts) == 2
        and stratum_counts.min() >= 2
        and int(np.ceil(len(group_summary) * test_size)) >= 2
        and len(group_summary) - int(np.ceil(len(group_summary) * test_size)) >= 2
    )

    group_train, group_test = train_test_split(
        group_summary["group"],
        test_size=test_size,
        random_state=random_state,
        stratify=(group_summary["stratum"] if can_stratify else None),
    )

    # Expand selected groups to original observations.
    train_mask = grouping_df["group"].isin(group_train).to_numpy()

    test_mask = grouping_df["group"].isin(group_test).to_numpy()

    X_train = X.iloc[np.flatnonzero(train_mask)].copy()

    X_test = X.iloc[np.flatnonzero(test_mask)].copy()

    y_train = y.iloc[np.flatnonzero(train_mask)].copy()

    y_test = y.iloc[np.flatnonzero(test_mask)].copy()

    # Verify that all observations were assigned.
    if np.any(train_mask & test_mask) or not np.all(train_mask | test_mask):
        raise RuntimeError("Train/test split lost observations or assigned them twice.")

    return X_train, X_test, y_train, y_test


def create_train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    validation_size: float = 0.20,
    random_state: int = RANDOM_STATE,
    duplicate_aware: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create train/validation/test partitions using the existing group-aware splitter.

    validation_size is relative to the development partition, not the full dataset.
    With test_size=.20 and validation_size=.20 the approximate fractions are
    64% training, 16% validation, and 20% testing.
    """
    config = PreprocessingConfig(
        test_size=test_size, validation_size=validation_size, random_state=random_state
    )
    X_dev, X_test, y_dev, y_test = create_train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state,
        duplicate_aware=duplicate_aware,
    )
    X_train, X_val, y_train, y_val = create_train_test_split(
        X_dev, y_dev, test_size=config.validation_size,
        random_state=config.random_state, duplicate_aware=duplicate_aware,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_cv_folds(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 3,
    calibration_size: float = 0.20,
    random_state: int = RANDOM_STATE,
    *,
    grouping_excluded_features: tuple[str, ...] = (),
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return positional fit, calibration and scoring indices for each CV fold.

    Only pass the original untransformed training partition. Complete predictor
    vectors define groups by default, even when duplicate observations have different labels.
    Optional grouping exclusions define a common reduced raw representation for
    all candidates without changing their model inputs or the saved partitions.
    The inner calibration fraction is relative to the outer training subset.
    No learned preprocessing or estimator fitting is performed here.
    """
    if list(X.columns) != FEATURE_COLUMNS:
        raise ValueError("CV requires the complete, ordered raw predictor columns.")
    if len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("X and y must have matching rows and indices.")
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("CV requires both binary target classes (0 and 1).")
    if isinstance(n_splits, bool) or not isinstance(n_splits, Integral) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2.")

    # Reuse the existing split configuration validation for fraction and seed.
    config = PreprocessingConfig(test_size=calibration_size, random_state=random_state)
    X_pos = X.reset_index(drop=True)
    y_pos = y.reset_index(drop=True)
    excluded = _normalize_excluded_features(grouping_excluded_features)
    grouping_frame = X_pos.drop(columns=list(excluded))
    groups = predictor_group_ids(grouping_frame)
    if any(len(np.unique(groups[y_pos.to_numpy() == label])) < n_splits for label in (0, 1)):
        raise ValueError("Too few distinct predictor groups per class for the requested folds.")

    splitter = StratifiedGroupKFold(
        n_splits=int(n_splits), shuffle=True, random_state=config.random_state,
    )
    folds = []
    for outer_train, scoring in splitter.split(X_pos, y_pos, groups):
        # Use the same reduced grouping basis for inner and outer splits.
        X_fit, X_cal, _, _ = create_train_test_split(
            grouping_frame.iloc[outer_train], y_pos.iloc[outer_train],
            test_size=config.test_size, random_state=config.random_state,
            duplicate_aware=True,
        )
        fit = X_fit.index.to_numpy(dtype=np.intp)
        calibration = X_cal.index.to_numpy(dtype=np.intp)
        scoring = np.asarray(scoring, dtype=np.intp)
        parts = (fit, calibration, scoring)

        # Reject infeasible stratification rather than scoring a single-class fold.
        if any(set(y_pos.iloc[part].unique()) != {0, 1} for part in parts):
            raise ValueError("Every fit, calibration and scoring subset must contain both classes.")
        group_sets = [set(groups[part]) for part in parts]
        if any(not group_sets[i].isdisjoint(group_sets[j]) for i, j in ((0, 1), (0, 2), (1, 2))):
            raise RuntimeError("Predictor groups overlap within a CV fold.")
        if not np.array_equal(np.sort(np.concatenate(parts)), np.arange(len(X_pos))):
            raise RuntimeError("CV fold lost observations or assigned them twice.")
        folds.append(parts)

    # Across folds, every original training row must be scored exactly once.
    if not np.array_equal(np.sort(np.concatenate([fold[2] for fold in folds])), np.arange(len(X_pos))):
        raise RuntimeError("CV scoring folds do not partition the training data.")
    return folds


# ============================================================
# 10. EDA-driven feature engineering
# ============================================================


class AdultFeatureEngineer(
    BaseEstimator,
    TransformerMixin,
):
    """
    Create optional EDA-driven features.

    This transformer is compatible with scikit-learn
    Pipeline and cross-validation.

    Feature transformations are deterministic and do not
    learn population statistics.

    Parameters
    ----------
    capital_transform:
        "original", "log1p", or "both".

    capital_indicators:
        Whether to add positive-capital indicators.

    education_representation:
        Which education representation to retain.

    interaction_features:
        Whether to create selected interaction features.

    excluded_features:
        Approved raw predictors to discard before feature engineering.
    """

    # Fallback for fitted reference artifacts serialized before this parameter existed.
    excluded_features = ()

    def __init__(
        self,
        capital_transform="log1p",
        capital_indicators=True,
        education_representation="education-num",
        interaction_features=False,
        excluded_features=(),
    ):

        """Initialize the configurable Adult Income feature transformations."""

        self.capital_transform = capital_transform

        self.capital_indicators = capital_indicators

        self.education_representation = education_representation

        self.interaction_features = interaction_features

        # Keep the constructor argument unchanged for sklearn.clone.
        self.excluded_features = excluded_features

    def fit(self, X, y=None):
        """
        Validate input columns and record input schema.
        """

        if not isinstance(X, pd.DataFrame):
            raise TypeError("AdultFeatureEngineer requires a DataFrame.")

        excluded = _normalize_excluded_features(self.excluded_features)
        missing = set(FEATURE_COLUMNS) - set(excluded) - set(X.columns)

        if missing:
            raise ValueError(f"Missing input features: {sorted(missing)}")

        self.feature_names_in_ = np.asarray(
            X.columns,
            dtype=object,
        )

        self.n_features_in_ = X.shape[1]

        # Preserve sklearn fitted-input metadata separately from the inference schema.
        self.required_features_ = np.asarray(
            [name for name in self.feature_names_in_ if name not in excluded], dtype=object
        )

        # Derive names from the actual transformation, preserving column order.
        self.output_features_ = np.asarray(
            self._transform_frame(X).columns, dtype=object
        )

        return self

    def get_feature_names_out(self, input_features=None):
        """Return output names and validate names propagated by Pipeline."""
        check_is_fitted(self, "output_features_")
        if input_features is not None and not np.array_equal(
            np.asarray(input_features, dtype=object), self.feature_names_in_
        ):
            raise ValueError("input_features must match feature_names_in_.")
        return self.output_features_.copy()

    def transform(self, X):
        """
        Apply deterministic feature transformations.
        """

        if not hasattr(self, "feature_names_in_"):
            raise RuntimeError("AdultFeatureEngineer must be fitted first.")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("AdultFeatureEngineer requires a DataFrame.")

        # Legacy fitted artifacts have only feature_names_in_.
        required = getattr(self, "required_features_", self.feature_names_in_)
        missing = set(required) - set(X.columns)

        if missing:
            raise ValueError(f"Missing input features: {sorted(missing)}")

        return self._transform_frame(X)

    def _transform_frame(self, X):
        """Build a DataFrame without invoking scikit-learn output wrapping."""
        required = getattr(self, "required_features_", self.feature_names_in_)
        transformed = X.loc[:, required].copy(deep=True)

        # ----------------------------------------------------
        # Capital gain/loss feature engineering
        # ----------------------------------------------------

        for column in CAPITAL_COLUMNS:
            values = pd.to_numeric(
                transformed[column],
                errors="coerce",
            )

            # Negative capital values are not expected
            # in the Adult Income dataset.
            if values.lt(0).any():
                raise ValueError(f"{column} contains negative values.")

            if self.capital_indicators:
                transformed[f"{column}-positive"] = (values > 0).astype("int64")

            if self.capital_transform in {
                "log1p",
                "both",
            }:
                transformed[f"{column}-log1p"] = np.log1p(values)

        # Remove original capital features when only
        # the logarithmic representation is requested.
        if self.capital_transform == "log1p":
            transformed = transformed.drop(columns=CAPITAL_COLUMNS)

        # ----------------------------------------------------
        # Education representation
        # ----------------------------------------------------

        if self.education_representation == "education-num":
            transformed = transformed.drop(columns=["education"])

        elif self.education_representation == "education":
            transformed = transformed.drop(columns=["education-num"])

        # ----------------------------------------------------
        # Selected interaction features
        # ----------------------------------------------------

        if self.interaction_features:
            # Age × educational attainment.
            #
            # education-num is an ordered numerical
            # representation of education level.
            transformed["age-education-interaction"] = pd.to_numeric(
                X["age"],
                errors="coerce",
            ) * pd.to_numeric(
                X["education-num"],
                errors="coerce",
            )

        return transformed


# ============================================================
# 11. Custom occupation-hours interaction transformer
# ============================================================


class OccupationHoursTransformer(
    BaseEstimator,
    TransformerMixin,
):
    """
    Create occupation-specific working-hours features.

    The transformer learns occupation categories only
    from the fitting dataset.

    Each category receives an interaction column:

        occupation_indicator * hours_per_week

    Unseen occupations produce zeros across the learned
    occupation-specific columns.
    """

    def fit(self, X, y=None):
        """Learn occupation-specific output feature names from training data."""

        if not isinstance(X, pd.DataFrame):
            raise TypeError("OccupationHoursTransformer requires a DataFrame.")

        missing = {"occupation", "hours-per-week"} - set(X.columns)
        if missing:
            raise ValueError(f"Missing input features: {sorted(missing)}")

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]

        self.occupations_ = sorted(
            X["occupation"].dropna().astype(str).unique().tolist()
        )

        self.output_features_ = np.asarray(
            list(self.feature_names_in_)
            + [
                f"occupation-hours-interaction-{index}"
                for index in range(len(self.occupations_))
            ],
            dtype=object,
        )

        return self

    def get_feature_names_out(self, input_features=None):
        """Return input names followed by the learned interaction names."""
        check_is_fitted(self, "output_features_")
        if input_features is not None and not np.array_equal(
            np.asarray(input_features, dtype=object), self.feature_names_in_
        ):
            raise ValueError("input_features must match feature_names_in_.")
        return self.output_features_.copy()

    def transform(self, X):
        """Generate occupation-specific working-hour interaction features."""

        if not hasattr(self, "occupations_"):
            raise RuntimeError("OccupationHoursTransformer must be fitted first.")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("OccupationHoursTransformer requires a DataFrame.")
        missing = set(self.feature_names_in_) - set(X.columns)
        if missing:
            raise ValueError(f"Missing input features: {sorted(missing)}")

        transformed = X.loc[:, self.feature_names_in_].copy(deep=True)

        hours = pd.to_numeric(
            transformed["hours-per-week"],
            errors="coerce",
        )

        for index, occupation in enumerate(self.occupations_):
            column_name = f"occupation-hours-interaction-{index}"

            transformed[column_name] = (
                transformed["occupation"].eq(occupation).fillna(False).astype("float64")
                * hours
            )

        return transformed


# ============================================================
# 12. Determine transformed feature groups
# ============================================================


def get_feature_groups(
    config: PreprocessingConfig,
) -> tuple[list[str], list[str]]:
    """
    Determine numerical and categorical columns after
    EDA-driven feature engineering.
    """

    # Exclusions apply to model inputs, not raw dataset grouping.
    numerical = [name for name in NUMERICAL_FEATURES if name not in config.excluded_features]

    categorical = [name for name in CATEGORICAL_FEATURES if name not in config.excluded_features]

    # Education representation.
    if config.education_representation == "education-num":
        categorical.remove("education")

    elif config.education_representation == "education":
        numerical.remove("education-num")

    # Capital representation.
    if config.capital_transform == "log1p":
        for column in CAPITAL_COLUMNS:
            numerical.remove(column)

    if config.capital_transform in {"log1p", "both"}:
        for column in CAPITAL_COLUMNS:
            numerical.append(f"{column}-log1p")

    # Binary capital indicators.
    if config.capital_indicators:
        for column in CAPITAL_COLUMNS:
            numerical.append(f"{column}-positive")

    # Interaction features.
    if config.interaction_features:
        numerical.append("age-education-interaction")

        # Dynamically select occupation-specific interactions.
        # These columns are generated during pipeline fitting.

    return numerical, categorical


# ============================================================
# 13. Build configurable preprocessing pipeline
# ============================================================
class NumericalFeatureSelector:
    """
    Select numerical features, including dynamically
    generated occupation-hours interactions.

    A top-level class is compatible with joblib serialization.
    """

    def __init__(self, feature_names, include_interactions=False):
        """Initialize the selector with configured numerical feature names."""

        self.feature_names = list(feature_names)
        self.include_interactions = include_interactions

    def __call__(self, X):
        """Select configured numerical columns and optional interaction features."""

        selected = list(self.feature_names)

        if self.include_interactions:
            selected.extend(
                column
                for column in X.columns
                if column.startswith("occupation-hours-interaction-")
            )

        return selected


def build_preprocessor(
    config: PreprocessingConfig | None = None,
) -> Pipeline:
    """
    Build the complete feature preprocessing pipeline.

    Pipeline stages
    ---------------
    1. EDA-driven feature engineering.
    2. Optional occupation-hours interactions.
    3. Numerical imputation and scaling.
    4. Categorical imputation and one-hot encoding.

    The returned pipeline is unfitted.

    It should be combined with a classifier for
    cross-validation.
    """

    if config is None:
        config = PreprocessingConfig()

    numerical_features, categorical_features = get_feature_groups(config)

    # --------------------------------------------------------
    # Numerical pipeline
    # --------------------------------------------------------

    numerical_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy=config.numerical_imputation,
                keep_empty_features=True,
            ),
        )
    ]

    if config.numerical_scaling == "standard":
        numerical_steps.append(("scaler", StandardScaler()))

    elif config.numerical_scaling == "robust":
        numerical_steps.append(("scaler", RobustScaler()))

    numerical_pipeline = Pipeline(steps=numerical_steps)

    # --------------------------------------------------------
    # Categorical pipeline
    # --------------------------------------------------------

    if config.categorical_imputation == "constant":
        categorical_imputer = SimpleImputer(
            strategy="constant",
            fill_value=config.missing_category,
            keep_empty_features=True,
        )

    else:
        categorical_imputer = SimpleImputer(
            strategy="most_frequent",
            keep_empty_features=True,
        )

    categorical_encoder = OneHotEncoder(
        handle_unknown=(
            "infrequent_if_exist"
            if config.rare_category_frequency is not None
            else "ignore"
        ),
        min_frequency=config.rare_category_frequency,
        sparse_output=False,
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", categorical_imputer),
            ("encoder", categorical_encoder),
        ]
    )

    # --------------------------------------------------------
    # ColumnTransformer
    # --------------------------------------------------------

    column_transformer = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NumericalFeatureSelector(
                    feature_names=numerical_features,
                    include_interactions=config.interaction_features,
                ),
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # --------------------------------------------------------
    # Complete preprocessing pipeline
    # --------------------------------------------------------

    steps = [
        (
            "feature_engineering",
            AdultFeatureEngineer(
                capital_transform=config.capital_transform,
                capital_indicators=config.capital_indicators,
                education_representation=(config.education_representation),
                interaction_features=config.interaction_features,
                excluded_features=config.excluded_features,
            ),
        )
    ]

    if config.interaction_features:
        steps.append(
            (
                "occupation_interactions",
                OccupationHoursTransformer(),
            )
        )

    steps.append(("column_transformer", column_transformer))

    return Pipeline(steps=steps)


# ============================================================
# 14. Transform training and testing datasets
# ============================================================


def transform_datasets(
    preprocessor: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_val: pd.DataFrame | None = None,
):
    """Fit on training only; transform test and optional validation unchanged.

    The two-way API remains available for existing callers. When X_val is
    provided, return (train, validation, test) processed DataFrames.
    """
    train_array = preprocessor.fit_transform(X_train)
    feature_names = preprocessor.get_feature_names_out()

    def frame(X, values):
        """Restore feature names and original row indices after transformation."""
        return pd.DataFrame(values, columns=feature_names, index=X.index)

    train_processed = frame(X_train, train_array)
    test_processed = frame(X_test, preprocessor.transform(X_test))
    if X_val is None:
        return train_processed, test_processed
    val_processed = frame(X_val, preprocessor.transform(X_val))
    return train_processed, val_processed, test_processed


# ============================================================
# 15. Save processed datasets
# ============================================================


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str | Path,
    *,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
) -> None:
    """Save train/test and, when supplied, validation datasets as CSV files."""
    if (X_val is None) != (y_val is None):
        raise ValueError("X_val and y_val must both be supplied or both omitted.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (
        ("X_train", X_train), ("X_test", X_test),
        ("y_train", y_train), ("y_test", y_test),
    ):
        data.to_csv(output_dir / f"{name}.csv", index=False)
    if X_val is not None:
        X_val.to_csv(output_dir / "X_val.csv", index=False)
        y_val.to_csv(output_dir / "y_val.csv", index=False)


def load_untransformed_partition(
    output_dir: str | Path,
    partition: str = "train",
) -> tuple[pd.DataFrame, pd.Series]:
    """Load cleaned, untransformed predictors and encoded labels for train or val.

    The test partition is deliberately unsupported. Files are generated by the
    regular preprocessing pipeline from its existing partition assignments.
    """
    if partition not in ("train", "val"):
        raise ValueError("Untransformed partition must be 'train' or 'val'.")
    path = Path(output_dir) / f"raw_{partition}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; rerun preprocessing to generate raw partitions.")

    # Empty cells represent missing values; preserve literal categorical strings.
    frame = pd.read_csv(
        path, keep_default_na=False, na_values=[""], float_precision="round_trip",
        dtype={column: str for column in CATEGORICAL_FEATURES},
    )
    if list(frame.columns) != FEATURE_COLUMNS + [TARGET_COLUMN]:
        raise ValueError(f"Unexpected untransformed partition schema: {path}")
    X, y = split_features_target(frame)
    for column in NUMERICAL_FEATURES:
        X[column] = pd.to_numeric(X[column], errors="raise")
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError(f"Untransformed partition requires both binary classes: {path}")
    return X, y.astype("int64")


# ============================================================
# 16. Save fitted preprocessor
# ============================================================


def save_preprocessor(
    preprocessor: Pipeline,
    output_path: str | Path,
) -> None:
    """
    Serialize a fitted preprocessing pipeline using joblib.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        preprocessor,
        output_path,
    )


# ============================================================
# 17. Save preprocessing configuration
# ============================================================


def save_config(
    config: PreprocessingConfig,
    output_path: str | Path,
) -> None:
    """
    Save the preprocessing configuration as JSON.

    This makes the experiment settings reproducible
    and easy to inspect or log with MLflow.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            asdict(config),
            file,
            indent=4,
        )


# ============================================================
# 18. Run complete preprocessing workflow
# ============================================================


def run_preprocessing_pipeline(
    paths: Sequence[str | Path],
    output_dir: str | Path,
    preprocessor_path: str | Path,
    config: PreprocessingConfig | None = None,
    remove_duplicates: bool = False,
) -> dict:
    """
    Execute the complete preprocessing workflow.

    Returns
    -------
    dict
        Summary of the preprocessing execution.
    """

    if config is None:
        config = PreprocessingConfig()

    # --------------------------------------------------------
    # Load raw data
    # --------------------------------------------------------

    raw_df = load_and_concatenate_data(paths)

    raw_rows = len(raw_df)

    # --------------------------------------------------------
    # Clean and validate
    # --------------------------------------------------------

    cleaned_df = clean_data(
        raw_df,
        remove_duplicates=remove_duplicates,
    )

    validate_data(cleaned_df)

    cleaned_rows = len(cleaned_df)

    # --------------------------------------------------------
    # Encode target
    # --------------------------------------------------------

    encoded_df = encode_target(cleaned_df)

    # --------------------------------------------------------
    # Separate predictors and target
    # --------------------------------------------------------

    X, y = split_features_target(encoded_df)

    # --------------------------------------------------------
    # Train/validation/test split
    # --------------------------------------------------------

    X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(
        X, y, test_size=config.test_size, validation_size=config.validation_size,
        random_state=config.random_state, duplicate_aware=config.duplicate_aware_split,
    )

    # --------------------------------------------------------
    # Build and fit preprocessor
    # --------------------------------------------------------

    preprocessor = build_preprocessor(config)

    X_train_processed, X_val_processed, X_test_processed = transform_datasets(
        preprocessor, X_train, X_test, X_val=X_val,
    )

    # --------------------------------------------------------
    # Save artifacts
    # --------------------------------------------------------

    save_processed_data(
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        output_dir,
        X_val=X_val_processed,
        y_val=y_val,
    )

    # Persist the exact existing development partitions for fold-local fitting.
    # Store labels with predictors to preserve row alignment; never export raw test here.
    for partition, X_raw, y_raw in (("train", X_train, y_train), ("val", X_val, y_val)):
        X_raw.assign(**{TARGET_COLUMN: y_raw}).to_csv(
            Path(output_dir) / f"raw_{partition}.csv", index=False,
        )

    save_preprocessor(
        preprocessor,
        preprocessor_path,
    )

    save_config(
        config,
        Path(output_dir) / "preprocessing_config.json",
    )

    # --------------------------------------------------------
    # Execution summary
    # --------------------------------------------------------

    summary = {
        "raw_rows": raw_rows,
        "cleaned_rows": cleaned_rows,
        "training_rows": len(X_train),
        "validation_rows": len(X_val),
        "testing_rows": len(X_test),
        "original_features": X.shape[1],
        "processed_features": X_train_processed.shape[1],
        "training_positive_rate": float(y_train.mean()),
        "validation_positive_rate": float(y_val.mean()),
        "testing_positive_rate": float(y_test.mean()),
        "test_size": config.test_size,
        "validation_size": config.validation_size,
        "random_state": config.random_state,
        "duplicate_aware_split": config.duplicate_aware_split,
        "remove_duplicates": remove_duplicates,
        "configuration": asdict(config),
    }

    return summary


# ============================================================
# 19. Main execution
# ============================================================


def main():
    """
    Execute preprocessing using the default EDA-driven
    configuration.

    Run from the project root:

        python -m src.data.preprocess
    """

    # Resolve project root from this module's location.
    project_root = Path(__file__).resolve().parents[2]

    raw_paths = [
        project_root / "data/raw/adult.data",
        project_root / "data/raw/adult.test",
    ]

    output_dir = project_root / "data/processed"

    preprocessor_path = project_root / "models/preprocessor.joblib"

    # EDA-driven starting configuration.
    config = PreprocessingConfig(
        categorical_imputation="constant",
        missing_category="Unknown",
        rare_category_frequency=0.01,
        numerical_imputation="median",
        numerical_scaling="standard",
        capital_transform="log1p",
        capital_indicators=True,
        education_representation="education-num",
        interaction_features=False,
        duplicate_aware_split=True,
        test_size=0.20,
        random_state=42,
    )

    summary = run_preprocessing_pipeline(
        paths=raw_paths,
        output_dir=output_dir,
        preprocessor_path=preprocessor_path,
        config=config,
        remove_duplicates=False,
    )

    print("\nPreprocessing completed successfully.\n")

    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    # Keep serialized custom transformers importable in a fresh Python process.
    # Also allow `python src/data/preprocess.py` by adding the project root.
    import sys

    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.data.preprocess import main as run_main

    run_main()