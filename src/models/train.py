
"""
Adult Income AI
Module: src/models/train.py

Train machine learning classifiers using previously saved
preprocessing partitions.

Usage:
    python -m src.models.train --quick
    python -m src.models.train --mode validation --evaluation cv
    python -m src.models.train --quick --evaluation cv  # Compatible alias
    python -m src.models.train

Required inputs for each preprocessing variant:

    data/processed/<variant>/X_train.csv
    data/processed/<variant>/y_train.csv
    data/processed/<variant>/X_val.csv
    data/processed/<variant>/y_val.csv
    data/processed/<variant>/X_test.csv
    data/processed/<variant>/y_test.csv

    models/preprocessors/<variant>.joblib

The script:
    - Never reloads raw datasets.
    - Never recreates dataset splits.
    - Never refits saved preprocessing transformers in holdout mode.
    - CV mode fits independent preprocessors from saved raw training rows.
    - Fits candidate models using training data only.
    - Selects models and thresholds using validation data.
    - Evaluates winners using held-out test data.
    - Tracks experiments using MLflow.
    - Preserves feature names in deployable pipelines.
    - Saves each training execution to a unique models/runs/<run_id>/ directory.
    - Never modifies DVC-tracked models/reference/ artifacts.
    - Does not automatically promote newly trained models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import os
import uuid
import tempfile
import warnings

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import joblib
import mlflow
import yaml
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .algorithms.base import BINARY_CLASSIFICATION
from .algorithms.registry import ALGORITHM_REGISTRY
from .tasks import binary_classification as binary_task
from . import cross_validation as cv
from src.data.preprocess import PreprocessingConfig, load_untransformed_partition

# Preserve the existing public evaluation functions for callers and tests.
from .tasks.binary_classification import evaluate, optimal_threshold, rank


# ============================================================
# 1. Configuration
# ============================================================

LOG = logging.getLogger("training")

ROOT = Path(__file__).resolve().parents[2]

VARIANTS = (
    "baseline",
    "robust",
    "interactions",
)

# Defaults also support older YAML files without a cross_validation section.
DEFAULT_CV_SETTINGS = {"n_splits": 3, "calibration_size": 0.20}




# ============================================================
# 2. Dataset container
# ============================================================

@dataclass
class Dataset:
    """Store previously generated dataset partitions."""

    X_train: pd.DataFrame
    y_train: pd.Series

    X_val: pd.DataFrame
    y_val: pd.Series

    X_test: pd.DataFrame | None
    y_test: pd.Series | None

    preprocessor_path: Path


# ============================================================
# 3. Load saved preprocessing partitions
# ============================================================

def load_variant(
    processed_dir: Path,
    preprocessor_dir: Path,
    variant: str,
    *,
    include_test: bool = True,
) -> Dataset:
    """
    Load partitions generated during preprocessing.

    No splitting, cleaning, or preprocessing is performed.
    Validation-only callers must set include_test=False.
    """

    folder = processed_dir / variant
    matrices = {}

    splits = ("train", "val", "test") if include_test else ("train", "val")
    for split in splits:

        x_path = folder / f"X_{split}.csv"
        y_path = folder / f"y_{split}.csv"

        for path in (x_path, y_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Required partition missing: {path}. "
                    "Run preprocessing first."
                )

        X = pd.read_csv(x_path)
        y_frame = pd.read_csv(y_path)

        if y_frame.shape[1] != 1:
            raise ValueError(
                f"Expected exactly one target column: {y_path}"
            )

        y = y_frame.iloc[:, 0]

        if X.empty or len(X) != len(y):
            raise ValueError(
                f"Empty or misaligned data: {variant}/{split}"
            )

        if X.columns.duplicated().any():
            raise ValueError(
                f"Duplicate feature columns: {variant}/{split}"
            )

        if X.isna().any().any():
            raise ValueError(
                f"Missing processed features: {variant}/{split}"
            )

        if not all(
            pd.api.types.is_numeric_dtype(dtype)
            for dtype in X.dtypes
        ):
            raise ValueError(
                f"Non-numeric features: {variant}/{split}"
            )

        if not np.isfinite(
            X.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"Non-finite features: {variant}/{split}"
            )

        # Delegate target validation to the binary classification task.
        matrices[split] = (
            X,
            binary_task.validate_target(y, f"{variant}/{split}"),
        )

    # Validate consistent feature columns and ordering.
    train_columns = matrices["train"][0].columns

    for split in splits[1:]:

        if not train_columns.equals(
            matrices[split][0].columns
        ):
            raise ValueError(
                f"Feature columns/order differ: {variant}/{split}"
            )

    # Training and validation require both classes.
    for split in ("train", "val"):

        binary_task.require_both_classes(
            matrices[split][1], f"{variant}/{split}"
        )

    preprocessor_path = (
        preprocessor_dir / f"{variant}.joblib"
    )

    if not preprocessor_path.is_file():
        raise FileNotFoundError(
            f"Missing fitted preprocessor: {preprocessor_path}"
        )

    return Dataset(
        *matrices["train"],
        *matrices["val"],
        *matrices.get("test", (None, None)),
        preprocessor_path=preprocessor_path,
    )


# ============================================================
# 4. Model candidates and hyperparameters
# ============================================================

def resolve_cv_settings(config: dict, args=None) -> dict:
    """Validate YAML CV settings and apply explicit validation-mode overrides."""
    section = config.get("cross_validation", {})
    if not isinstance(section, dict):
        raise ValueError("cross_validation must be a mapping.")
    unknown = section.keys() - DEFAULT_CV_SETTINGS.keys()
    if unknown:
        raise ValueError(f"Unknown cross_validation keys: {sorted(map(str, unknown))}")
    settings = {**DEFAULT_CV_SETTINGS, **section}

    def validate(values: dict) -> None:
        """Reject invalid fold counts and calibration fractions before training."""
        folds = values["n_splits"]
        fraction = values["calibration_size"]
        if type(folds) is not int or folds < 2:
            raise ValueError("cross_validation.n_splits must be an integer of at least 2.")
        if type(fraction) not in (int, float) or not 0 < fraction < 1:
            raise ValueError("cross_validation.calibration_size must be between 0 and 1 (exclusive).")

    # Invalid YAML is not silently repaired by CLI overrides.
    validate(settings)
    overrides = {
        "n_splits": getattr(args, "cv_folds", None),
        "calibration_size": getattr(args, "calibration_size", None),
    }
    overrides = {key: value for key, value in overrides.items() if value is not None}
    if overrides and resolve_evaluation(args) != "cv":
        raise ValueError("CV overrides require --evaluation cv.")
    settings.update(overrides)
    validate(settings)
    return settings


def load_training_config(path: Path | None = None) -> dict:
    """Load and validate YAML settings before starting any training."""
    path = Path(path) if path is not None else ROOT / "configs/config.yml"
    if not path.is_file():
        raise FileNotFoundError(f"Training YAML configuration not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a YAML mapping.")
    required = {"seed", "variants", "selection_metric", "paths", "mlflow", "models"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing YAML configuration keys: {sorted(missing)}")
    if config["selection_metric"] != "val_f1":
        raise ValueError("Only validation F1 selection is supported.")
    if not isinstance(config["seed"], int):
        raise ValueError("seed must be an integer.")
    if not isinstance(config["variants"], list) or not config["variants"]:
        raise ValueError("variants must be a nonempty list.")
    if not set(config["variants"]).issubset(VARIANTS):
        raise ValueError(f"Unsupported preprocessing variant: {config['variants']}")
    if len(set(config["variants"])) != len(config["variants"]):
        raise ValueError("Duplicate preprocessing variants are not allowed.")
    if not isinstance(config["models"], dict) or not config["models"]:
        raise ValueError("models must be a nonempty mapping.")
    unsupported = set(config["models"]) - set(ALGORITHM_REGISTRY.names())
    if unsupported:
        raise ValueError(f"Unsupported model families: {sorted(unsupported)}")
    for name, spec in config["models"].items():
        # Reject algorithms that do not support the current prediction task.
        ALGORITHM_REGISTRY.get(name, task_type=BINARY_CLASSIFICATION)
        if not isinstance(spec, dict) or not isinstance(spec.get("fixed"), dict):
            raise ValueError(f"{name}: fixed must be a mapping.")
        grid = spec.get("grid")
        if not isinstance(grid, dict) or not grid:
            raise ValueError(f"{name}: grid must be a nonempty mapping.")
        if any(not isinstance(values, list) or not values for values in grid.values()):
            raise ValueError(f"{name}: every grid entry must be a nonempty list.")
        if set(spec["fixed"]) & set(grid):
            raise ValueError(f"{name}: fixed and grid parameters overlap.")
    for key in ("processed_dir", "preprocessor_dir", "output_dir",
                "mlflow_db", "mlflow_artifacts"):
        if not isinstance(config["paths"].get(key), str):
            raise ValueError(f"paths.{key} must be a string.")
    if not isinstance(config["mlflow"].get("experiment"), str):
        raise ValueError("mlflow.experiment must be a string.")
    config["cross_validation"] = resolve_cv_settings(config)
    return config


def resolve_training_scope(config: dict, args) -> tuple[dict, str]:
    """Resolve CLI scope without mutating the source YAML configuration.

    Preserve YAML ordering so deterministic candidate tie-breaking remains
    unchanged, regardless of the order of CLI selection arguments.
    """
    effective = dict(config)
    for key in ("models", "variants"):
        available = config[key]
        requested = getattr(args, key, None)
        if requested is None:
            continue
        if not requested or len(set(requested)) != len(requested):
            raise ValueError(f"{key} selection must be nonempty and unique.")
        unknown = set(requested) - set(available)
        if unknown:
            raise ValueError(f"Unavailable {key}: {sorted(unknown)}")
        effective[key] = (
            {name: spec for name, spec in available.items() if name in requested}
            if key == "models"
            else [name for name in available if name in requested]
        )

    # Search depth is independent of test evaluation and packaging permissions.
    search = getattr(args, "search", None)
    if search is None:
        search = "first" if args.quick else "grid"
    if search not in ("first", "grid"):
        raise ValueError(f"Unsupported search strategy: {search}")
    return effective, search


def candidates(quick: bool = False, config: dict | None = None):
    """Enumerate YAML candidates; quick=True selects the first grid entry.

    This argument controls enumeration only, not test evaluation or promotion.
    """
    config = load_training_config() if config is None else config
    seed = config["seed"]
    specs = []
    for family, spec in config["models"].items():
        plugin = ALGORITHM_REGISTRY.get(
            family, task_type=BINARY_CLASSIFICATION
        )
        grid = spec["grid"]
        keys = list(grid)
        combinations = product(*(grid[key] for key in keys))
        for values in combinations:
            params = dict(zip(keys, values))
            # The plugin owns estimator construction, not the engine.
            estimator = plugin.build({**spec["fixed"], **params}, seed)
            specs.append((family, params, estimator))
            if quick:
                break  # One configuration per family, preserving YAML order.
    return specs


# ============================================================
# 5. Artifact fingerprint
# ============================================================

def file_sha256(path: Path) -> str:
    """Calculate SHA-256 for any local file artifact."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def git_revision() -> str | None:
    """Return the source revision when running inside a Git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def dataset_fingerprints(processed_dir: Path, variant: str) -> dict[str, str]:
    """Hash the exact saved train/validation/test inputs used by training."""
    folder = processed_dir / variant
    return {
        f"{kind}_{split}.csv": file_sha256(folder / f"{kind}_{split}.csv")
        for split in ("train", "val", "test")
        for kind in ("X", "y")
    }


def verify_saved_model(path: Path, data: Dataset, classifier, threshold: float) -> None:
    """Reload serialized pipeline and check compatibility with saved features."""
    pipeline = joblib.load(path)  # Only load artifacts created by trusted code.
    if not hasattr(pipeline, "predict_proba"):
        raise ValueError("Promoted pipeline does not implement predict_proba")
    # The deployable pipeline accepts raw columns; the training classifier
    # accepts transformed columns. Use the fitted preprocessor's input schema.
    preprocessor = pipeline.named_steps["preprocess"]
    if not hasattr(preprocessor, "feature_names_in_"):
        raise ValueError("Preprocessor does not expose raw feature names")
    raw_names = list(preprocessor.feature_names_in_)
    if not raw_names:
        raise ValueError("Empty raw feature schema")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("Invalid decision threshold")
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("Saved model is missing or empty")
    # Checking raw predictions requires raw examples; the processed CSVs
    # cannot be passed through preprocessing again. Validate serialization
    # against the stored fitted estimator and its transformed input instead.
    saved_classifier = pipeline.named_steps["classifier"]
    sample = data.X_val.iloc[: min(10, len(data.X_val))]
    actual = saved_classifier.predict_proba(sample)
    expected = classifier.predict_proba(sample)
    if actual.shape != (len(sample), 2):
        raise ValueError("Invalid prediction probability shape")
    if not np.isfinite(actual).all() or not np.allclose(actual.sum(axis=1), 1.0):
        raise ValueError("Invalid predicted probabilities")
    if not np.allclose(actual, expected, atol=1e-10, rtol=1e-10):
        raise ValueError("Serialized classifier differs from trained winner")


def atomic_json_write(path: Path, payload: dict) -> None:
    """Write metadata via a temporary file on the same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, indent=2)
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


# ============================================================
# 6. MLflow configuration
# ============================================================

def configure_mlflow(
    experiment: str,
    db: Path,
    artifacts: Path,
):
    """Configure local MLflow tracking using SQLite."""

    db.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifacts.mkdir(
        parents=True,
        exist_ok=True,
    )

    tracking_uri = (
        f"sqlite:///{db.resolve().as_posix()}"
    )

    mlflow.set_tracking_uri(tracking_uri)

    existing = mlflow.get_experiment_by_name(
        experiment
    )

    if existing is None:
        mlflow.create_experiment(
            experiment,
            artifact_location=(
                artifacts.resolve().as_uri()
            ),
        )

    mlflow.set_experiment(experiment)


# ============================================================
# 7. Deployable pipeline construction
# ============================================================

def build_deployable_pipeline(
    fitted_preprocessor,
    classifier,
    feature_names,
    sample_features,
):
    """
    Assemble an inference pipeline without refitting models.

    The fitted preprocessor produces a NumPy array.
    The classifier was fitted with named DataFrame columns.

    FunctionTransformer restores those column names only after
    preprocessing, avoiding feature-name warnings.

    The adapter does not alter numerical feature values.
    """

    feature_names = list(feature_names)

    # Verify the classifier's expected feature names.
    if hasattr(classifier, "feature_names_in_"):

        classifier_names = list(
            classifier.feature_names_in_
        )

        if classifier_names != feature_names:
            raise ValueError(
                "Classifier feature names do not match "
                "the fitted preprocessor output."
            )

    # Convert NumPy output to a named DataFrame.
    # pandas.DataFrame is importable and joblib-compatible.
    adapter = FunctionTransformer(
        pd.DataFrame,
        kw_args={"columns": feature_names},
        validate=False,
        feature_names_out="one-to-one",
    )

    # Fit only the stateless adapter's metadata.
    # The preprocessor and classifier are NOT refitted.
    adapter.fit(sample_features)

    deployable = Pipeline([
        ("preprocess", fitted_preprocessor),
        ("feature_names", adapter),
        ("classifier", classifier),
    ])

    return deployable


# ============================================================
# 8. Main training workflow
# ============================================================

def resolve_evaluation(args) -> str:
    """Resolve the evaluation protocol and reject CV in the full lifecycle."""
    evaluation = getattr(args, "evaluation", "holdout")
    if evaluation not in ("holdout", "cv"):
        raise ValueError(f"Unsupported evaluation protocol: {evaluation}")
    if evaluation == "cv" and not args.quick:
        raise ValueError("Cross-validation requires --quick or --mode validation; full evaluation is separate.")
    return evaluation


def cv_rank(row: dict) -> tuple[float, float, float]:
    """Rank cross-validation candidates without mixing holdout metrics."""
    return row["cv_f1"], row["cv_pr_auc"], row["cv_roc_auc"]


def log_cv_fold(fold: cv.PreparedFold, model, result: cv.FoldResult) -> None:
    """Log a complete fold-specific research pipeline and threshold to MLflow."""
    pipeline = build_deployable_pipeline(
        fold.preprocessor, model, fold.X_fit.columns, fold.X_fit,
    )
    # Each fold has different learned transformations and a separate threshold.
    # These research artifacts are never packaged as promotion candidates.
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)
        joblib.dump(pipeline, path / "model_pipeline.joblib")
        (path / "fold_result.json").write_text(
            json.dumps(asdict(result), indent=2), encoding="utf-8",
        )
        for name in ("model_pipeline.joblib", "fold_result.json"):
            mlflow.log_artifact(str(path / name), artifact_path=f"cv/fold_{result.number}")


def run_cv_search(config: dict, scope: dict, processed: Path, output: Path,
                  execution_id: str, config_sha256: str) -> None:
    """Evaluate the existing first/grid candidates on raw training folds only.

    Reuse the registry, candidate generator and MLflow experiment configured by
    train(). CV has distinct metrics and outputs, and cannot promote a model.
    """
    specifications = candidates(scope["search"] == "first", config=config)
    rows, winners = [], {}
    # Require identical training rows across selected variants before fitting any model.
    hashes = {variant: file_sha256(processed / variant / "raw_train.csv")
              for variant in scope["variants"]}
    if len(set(hashes.values())) != 1:
        raise ValueError("Raw training partitions differ across preprocessing variants.")
    raw_hash = next(iter(hashes.values()))
    revision = git_revision()
    cv_settings = resolve_cv_settings(config)
    n_splits = cv_settings["n_splits"]
    calibration_size = cv_settings["calibration_size"]
    for variant in scope["variants"]:
        folder = processed / variant
        config_path = folder / "preprocessing_config.json"
        preprocess_config = PreprocessingConfig(**json.loads(config_path.read_text(encoding="utf-8")))
        X, y = load_untransformed_partition(folder, "train")
        # Prepare once per feature representation; every candidate refits its estimator.
        folds = cv.prepare_cv_folds(X, y, preprocess_config, n_splits=n_splits,
                                   calibration_size=calibration_size, random_state=config["seed"])
        best = None
        for family, params, prototype in specifications:
            with mlflow.start_run(run_name=f"cv-{variant}-{family}-{len(rows):03d}") as run:
                mlflow.set_tags({
                    "preprocessing_variant": variant, "model_family": family,
                    "mode": "quick", "evaluation_protocol": "cv",
                    "data_source": "raw_training_partition",
                    "training_execution_id": execution_id,
                    "candidate_index": str(len(rows)), "search_strategy": scope["search"],
                })
                if revision:
                    mlflow.set_tag("git_commit", revision)
                mlflow.log_params({
                    "variant": variant, "model": family, "config_sha256": config_sha256,
                    "data_version": raw_hash, "preprocessing_config_sha256": file_sha256(config_path),
                    "seed": config["seed"], "n_splits": n_splits,
                    "calibration_size": calibration_size,
                    "search_strategy": scope["search"],
                    "threshold_policy": "fold_fit_inner_calibration",
                    **{f"fixed_{k}": str(v) for k, v in config["models"][family]["fixed"].items()},
                    **{f"model_{k}": str(v) for k, v in params.items()},
                })
                # Let failures propagate so MLflow marks the run FAILED and the CLI exits nonzero.
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    result = cv.evaluate_cv_candidate(prototype, folds, on_fold=log_cv_fold)
                warning_count = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
                metrics = {f"cv_{key}": value for key, value in result.mean_metrics.items()}
                metrics.update({f"cv_{key}_std": value for key, value in result.std_metrics.items()})
                mlflow.log_metrics({**metrics, "convergence_warning_count": warning_count})
                row = {
                    "run_id": run.info.run_id, "variant": variant, "model": family,
                    "parameters": json.dumps(params, sort_keys=True),
                    **metrics, "convergence_warning_count": warning_count, "status": "FINISHED",
                }
                rows.append(row)
                if best is None or cv_rank(row) > cv_rank(best):
                    best = row
                LOG.info("CV %s / %s: F1=%.4f (+/- %.4f)", variant, family,
                         row["cv_f1"], row["cv_f1_std"])
            # Persist completed candidates even if a later candidate fails.
            pd.DataFrame(rows).sort_values(
                ["cv_f1", "cv_pr_auc", "cv_roc_auc"], ascending=False, kind="stable",
            ).to_csv(output / "cv_leaderboard.csv", index=False)
        if best is None:
            raise RuntimeError(f"No CV candidates completed for {variant}")
        winners[variant] = {"cross_validation": best}
    preferred = max(winners, key=lambda variant: cv_rank(winners[variant]["cross_validation"]))
    atomic_json_write(output / "cv_training_summary.json", {
        "execution_id": execution_id, "output_dir": str(output), "quick": True,
        "evaluation_protocol": "cv", "scope": scope, "config_sha256": config_sha256,
        "raw_training_sha256": raw_hash, "git_commit": revision, "n_splits": n_splits,
        "calibration_size": calibration_size, "candidate_count": len(rows),
        "successful_candidates": len(rows), "preferred_variant_by_cv": preferred,
        "winners": winners,
    })
    LOG.info("Preferred configuration by CV: %s", preferred)


def train(args):
    """
    Run model selection across preprocessing configurations.

    Training:
        Fit candidates using training data.

    Validation:
        Optimize thresholds and select winners.

    Testing:
        Evaluate each winning model once.

    Existing preprocessing artifacts are never modified.
    """

    evaluation = resolve_evaluation(args)
    config = load_training_config(getattr(args, "config", None))
    config, search = resolve_training_scope(config, args)
    # Resolve effective settings before creating any output or tracking artifacts.
    config["cross_validation"] = resolve_cv_settings(config, args)
    active_variants = config["variants"]
    scope = {
        "models": list(config["models"]),
        "variants": list(active_variants),
        "search": search,
    }
    processed = args.processed_dir.resolve()
    preprocessor_dir = args.preprocessor_dir.resolve()
    # Each invocation owns a unique directory, including quick runs.
    # Never reuse an existing directory or write into the DVC reference tree.
    output_root = args.output_dir.resolve()
    reference_dir = (ROOT / "models/reference").resolve()
    if output_root == reference_dir or reference_dir in output_root.parents or output_root in reference_dir.parents:
        raise ValueError("Training output directory must be separate from models/reference")
    output_root.mkdir(parents=True, exist_ok=True)
    execution_id = uuid.uuid4().hex
    output = output_root / execution_id
    output.mkdir(exist_ok=False)
    LOG.info("Training execution ID: %s", execution_id)
    LOG.info("Training output: %s", output)

    db = args.mlflow_db.resolve()
    artifacts = args.mlflow_artifacts.resolve()

    # Record configuration and initialize tracking for either evaluation protocol.
    config_sha256 = file_sha256(Path(getattr(args, "config", None) or ROOT / "configs/config.yml"))
    configure_mlflow(args.experiment, db, artifacts)
    if evaluation == "cv":
        run_cv_search(config, scope, processed, output, execution_id, config_sha256)
        return

    # Only the full lifecycle may open the held-out test CSVs.
    datasets = {
        variant: load_variant(
            processed,
            preprocessor_dir,
            variant,
            include_test=not args.quick,
        )
        for variant in active_variants
    }

    leaderboard = []
    winners = {}

    # The mode governs safety; the search strategy governs enumeration.
    specifications = candidates(search == "first", config=config)

    LOG.info(
        "Training %s candidates across %s variants",
        len(specifications) * len(active_variants),
        len(active_variants),
    )

    # --------------------------------------------------------
    # Train candidate models
    # --------------------------------------------------------

    for variant, data in datasets.items():

        best = None
        best_model = None

        for family, params, prototype in specifications:

            run_name = (
                f"{variant}-{family}-{len(leaderboard):03d}"
            )

            with mlflow.start_run(
                run_name=run_name
            ) as run:

                mlflow.set_tags({
                    "preprocessing_variant": variant,
                    "model_family": family,
                    "data_source": (
                        "saved_preprocessing_partitions"
                    ),
                    "mode": (
                        "quick" if args.quick else "full"
                    ),
                    # Associate every candidate with its training execution.
                    "training_execution_id": execution_id,
                    # Preserve the candidate's original evaluation order for
                    # deterministic tie-breaking in MLflow comparisons.
                    "candidate_index": str(len(leaderboard)),
                    "search_strategy": search,
                })

                mlflow.log_params({
                    "variant": variant,
                    "model": family,
                    "config_sha256": config_sha256,
                    "search_strategy": search,
                    "selected_models": json.dumps(scope["models"]),
                    "selected_variants": json.dumps(scope["variants"]),
                    "data_version": file_sha256(processed / variant / "X_train.csv"),
                    "seed": config["seed"],
                    **{f"fixed_{k}": str(v) for k, v in config["models"][family]["fixed"].items()},
                    **{
                        f"model_{key}": str(value)
                        for key, value in params.items()
                    },
                })

                try:

                    # Fit using training data only.
                    model = clone(prototype)

                    with warnings.catch_warnings(
                        record=True
                    ) as caught_warnings:

                        warnings.simplefilter(
                            "always",
                            ConvergenceWarning,
                        )

                        model.fit(
                            data.X_train,
                            data.y_train,
                        )

                    convergence_warnings = [
                        str(warning.message)
                        for warning in caught_warnings
                        if issubclass(
                            warning.category,
                            ConvergenceWarning,
                        )
                    ]

                    mlflow.log_metric(
                        "convergence_warning_count",
                        len(convergence_warnings),
                    )

                    mlflow.set_tag(
                        "converged",
                        not bool(convergence_warnings),
                    )

                    if convergence_warnings:

                        LOG.warning(
                            "%s / %s: %s convergence warning(s)",
                            variant,
                            family,
                            len(convergence_warnings),
                        )

                        mlflow.set_tag(
                            "convergence_warning",
                            convergence_warnings[0][:500],
                        )

                    # Interpret predictions through the binary task module.
                    probabilities = binary_task.positive_probabilities(
                        model, data.X_val
                    )

                    threshold = optimal_threshold(
                        data.y_val,
                        probabilities,
                    )

                    metrics = evaluate(
                        data.y_val,
                        probabilities,
                        threshold,
                    )

                    default_f1 = binary_task.default_f1(
                        data.y_val, probabilities
                    )

                    # Log validation metrics.
                    mlflow.log_metrics({
                        f"val_{key}": value
                        for key, value in metrics.items()
                        if (
                            key != "confusion_matrix"
                            and value is not None
                        )
                    })

                    mlflow.log_metric(
                        "val_f1_default_050",
                        float(default_f1),
                    )

                    mlflow.log_metric(
                        "optimal_threshold",
                        threshold,
                    )

                    # The plugin owns native MLflow model serialization.
                    ALGORITHM_REGISTRY.get(
                        family, task_type=BINARY_CLASSIFICATION
                    ).log_model(model)

                    row = {
                        "run_id": run.info.run_id,
                        "variant": variant,
                        "model": family,
                        "parameters": json.dumps(
                            params,
                            sort_keys=True,
                        ),
                        "optimal_threshold": threshold,
                        "val_f1_default_050": float(
                            default_f1
                        ),
                        **{
                            f"val_{key}": value
                            for key, value in metrics.items()
                            if key != "confusion_matrix"
                        },
                        "convergence_warning_count": len(
                            convergence_warnings
                        ),
                        "status": "FINISHED",
                    }

                    # Select best validation candidate.
                    if (
                        best is None
                        or rank(row) > rank(best)
                    ):
                        best = row
                        best_model = model

                except Exception as exc:

                    LOG.exception(
                        "Candidate failed: %s / %s / %s",
                        variant,
                        family,
                        params,
                    )

                    mlflow.set_tag(
                        "candidate_error",
                        str(exc)[:500],
                    )

                    row = {
                        "run_id": run.info.run_id,
                        "variant": variant,
                        "model": family,
                        "parameters": json.dumps(
                            params,
                            sort_keys=True,
                        ),
                        "status": "FAILED",
                        "error": str(exc),
                    }

                leaderboard.append(row)

        if best is None or best_model is None:
            raise RuntimeError(
                f"All candidates failed for {variant}"
            )

        winners[variant] = (
            best,
            best_model,
            data,
        )

    # Quick mode is a smoke check. Any failed candidate means the check fails,
    # even if another candidate in the same preprocessing variant succeeded.
    if args.quick:
        failures = [row for row in leaderboard if row["status"] != "FINISHED"]
        if failures:
            raise RuntimeError(
                f"Quick smoke check failed: {len(failures)} candidate(s) failed; "
                f"first failure: {failures[0]['variant']}/{failures[0]['model']}: "
                f"{failures[0].get('error', 'unknown error')}"
            )

    # --------------------------------------------------------
    # Save experiment leaderboard
    # --------------------------------------------------------

    leaderboard_frame = pd.DataFrame(
        leaderboard
    )

    leaderboard_frame.sort_values(
        [
            "status",
            "val_f1",
            "val_pr_auc",
            "val_roc_auc",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
        na_position="last",
    ).to_csv(
        output / "sweep_leaderboard.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Select preferred configuration using validation
    # --------------------------------------------------------

    preferred = max(
        winners,
        key=lambda variant: rank(
            winners[variant][0]
        ),
    )

    summary = {
        "execution_id": execution_id,
        "output_dir": str(output),
        "preferred_variant_by_validation": preferred,
        "quick": args.quick,
        "scope": scope,
        "config_sha256": config_sha256,
        "candidate_count": len(leaderboard),
        "successful_candidates": sum(
            row["status"] == "FINISHED"
            for row in leaderboard
        ),
        "winners": {},
    }

    # --------------------------------------------------------
    # Package and evaluate winning models
    # --------------------------------------------------------

    for variant, (
        best,
        classifier,
        data,
    ) in winners.items():

        # Quick mode is a smoke check only: training, validation and MLflow
        # logging have already succeeded. Do not package a deployable
        # pipeline or evaluate the held-out test partition.
        if args.quick:
            summary["winners"][variant] = {"validation": best}
            LOG.info(
                "Quick check %s: %s, validation F1=%.4f",
                variant, best["model"], best["val_f1"],
            )
            continue

        variant_dir = output / variant

        variant_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Load the original fitted preprocessor.
        fitted_preprocessor = joblib.load(
            data.preprocessor_path
        )

        # Verify feature names and ordering.
        feature_names = list(
            data.X_train.columns
        )

        if hasattr(
            fitted_preprocessor,
            "get_feature_names_out",
        ):

            expected = list(
                fitted_preprocessor.get_feature_names_out()
            )

            if expected != feature_names:
                raise ValueError(
                    "Saved preprocessor feature names "
                    f"do not match CSV columns for {variant}"
                )

        # Build corrected deployable pipeline.
        deployable = build_deployable_pipeline(
            fitted_preprocessor=fitted_preprocessor,
            classifier=classifier,
            feature_names=feature_names,
            sample_features=data.X_train,
        )

        # Do not call deployable.fit().
        # Both preprocessing and classifier are already fitted.

        model_path = (
            variant_dir / "model_pipeline.joblib"
        )

        joblib.dump(
            deployable,
            model_path,
        )

        # Save the selected validation threshold.
        threshold_path = (
            variant_dir / "decision_threshold.json"
        )

        threshold_path.write_text(
            json.dumps(
                {
                    "threshold": best["optimal_threshold"],
                    "positive_class": binary_task.POSITIVE_CLASS,
                    "variant": variant,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        winner_info = {
            "validation": best,
            "model_path": str(model_path),
            "preprocessor_sha256": file_sha256(
                data.preprocessor_path
            ),
        }

        # ----------------------------------------------------
        # Evaluate on held-out test data
        # ----------------------------------------------------

        if not args.quick:

            test_probabilities = binary_task.positive_probabilities(
                classifier, data.X_test
            )

            test_metrics = evaluate(
                data.y_test,
                test_probabilities,
                best["optimal_threshold"],
            )

            winner_info["test"] = test_metrics

            with mlflow.start_run(
                run_name=f"final-test-{variant}"
            ):

                mlflow.set_tags({
                    "preprocessing_variant": variant,
                    "stage": "final_test",
                    "selected_run_id": best["run_id"],
                    "selected_by": "validation_f1",
                })

                mlflow.log_metrics({
                    f"test_{key}": value
                    for key, value in test_metrics.items()
                    if (
                        key != "confusion_matrix"
                        and value is not None
                    )
                })

                mlflow.log_metric(
                    "decision_threshold",
                    best["optimal_threshold"],
                )

                mlflow.log_artifact(
                    str(model_path),
                    artifact_path="model",
                )

                mlflow.log_artifact(
                    str(threshold_path),
                    artifact_path="model",
                )

        summary["winners"][variant] = winner_info

        LOG.info(
            "Winner %s: %s, validation F1=%.4f%s",
            variant,
            best["model"],
            best["val_f1"],
            (
                f", test F1={winner_info['test']['f1']:.4f}"
                if not args.quick
                else ""
            ),
        )

    # --------------------------------------------------------
    # Save the overall validation winner as a run-local candidate.
    # This is NOT a promotion. The DVC reference remains untouched.
    # --------------------------------------------------------

    if not args.quick:
        best_info, selected_classifier, selected_data = winners[preferred]
        source_model = output / preferred / "model_pipeline.joblib"
        source_threshold = output / preferred / "decision_threshold.json"

        if not source_model.is_file() or not source_threshold.is_file():
            raise FileNotFoundError("Winning model or threshold artifact is missing")

        threshold_info = json.loads(source_threshold.read_text(encoding="utf-8"))
        threshold = float(threshold_info["threshold"])
        verify_saved_model(source_model, selected_data, selected_classifier, threshold)

        # Save a self-contained, verified model and metadata in this execution.
        # An explicit, separate promotion procedure can later select this pair.
        candidate_model = output / "best_model_pipeline.joblib"
        candidate_metadata = output / "best_model_metadata.json"
        model_bytes = source_model.read_bytes()
        with tempfile.NamedTemporaryFile(
            dir=output, prefix=".best_model_", suffix=".joblib", delete=False,
        ) as handle:
            temporary_model = Path(handle.name)
        try:
            temporary_model.write_bytes(model_bytes)
            if file_sha256(temporary_model) != file_sha256(source_model):
                raise ValueError("Model checksum changed during copying")
            verify_saved_model(temporary_model, selected_data, selected_classifier, threshold)
            os.replace(temporary_model, candidate_model)
        finally:
            temporary_model.unlink(missing_ok=True)

        metadata = {
            "model_family": best_info["model"],
            "preprocessing_variant": preferred,
            "decision_threshold": threshold,
            "positive_class": threshold_info["positive_class"],
            "validation_f1": best_info["val_f1"],
            "validation_pr_auc": best_info["val_pr_auc"],
            "validation_roc_auc": best_info["val_roc_auc"],
            "mlflow_run_id": best_info["run_id"],
            "execution_id": execution_id,
            "git_commit": git_revision(),
            "dataset_sha256": dataset_fingerprints(processed, preferred),
            "preprocessor_sha256": file_sha256(selected_data.preprocessor_path),
            "source_model_path": str(source_model),
            "model_filename": candidate_model.name,
            "model_sha256": file_sha256(candidate_model),
            "training_mode": "quick" if args.quick else "full",
            "scope": scope,
        }
        atomic_json_write(candidate_metadata, metadata)

        # Attach the candidate artifacts to the selected MLflow run.
        # Do not set a "promoted" tag: selection is not promotion.
        with mlflow.start_run(run_id=best_info["run_id"]):
            mlflow.set_tag("training_execution_id", execution_id)
            mlflow.set_tag("model_sha256", metadata["model_sha256"])
            mlflow.set_tag("candidate_for_promotion", "true")
            if metadata["git_commit"]:
                mlflow.set_tag("git_commit", metadata["git_commit"])
            mlflow.log_artifact(str(candidate_metadata), artifact_path="candidate")
            mlflow.log_artifact(str(candidate_model), artifact_path="candidate")

        summary["candidate_model"] = str(candidate_model)
        summary["candidate_metadata"] = str(candidate_metadata)
        LOG.info("Saved unpromoted model candidate: %s", candidate_model)
        LOG.info("Saved candidate metadata: %s", candidate_metadata)

    # --------------------------------------------------------
    # Save training summary
    # --------------------------------------------------------

    summary_path = output / (
        "quick_training_summary.json"
        if args.quick
        else "training_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    LOG.info(
        "Preferred configuration by validation: %s",
        preferred,
    )

    LOG.info(
        "Saved leaderboard and summary under %s",
        output,
    )


# ============================================================
# 9. Command-line arguments
# ============================================================

def parse_args():
    """Parse CLI options and resolve YAML-based defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/config.yml",
        help="YAML configuration file (default: configs/config.yml).",
    )
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--preprocessor-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mlflow-db", type=Path, default=None)
    parser.add_argument("--mlflow-artifacts", type=Path, default=None)
    parser.add_argument("--experiment", default=None)
    lifecycle = parser.add_mutually_exclusive_group()
    lifecycle.add_argument(
        "--mode", choices=("validation", "full"), default=None,
        help="Validation-only development or full held-out evaluation (default: full).",
    )
    lifecycle.add_argument(
        "--quick", action="store_true",
        help="Backward-compatible alias for --mode validation.",
    )
    parser.add_argument(
        "--models", nargs="+", choices=ALGORITHM_REGISTRY.names(),
        help="Select model families enabled in the YAML configuration.",
    )
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS,
        help="Select preprocessing variants enabled in the YAML configuration.",
    )
    parser.add_argument(
        "--search", choices=("first", "grid"), default=None,
        help="Search depth (default: first in quick mode, grid in full mode).",
    )
    parser.add_argument(
        "--evaluation", choices=("holdout", "cv"), default="holdout",
        help="Saved validation (default), or training-only CV in validation mode.",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=None,
        help="Override YAML cross_validation.n_splits (CV only).",
    )
    parser.add_argument(
        "--calibration-size", type=float, default=None,
        help="Override YAML cross_validation.calibration_size (CV only).",
    )
    parser.add_argument("--show-mlflow-command", action="store_true")
    args = parser.parse_args()
    # Keep the existing internal quick flag and output contracts for CI and callers.
    args.mode = args.mode or ("validation" if args.quick else "full")
    args.quick = args.mode == "validation"
    try:
        config = load_training_config(args.config)
        resolve_training_scope(config, args)
        resolve_evaluation(args)
        resolve_cv_settings(config, args)
    except (ValueError, FileNotFoundError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    # Resolve relative paths against repository root, not the shell's cwd.
    for attr in ("processed_dir", "preprocessor_dir", "output_dir",
                 "mlflow_db", "mlflow_artifacts"):
        if getattr(args, attr) is None:
            setattr(args, attr, ROOT / config["paths"][attr])
    if args.experiment is None:
        args.experiment = config["mlflow"]["experiment"]
    return args


# ============================================================
# 10. Entry point
# ============================================================

def main():
    """Configure logging and run the requested training command."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    args = parse_args()

    if args.show_mlflow_command:

        uri = (
            "sqlite:///"
            f"{args.mlflow_db.resolve().as_posix()}"
        )

        print(
            f'mlflow ui --backend-store-uri "{uri}" '
            "--port 5000"
        )

        return

    # Report the final execution status without hiding training errors.
    mode = "Quick" if args.quick else "Full"
    try:
        train(args)
    except Exception as exc:
        LOG.error("%s Training Failed: %s", mode, exc)
        raise  # Preserve the original traceback and nonzero CLI exit status.

    if args.quick:
        LOG.info("Quick Training Successful")
    else:
        # Full mode may complete with individual candidate failures.
        LOG.info("Full Training Completed")


if __name__ == "__main__":
    main()
