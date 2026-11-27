"""Compare Adult Income AI candidates using the MLflow tracking store.

Usage (from the repository root):
    python -m scripts.compare_experiments
    python -m scripts.compare_experiments --top 10
    python -m scripts.compare_experiments --variant baseline
    python -m scripts.compare_experiments --execution-id <FULL_EXECUTION_ID>
    python -m scripts.compare_experiments --mode all  # include validation runs
    python -m scripts.compare_experiments --evaluation cv --execution-id <CV_EXECUTION_ID>
    python -m scripts.compare_experiments --output reports/mlflow_comparison.csv

By default, compare full-mode holdout candidates by validation F1.
CV comparisons default to validation-only runs and rank by mean fold F1.
Never combine evaluation protocols or incompatible CV settings in one ranking.
This script never evaluates the test partition or promotes models.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import mlflow
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
METRIC = "metrics.val_f1"  # Retain the historical holdout metric constant.
CV_COMPARISON_PARAMS = (
    "params.data_version", "params.seed", "params.n_splits",
    "params.calibration_size", "params.threshold_policy",
)
CV_REPRESENTATION_PARAMS = (
    "params.representation", "params.excluded_features", "params.grouping_excluded_features",
    "params.grouping_policy", "params.fold_sha256",
)
EXECUTION_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def load_settings(config_path: Path) -> tuple[str, Path, list[str]]:
    """Read experiment, tracking database, and variant order from YAML."""
    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Training YAML must contain a mapping.")
    try:
        experiment = config["mlflow"]["experiment"]
        db_path = Path(config["paths"]["mlflow_db"])
        variants = config["variants"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Training YAML requires mlflow.experiment, paths.mlflow_db, and variants."
        ) from exc
    if not isinstance(experiment, str) or not experiment.strip():
        raise ValueError("mlflow.experiment must be a nonempty string.")
    if not str(db_path).strip():
        raise ValueError("paths.mlflow_db must be a nonempty path.")
    if (not isinstance(variants, list) or not variants
            or any(not isinstance(v, str) or not v.strip() for v in variants)
            or len(set(variants)) != len(variants)):
        raise ValueError("variants must be a nonempty list of unique names.")
    # Relative paths follow the same convention as the training runner: repo root.
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    return experiment, db_path.resolve(), variants


def ranking_metrics(evaluation: str) -> tuple[str, str, str]:
    """Return the three ranking metrics for an explicit evaluation protocol."""
    if evaluation not in ("holdout", "cv"):
        raise ValueError("Evaluation must be holdout or cv.")
    prefix = "cv" if evaluation == "cv" else "val"
    return tuple(f"metrics.{prefix}_{name}" for name in ("f1", "pr_auc", "roc_auc"))


def validate_cv_comparability(runs: pd.DataFrame) -> None:
    """Reject missing or incompatible CV data, seed, split or threshold settings."""
    # Historical-only comparisons remain supported, but do not mix old and new runs.
    has_identity = any(column in runs and runs[column].notna().any()
                       for column in CV_REPRESENTATION_PARAMS)
    columns = CV_COMPARISON_PARAMS + (CV_REPRESENTATION_PARAMS if has_identity else ())
    for column in columns:
        if column not in runs or runs[column].isna().any():
            raise ValueError(f"Missing CV comparison metadata: {column}.")
        values = runs[column].astype(str).str.strip()
        if values.eq("").any():
            raise ValueError(f"Missing CV comparison metadata: {column}.")
        if values.nunique() != 1:
            raise ValueError(
                f"Incompatible CV comparison metadata: {column}. "
                "Use --execution-id to compare one compatible execution."
            )

    column = "params.effective_preprocessing_sha256"
    if column in runs and runs[column].notna().any():
        if runs[column].isna().any() or runs[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing CV comparison metadata: {column}.")
        if runs.groupby("tags.preprocessing_variant")[column].nunique().gt(1).any():
            raise ValueError(f"Incompatible CV comparison metadata: {column} within a variant.")


def rank_candidates(
    runs: pd.DataFrame,
    *,
    mode: str | None = None,
    variant: str | None = None,
    execution_id: str | None = None,
    execution_ids: list[str] | None = None,
    variant_order: list[str] | None = None,
    evaluation: str = "holdout",
) -> pd.DataFrame:
    """Select successful candidate runs with valid F1, sorted best first.

    The training pipeline logs separate 'final_test' records. Requiring a
    candidate family and variant excludes these and unrelated MLflow runs.
    """
    metric, pr_auc, roc_auc = ranking_metrics(evaluation)
    if mode == "validation":
        mode = "quick"  # Existing training runs retain the quick tag.
    columns = [
        "run_id", "start_time", "tags.model_family", "tags.preprocessing_variant",
        "tags.mode", "tags.evaluation_protocol", metric, pr_auc, roc_auc,
        "params.config_sha256", "params.data_version",
        "tags.training_execution_id", "tags.candidate_index",
    ]
    if runs.empty or metric not in runs.columns:
        return pd.DataFrame(columns=columns)

    result = runs.copy()
    if "status" in result.columns:
        result = result.loc[result["status"] == "FINISHED"]
    # Failed holdout candidates can have a FINISHED MLflow lifecycle when
    # training catches an exception after logging metrics; exclude them too.
    if "tags.candidate_error" in result.columns:
        error = result["tags.candidate_error"]
        result = result.loc[error.isna() | error.astype(str).str.strip().eq("")]
    if "tags.stage" in result.columns:
        result = result.loc[~result["tags.stage"].eq("final_test").fillna(False)]
    if "tags.evaluation_protocol" in result.columns:
        protocol = result["tags.evaluation_protocol"]
        # Older holdout records have no protocol tag. CV must be explicit.
        mask = protocol.eq("cv") if evaluation == "cv" else (
            protocol.isna() | protocol.eq("holdout")
        )
        result = result.loc[mask.fillna(False)]
    elif evaluation == "cv":
        return pd.DataFrame(columns=columns)
    for required in ("tags.model_family", "tags.preprocessing_variant"):
        if required not in result.columns:
            return pd.DataFrame(columns=columns)
        result = result.loc[result[required].notna() & result[required].astype(str).ne("")]

    if mode is not None:
        if "tags.mode" not in result.columns:
            return pd.DataFrame(columns=columns)
        result = result.loc[result["tags.mode"] == mode]
    if variant is not None:
        result = result.loc[result["tags.preprocessing_variant"] == variant]
    if execution_id is not None and execution_ids is not None:
        raise ValueError("Specify only one execution filter.")
    selected_ids = execution_ids if execution_ids is not None else ([execution_id] if execution_id is not None else None)
    if selected_ids is not None:
        if "tags.training_execution_id" not in result.columns:
            return pd.DataFrame(columns=columns)
        result = result.loc[result["tags.training_execution_id"].isin(selected_ids)]

    result[metric] = pd.to_numeric(result[metric], errors="coerce")
    result = result.loc[result[metric].between(0, 1, inclusive="both")]
    for tie_breaker in (pr_auc, roc_auc):
        if tie_breaker not in result.columns:
            result[tie_breaker] = float("nan")
        result[tie_breaker] = pd.to_numeric(result[tie_breaker], errors="coerce")
        valid = result[tie_breaker].between(0, 1, inclusive="both")
        if evaluation == "cv":
            # The CV engine always logs both tie-breakers; incomplete runs
            # cannot be ranked consistently with the trainer.
            result = result.loc[valid]
        else:
            result[tie_breaker] = result[tie_breaker].where(valid)
    if selected_ids is not None and execution_ids is not None:
        missing = set(selected_ids) - set(result["tags.training_execution_id"])
        if missing:
            raise ValueError(f"No valid candidates for selected executions: {sorted(missing)}")
    if evaluation == "cv" and not result.empty:
        validate_cv_comparability(result)

    # train.py chooses the first candidate when all three metrics tie.
    # First means: YAML variant order, then candidate creation order.
    order = variant_order or ["baseline", "robust", "interactions"]
    priority = {name: index for index, name in enumerate(order)}
    result["_variant_priority"] = (
        result["tags.preprocessing_variant"].map(priority).fillna(len(order))
    )
    if "tags.candidate_index" not in result.columns:
        result["tags.candidate_index"] = pd.NA
    result["_candidate_priority"] = pd.to_numeric(
        result["tags.candidate_index"], errors="coerce"
    )
    if "start_time" not in result.columns:
        result["start_time"] = pd.NaT
    if "run_id" not in result.columns:
        result["run_id"] = ""
    result = result.sort_values(
        by=[metric, pr_auc, roc_auc,
            "_variant_priority", "_candidate_priority", "start_time", "run_id"],
        ascending=[False, False, False, True, True, True, True],
        kind="stable",
        na_position="last",
    )
    return result.drop(columns=["_variant_priority", "_candidate_priority"]).reset_index(drop=True)


def compare(
    *,
    config_path: Path,
    top: int = 10,
    mode: str | None = None,
    variant: str | None = None,
    execution_id: str | None = None,
    execution_ids: list[str] | None = None,
    output: Path | None = None,
    evaluation: str = "holdout",
) -> pd.DataFrame:
    """Query MLflow, display a leaderboard and optionally export it."""
    metric, pr_auc, roc_auc = ranking_metrics(evaluation)
    mode = mode if mode is not None else ("quick" if evaluation == "cv" else "full")
    if mode not in ("full", "quick", "validation", "all"):
        raise ValueError("Invalid comparison mode.")
    if evaluation == "cv" and mode == "full":
        raise ValueError("CV comparisons require validation-only mode or all.")
    if top < 1:
        raise ValueError("--top must be at least 1")
    if execution_id is not None and execution_ids is not None:
        raise ValueError("Specify only one execution filter.")
    selected_ids = execution_ids if execution_ids is not None else ([execution_id] if execution_id is not None else [])
    if (any(not EXECUTION_ID_RE.fullmatch(value) for value in selected_ids)
            or len(set(selected_ids)) != len(selected_ids) or execution_ids == []):
        raise ValueError("--execution-id/--execution-ids requires unique 32-character lowercase hex IDs")
    experiment_name, db_path, variant_order = load_settings(config_path)
    if variant is not None and variant not in variant_order:
        raise ValueError(f"Unknown preprocessing variant: {variant}.")
    if not db_path.is_file():
        raise FileNotFoundError(
            f"MLflow tracking database not found: {db_path}. "
            "Run training first (python -m src.models.train --quick)."
        )
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name!r}")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        output_format="pandas",
        max_results=100000,
    )
    ranked = rank_candidates(
        runs,
        mode=None if mode == "all" else mode,
        variant=variant,
        execution_id=execution_id,
        execution_ids=execution_ids,
        variant_order=variant_order,
        evaluation=evaluation,
    )
    if ranked.empty:
        raise ValueError(f"No completed {evaluation} training candidates with valid F1 were found.")

    # MLflow returns columns from both protocols in a shared experiment.
    # Keep the exported leaderboard specific to the selected protocol.
    other_prefix = "metrics.val_" if evaluation == "cv" else "metrics.cv_"
    ranked = ranked.drop(columns=[c for c in ranked.columns if c.startswith(other_prefix)])

    display_columns = [
        "run_id", "tags.model_family", "tags.preprocessing_variant",
        "tags.mode", metric, pr_auc, roc_auc,
        "metrics.cv_f1_std",
        "params.config_sha256", "params.data_version",
        "tags.training_execution_id",
    ]
    display_columns = [col for col in display_columns if col in ranked.columns]
    print(f"Experiment: {experiment_name}")
    print(f"Tracking DB: {db_path}")
    print(f"Evaluation protocol: {evaluation}")
    print(f"Valid candidates: {len(ranked)}")
    if execution_id is not None:
        print(f"Execution ID: {execution_id}")
    if execution_ids is not None:
        print(f"Execution IDs: {', '.join(execution_ids)}")
    print(f"Best run: {ranked.iloc[0]['run_id']}")
    label = "mean CV F1" if evaluation == "cv" else "validation F1"
    print(f"Best {label}: {ranked.iloc[0][metric]:.4f}")
    print("\nLeaderboard:")
    print(ranked.loc[:, display_columns].head(top).to_string(index=False))

    winner = ranked.iloc[0]
    params = {
        key.removeprefix("params."): value
        for key, value in winner.items()
        if key.startswith("params.") and pd.notna(value)
    }
    print("\nWinning run parameters:")
    for key, value in sorted(params.items()):
        print(f"  {key}: {value}")

    if output is not None:
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(output, index=False)
        print(f"\nExported complete leaderboard: {output}")
    return ranked


def main() -> int:
    """Parse comparison arguments and report failures through the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/config.yml")
    parser.add_argument("--top", type=int, default=10, help="Number of rows to display")
    parser.add_argument(
        "--mode", choices=("full", "quick", "validation", "all"), default=None,
        help="Lifecycle filter (default: full for holdout, validation for CV)",
    )
    parser.add_argument("--evaluation", choices=("holdout", "cv"), default="holdout")
    parser.add_argument("--variant", help="Preprocessing variant defined in YAML")
    executions = parser.add_mutually_exclusive_group()
    executions.add_argument(
        "--execution-id", type=str,
        help="Limit comparison to one training execution (32 hex characters)",
    )
    executions.add_argument("--execution-ids", nargs="+", help="Compare an explicit cohort of training executions")
    parser.add_argument("--output", type=Path, help="Optional CSV leaderboard output")
    args = parser.parse_args()
    try:
        compare(
            config_path=args.config,
            top=args.top,
            mode=args.mode,
            variant=args.variant,
            execution_id=args.execution_id,
            execution_ids=args.execution_ids,
            output=args.output,
            evaluation=args.evaluation,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
