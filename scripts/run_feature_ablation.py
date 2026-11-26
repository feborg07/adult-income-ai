"""Compare fixed candidates on paired training-only CV representations.

Usage: python -m scripts.run_feature_ablation
       python -m scripts.run_feature_ablation --representations control no_fnlwgt no_fnlwgt_race no_fnlwgt_sex no_fnlwgt_demographics
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import logging
from pathlib import Path
import uuid
import warnings

import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning

from src.data.preprocess import (
    FEATURE_COLUMNS, PreprocessingConfig, create_cv_folds, load_untransformed_partition, predictor_group_ids,
)
from src.models import cross_validation as cv, train as training
from src.models.algorithms.base import BINARY_CLASSIFICATION

LOG = logging.getLogger(__name__)
ROOT = training.ROOT


def load_plan(path: Path) -> dict:
    """Load and validate the fixed experiment plan, including an unablated control."""
    plan = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or set(plan) != {"representations", "candidates"}:
        raise ValueError("Expected representations and candidates in the ablation plan.")
    representations = plan["representations"]
    if not isinstance(representations, dict) or representations.get("control") != []:
        raise ValueError("The plan requires an empty control representation.")
    seen = set()
    for name, excluded in representations.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Invalid representation name.")
        canonical = PreprocessingConfig(excluded_features=excluded).excluded_features
        if canonical in seen:
            raise ValueError("Duplicate feature representations.")
        seen.add(canonical)
    if not isinstance(plan["candidates"], dict) or not plan["candidates"]:
        raise ValueError("The plan requires fixed candidates.")
    for name, spec in plan["candidates"].items():
        if not isinstance(name, str) or not name or not isinstance(spec, dict):
            raise ValueError("Invalid candidate specification.")
        if set(spec) != {"family", "variant", "parameters", "historical_cv_f1"}:
            raise ValueError(f"Invalid candidate fields: {name}")
        if spec["variant"] not in training.VARIANTS or not isinstance(spec["parameters"], dict):
            raise ValueError(f"Invalid candidate variant or parameters: {name}")
        score = spec["historical_cv_f1"]
        if type(score) not in (float, int) or not 0 <= score <= 1:
            raise ValueError(f"Invalid historical score: {name}")
        training.ALGORITHM_REGISTRY.get(spec["family"], task_type=BINARY_CLASSIFICATION)
    return plan


def common_grouping_exclusions(plan: dict) -> tuple[str, ...]:
    """Use every planned exclusion, not only the currently selected representations."""
    excluded = {feature for features in plan["representations"].values() for feature in features}
    return PreprocessingConfig(excluded_features=list(excluded)).excluded_features


def audit_overlap(X: pd.DataFrame, indices, excluded: tuple[str, ...]) -> list[dict]:
    """Count exact retained-raw-vector overlap between each pair of fold subsets.

    This diagnostic does not regroup or remove rows and is not an audit of all
    possible collisions caused by subsequent learned feature transformations.
    """
    groups = predictor_group_ids(X.drop(columns=list(excluded)))
    result = []
    for number, parts in enumerate(indices, 1):
        for i, j, name in ((0, 1, "fit_calibration"), (0, 2, "fit_scoring"),
                           (1, 2, "calibration_scoring")):
            left, right = groups[parts[i]], groups[parts[j]]
            overlap = np.intersect1d(left, right)
            result.append({"fold": number, "pair": name, "shared_groups": len(overlap),
                           "left_rows": int(np.isin(left, overlap).sum()),
                           "right_rows": int(np.isin(right, overlap).sum()),
                           "left_total": len(left), "right_total": len(right)})
    return result


def assert_paired_folds(folds, indices, X: pd.DataFrame) -> None:
    """Reject a preprocessing run if any fold membership differs from the control."""
    if len(folds) != len(indices):
        raise RuntimeError("Fold count changed during ablation.")
    for fold, parts in zip(folds, indices):
        for actual, part in zip((fold.X_fit, fold.X_calibration, fold.X_scoring), parts):
            if not actual.index.equals(X.iloc[part].index):
                raise RuntimeError("Fold membership changed during ablation.")


def paired_comparison(rows: list[dict], fold_rows: list[dict]) -> list[dict]:
    """Compute candidate-specific F1 changes against the paired control."""
    comparisons = []
    controls = {row["candidate"]: row for row in rows if row["representation"] == "control"}
    for row in rows:
        name, representation = row["candidate"], row["representation"]
        if representation == "control":
            continue
        control = controls[name]
        baseline = {f["fold"]: f["f1"] for f in fold_rows
                    if f["candidate"] == name and f["representation"] == "control"}
        scores = {f["fold"]: f["f1"] for f in fold_rows
                  if f["candidate"] == name and f["representation"] == representation}
        if not baseline or baseline.keys() != scores.keys():
            raise ValueError("Paired comparisons require identical fold numbers.")
        changes = [scores[k] - baseline[k] for k in sorted(baseline)]
        comparisons.append({"candidate": name, "representation": representation,
                            "control_cv_f1": control["cv_f1"], "ablation_cv_f1": row["cv_f1"],
                            "delta_cv_f1": float(np.mean(changes)),
                            "delta_fold_std": float(np.std(changes)), "fold_deltas": changes})
    return comparisons


def run_ablation(args) -> Path:
    """Run fixed paired experiments using only the original raw training rows."""
    config = training.load_training_config(args.config)
    plan = load_plan(args.plan)
    selected = args.representations
    if (not selected or "control" not in selected or len(set(selected)) != len(selected)
            or set(selected) - plan["representations"].keys() or len(selected) < 2):
        raise ValueError("Select control and at least one unique known ablation.")
    selected = [name for name in plan["representations"] if name in selected]
    paths = config["paths"]
    processed = Path(args.processed_dir or ROOT / paths["processed_dir"]).resolve()
    root = Path(args.output_dir or ROOT / paths["output_dir"]).resolve()
    reference = (ROOT / "models/reference").resolve()
    if root == reference or root in reference.parents or reference in root.parents:
        raise ValueError("Ablation output must be separate from models/reference.")
    variants = {spec["variant"] for spec in plan["candidates"].values()}
    raw_hashes = {v: training.file_sha256(processed / v / "raw_train.csv") for v in variants}
    if len(set(raw_hashes.values())) != 1:
        raise ValueError("Raw training partitions differ across variants.")
    raw_hash = next(iter(raw_hashes.values()))
    X, y = load_untransformed_partition(processed / sorted(variants)[0], "train")
    settings = {**config["cross_validation"], "random_state": config["seed"]}
    grouping_exclusions = common_grouping_exclusions(plan)
    settings["grouping_excluded_features"] = grouping_exclusions
    indices = create_cv_folds(X, y, **settings)
    # Fail before tracking or fitting if the common basis is not isolated.
    common_audit = audit_overlap(X, indices, grouping_exclusions)
    if any(row["shared_groups"] for row in common_audit):
        raise RuntimeError("Common representation predictor groups overlap.")
    audit = [{"representation": name, **row}
             for name in selected
             for row in audit_overlap(X, indices, tuple(plan["representations"][name]))]
    if any(row["shared_groups"] for row in audit):
        raise RuntimeError("Candidate representation predictor groups overlap.")
    fold_balance = [{"fold": number, "subset": name, "rows": len(part),
                     "positive_rate": float(y.iloc[part].mean())}
                    for number, parts in enumerate(indices, 1)
                    for name, part in zip(("fit", "calibration", "scoring"), parts)]
    fold_hash = hashlib.sha256(json.dumps([[p.tolist() for p in fold] for fold in indices]).encode()).hexdigest()
    base_configs = {v: PreprocessingConfig(**json.loads((processed / v / "preprocessing_config.json").read_text(encoding="utf-8"))) for v in variants}
    if any(c.excluded_features for c in base_configs.values()):
        raise ValueError("Control preprocessing configurations must not exclude features.")
    # Freeze all model specifications before any training or tracking side effects.
    prototypes = {name: training.ALGORITHM_REGISTRY.get(spec["family"], task_type=BINARY_CLASSIFICATION).build(spec["parameters"], config["seed"]) for name, spec in plan["candidates"].items()}
    execution_id = uuid.uuid4().hex
    output = root / execution_id
    output.mkdir(parents=True, exist_ok=False)
    LOG.info("Ablation execution: %s", output)
    plan_hash = training.file_sha256(args.plan)
    config_hash = training.file_sha256(args.config)
    experiment = args.experiment or config["mlflow"]["experiment"] + "-ablation-representation-safe"
    training.configure_mlflow(experiment, Path(args.mlflow_db or ROOT / paths["mlflow_db"]),
                              Path(args.mlflow_artifacts or ROOT / paths["mlflow_artifacts"]))
    rows, fold_rows, cache = [], [], {}
    for representation in selected:
        excluded = PreprocessingConfig(excluded_features=plan["representations"][representation]).excluded_features
        for name, spec in plan["candidates"].items():
            variant = spec["variant"]
            key = (variant, representation)
            if key not in cache:
                preprocessor_config = replace(base_configs[variant], excluded_features=excluded)
                cache[key] = cv.prepare_cv_folds(X, y, preprocessor_config, **settings)
                assert_paired_folds(cache[key], indices, X)
            with training.mlflow.start_run(run_name=f"ablation-{name}-{representation}") as run:
                training.mlflow.set_tags({"evaluation_protocol": "cv_ablation", "mode": "quick",
                                          "training_execution_id": execution_id, "candidate": name,
                                          "grouping_policy": "common_retained_raw_predictors_v1",
                                          "representation": representation, "model_family": spec["family"],
                                          "preprocessing_variant": variant})
                training.mlflow.log_params({"data_version": raw_hash, "fold_sha256": fold_hash,
                    "plan_sha256": plan_hash, "config_sha256": config_hash,
                    "preprocessing_config_sha256": training.file_sha256(processed / variant / "preprocessing_config.json"),
                    "excluded_features": json.dumps(excluded),
                    "grouping_excluded_features": json.dumps(grouping_exclusions), "seed": config["seed"],
                    "n_splits": settings["n_splits"], "calibration_size": settings["calibration_size"],
                    "threshold_policy": "fold_fit_inner_calibration",
                    **{f"model_{k}": str(v) for k, v in spec["parameters"].items()}})
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    result = cv.evaluate_cv_candidate(prototypes[name], cache[key], on_fold=training.log_cv_fold)
                metrics = {f"cv_{k}": v for k, v in result.mean_metrics.items()}
                metrics.update({f"cv_{k}_std": v for k, v in result.std_metrics.items()})
                warning_count = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
                training.mlflow.log_metrics({**metrics, "convergence_warning_count": warning_count})
                run_id = run.info.run_id
            rows.append({"candidate": name, "family": spec["family"], "variant": variant,
                         "representation": representation, "excluded_features": json.dumps(excluded),
                         "fold_sha256": fold_hash, "grouping_policy": "common_retained_raw_predictors_v1",
                         "parameters": json.dumps(spec["parameters"], sort_keys=True), "run_id": run_id,
                         **metrics, "convergence_warning_count": warning_count})
            fold_rows.extend({"candidate": name, "representation": representation, "run_id": run_id,
                              "fold": f.number, "threshold": f.threshold, **f.metrics} for f in result.folds)
            LOG.info("%s / %s: F1=%.6f", name, representation, metrics["cv_f1"])
            # Keep successful candidates available if a later candidate fails.
            pd.DataFrame(rows).to_csv(output / "ablation_leaderboard.csv", index=False)
            pd.DataFrame(fold_rows).to_csv(output / "ablation_fold_metrics.csv", index=False)
    comparisons = paired_comparison(rows, fold_rows)
    pd.DataFrame(audit).to_csv(output / "ablation_duplicate_audit.csv", index=False)
    summary = {"execution_id": execution_id, "evaluation_protocol": "cv_ablation",
               "data_source": "raw_training_partition", "raw_training_sha256": raw_hash,
               "fold_sha256": fold_hash, "settings": settings, "git_commit": training.git_revision(),
               "grouping_policy": "common_retained_raw_predictors_v1",
               "grouping_features": [name for name in FEATURE_COLUMNS if name not in grouping_exclusions],
               "common_group_count": len(np.unique(predictor_group_ids(X.drop(columns=list(grouping_exclusions))))),
               "common_group_overlap": common_audit, "fold_balance": fold_balance,
               "code_identity": training.optimization_code_identity(), "runner_sha256": training.file_sha256(Path(__file__)),
               "config_sha256": config_hash, "plan_sha256": plan_hash, "plan": plan,
               "representations": selected, "candidate_count": len(rows), "comparisons": comparisons,
               "historical_control_comparison": [{"candidate": r["candidate"], "cv_f1": r["cv_f1"],
                   "directly_comparable": False, "reason": "Different grouping protocol and folds",
                   "historical_cv_f1": plan["candidates"][r["candidate"]]["historical_cv_f1"],
                   "difference": r["cv_f1"] - plan["candidates"][r["candidate"]]["historical_cv_f1"]}
                   for r in rows if r["representation"] == "control"], "mlflow_experiment": experiment}
    training.atomic_json_write(output / "ablation_summary.json", summary)
    LOG.info("Ablation successful: %s candidates", len(rows))
    return output


def parse_args(argv=None):
    """Parse a validation-only experiment; no full/test/promotion options exist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/config.yml")
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/feature_ablation.yml")
    parser.add_argument("--representations", nargs="+", default=["control", "no_fnlwgt"])
    for field in ("processed-dir", "output-dir", "mlflow-db", "mlflow-artifacts"):
        parser.add_argument("--" + field, type=Path)
    parser.add_argument("--experiment")
    return parser.parse_args(argv)


def main() -> None:
    """Execute the selected controlled experiment."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_ablation(parse_args())


if __name__ == "__main__":
    main()
