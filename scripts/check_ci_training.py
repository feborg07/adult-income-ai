"""Run first-configuration CV and gate only the new execution's mean F1.

Usage: python -m scripts.check_ci_training --min-f1 0.65
The gate checks training health, not eligibility for model promotion.
"""

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

from scripts.run_pipeline import PROJECT_ROOT, execution_ids, training_runs_directory
from src.models.configuration import load_resolved_config


def validate_execution(runs_dir, previous_ids, config, min_f1):
    """Require one new complete CV smoke run and return its best mean F1."""
    if not math.isfinite(min_f1) or not 0 <= min_f1 <= 1:
        raise ValueError("Minimum F1 must be finite and within [0, 1].")
    created = execution_ids(runs_dir) - previous_ids
    if len(created) != 1:
        raise RuntimeError(f"Expected exactly one new CV execution; found {len(created)}.")
    execution_id = created.pop()
    run_dir = runs_dir / execution_id
    summary = json.loads((run_dir / "cv_training_summary.json").read_text(encoding="utf-8"))
    models = list(config["models"])
    variants = config["variants"]
    expected = {(variant, model) for variant in variants for model in models}
    count = len(expected)
    # Reject old holdout runs, incomplete runs and a narrowed search scope.
    if (not expected or not isinstance(summary, dict)
            or summary.get("execution_id") != execution_id
            or summary.get("quick") is not True
            or summary.get("evaluation_protocol") != "cv"
            or summary.get("scope") != {"models": models, "variants": variants, "search": "first"}
            or summary.get("n_splits") != config["cross_validation"]["n_splits"]
            or type(summary.get("candidate_count")) is not int
            or type(summary.get("successful_candidates")) is not int
            or summary["candidate_count"] != count
            or summary["successful_candidates"] != count):
        raise RuntimeError("Incomplete or incompatible CV smoke execution.")

    with (run_dir / "cv_leaderboard.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    scores = {}
    run_ids = set()
    for row in rows:
        key = (row.get("variant"), row.get("model"))
        run_id = row.get("run_id")
        if (key not in expected or key in scores or not run_id or run_id in run_ids
                or row.get("status") != "FINISHED"):
            raise RuntimeError("Unexpected, duplicate or unfinished CV candidate.")
        try:
            score = float(row["cv_f1"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Missing or invalid candidate CV F1.") from exc
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise RuntimeError("Candidate CV F1 is outside [0, 1] or nonfinite.")
        scores[key] = score
        run_ids.add(run_id)
    if scores.keys() != expected:
        raise RuntimeError("CV leaderboard does not cover the configured smoke scope.")
    best = max(scores.values())
    print(f"CV execution: {execution_id}\nCompleted candidates: {count}\n"
          f"Best mean CV F1: {best:.4f}\nMinimum mean CV F1: {min_f1:.4f}", flush=True)
    if best < min_f1:
        raise RuntimeError("Best mean CV F1 is below the CI smoke threshold.")
    return best


def run_check(min_f1):
    """Train using the configured CV scope, then validate fresh artifacts only."""
    if not math.isfinite(min_f1) or not 0 <= min_f1 <= 1:
        raise ValueError("Minimum F1 must be finite and within [0, 1].")
    config = load_resolved_config(PROJECT_ROOT / "configs/config.yml")
    runs_dir = training_runs_directory()
    previous_ids = execution_ids(runs_dir)
    # Explicitly prohibit the legacy held-out training path.
    subprocess.run([sys.executable, "-m", "src.models.train", "--mode", "validation",
                    "--evaluation", "cv", "--search", "first"], cwd=PROJECT_ROOT, check=True)
    return validate_execution(runs_dir, previous_ids, config, min_f1)


def main():
    """Parse the smoke threshold and fail the CI job on any validation error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-f1", type=float, default=0.65)
    args = parser.parse_args()
    try:
        run_check(args.min_f1)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"CI training check failed: {exc}\n")
    print("PASSED: Current CV execution meets the CI smoke threshold.")


if __name__ == "__main__":
    main()
