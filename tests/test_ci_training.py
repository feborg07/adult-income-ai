"""Test the execution-scoped CI gate without fitting expensive models."""

import csv
import json
import subprocess

import pytest

from scripts import check_ci_training as gate

RUN_ID = "a" * 32
OLD_ID = "b" * 32
CONFIG = {"models": {"xgboost": {}, "lightgbm": {}}, "variants": ["baseline", "robust"],
          "cross_validation": {"n_splits": 3}}


def write_execution(root, run_id=RUN_ID, scores=None, overrides=None, rows_override=None):
    """Write a synthetic complete run, with optional malformed metadata or rows."""
    folder = root / run_id
    folder.mkdir(parents=True, exist_ok=True)
    pairs = [(v, m) for v in CONFIG["variants"] for m in CONFIG["models"]]
    summary = {"execution_id": run_id, "quick": True, "evaluation_protocol": "cv",
               "scope": {"models": list(CONFIG["models"]), "variants": CONFIG["variants"], "search": "first"},
               "n_splits": 3, "candidate_count": 4, "successful_candidates": 4}
    summary.update(overrides or {})
    (folder / "cv_training_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    rows = [{"variant": v, "model": m, "run_id": str(i), "status": "FINISHED", "cv_f1": score}
            for i, ((v, m), score) in enumerate(zip(pairs, scores if scores is not None else [.60, .65, .72, .71]))]
    if rows_override is not None:
        rows = rows_override(rows)
    with (folder / "cv_leaderboard.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "model", "run_id", "status", "cv_f1"])
        writer.writeheader()
        writer.writerows(rows)


def test_current_execution_passes(tmp_path):
    """A complete current CV run is gated on its own highest mean score."""
    write_execution(tmp_path, OLD_ID, scores=[.99]*4)
    write_execution(tmp_path)
    assert gate.validate_execution(tmp_path, {OLD_ID}, CONFIG, .65) == .72


def test_stale_score_cannot_pass(tmp_path):
    """A strong historical score cannot rescue a weak new execution."""
    write_execution(tmp_path, OLD_ID, scores=[.99]*4)
    write_execution(tmp_path, scores=[.64]*4)
    with pytest.raises(RuntimeError, match="below"):
        gate.validate_execution(tmp_path, {OLD_ID}, CONFIG, .65)


def test_missing_or_multiple_runs(tmp_path):
    """Reject missing or ambiguous execution identity."""
    with pytest.raises(RuntimeError, match="exactly one"):
        gate.validate_execution(tmp_path, set(), CONFIG, .65)
    write_execution(tmp_path)
    write_execution(tmp_path, OLD_ID)
    with pytest.raises(RuntimeError, match="exactly one"):
        gate.validate_execution(tmp_path, set(), CONFIG, .65)


@pytest.mark.parametrize("overrides", [{"quick": False}, {"evaluation_protocol": "holdout"},
    {"scope": {"search": "grid"}}, {"successful_candidates": 3}, {"candidate_count": 3},
    {"n_splits": 2}, {"execution_id": OLD_ID}])
def test_reject_incompatible_execution(tmp_path, overrides):
    """Reject wrong training protocol, scope or incomplete summary."""
    write_execution(tmp_path, overrides=overrides)
    with pytest.raises(RuntimeError, match="incompatible"):
        gate.validate_execution(tmp_path, set(), CONFIG, .65)


@pytest.mark.parametrize("score", ["NaN", "inf", -0.1, 1.1, "bad", ""])
def test_invalid_scores(tmp_path, score):
    """Reject corrupt or nonfinite metrics even if another score is good."""
    write_execution(tmp_path, scores=[score, .65, .72, .71])
    with pytest.raises(RuntimeError, match="CV F1"):
        gate.validate_execution(tmp_path, set(), CONFIG, .65)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "failed", "duplicate_run", "unknown"])
def test_incomplete_leaderboard(tmp_path, mutation):
    """Require one successfully tracked candidate for every configured pair."""
    def alter(rows):
        """Simulate one bad leaderboard condition."""
        if mutation == "missing":
            return rows[:-1]
        if mutation == "duplicate":
            return rows + [rows[0]]
        if mutation == "failed":
            rows[0]["status"] = "FAILED"
        if mutation == "duplicate_run":
            rows[0]["run_id"] = rows[1]["run_id"]
        if mutation == "unknown":
            rows[0]["model"] = "unknown"
        return rows
    write_execution(tmp_path, rows_override=alter)
    with pytest.raises(RuntimeError, match="candidate|scope"):
        gate.validate_execution(tmp_path, set(), CONFIG, .65)


def test_run_check_uses_cv_only(tmp_path, monkeypatch):
    """The CI runner uses the explicit first-configuration CV training command."""
    monkeypatch.setattr(gate, "training_runs_directory", lambda: tmp_path)
    monkeypatch.setattr(gate, "load_resolved_config", lambda path: CONFIG)
    def fake_run(command, **kwargs):
        """Check the command and simulate successful output."""
        assert command[1:] == ["-m", "src.models.train", "--mode", "validation", "--evaluation", "cv", "--search", "first"]
        assert kwargs["check"] is True
        write_execution(tmp_path)
    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    assert gate.run_check(.65) == .72


def test_training_failure_stops_gate(tmp_path, monkeypatch):
    """A failed subprocess cannot be ignored even with historical output."""
    monkeypatch.setattr(gate, "training_runs_directory", lambda: tmp_path)
    monkeypatch.setattr(gate, "load_resolved_config", lambda path: CONFIG)
    def fail(command, **kwargs):
        """Simulate a failed training process."""
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(gate.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        gate.run_check(.65)


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -.1, 1.1])
def test_invalid_threshold(tmp_path, threshold):
    """Invalid thresholds cannot bypass the gate."""
    with pytest.raises(ValueError, match="Minimum F1"):
        gate.validate_execution(tmp_path, set(), CONFIG, threshold)
