"""Regression tests for validation-only downstream orchestration."""

import json

import pytest

from scripts import run_pipeline as pipeline


RUN_ID = "a" * 32


def write_cv_summary(root, **overrides):
    """Create a synthetic CV run summary with optional invalid fields."""
    folder = root / RUN_ID
    folder.mkdir(parents=True, exist_ok=True)
    summary = {
        "execution_id": RUN_ID,
        "quick": True,
        "evaluation_protocol": "cv",
        "scope": {"search": "grid"},
        "successful_candidates": 1,
    }
    summary.update(overrides)
    (folder / "cv_training_summary.json").write_text(json.dumps(summary))


@pytest.mark.parametrize("mode,search", [("quick", "first"), ("full", "grid")])
def test_pipeline_uses_validation_only_cv(tmp_path, monkeypatch, mode, search):
    """Both modes avoid legacy test scoring and preserve reference artifacts."""
    reference = tmp_path / "models/reference"
    reference.mkdir(parents=True)
    for name in ("best_model_pipeline.joblib", "best_model_metadata.json"):
        (reference / name).write_bytes(b"unchanged")
    runs = tmp_path / "models/runs"
    commands = []

    def fake_stage(name, command, allowed_exit_codes=(0,)):
        """Record orchestration and simulate training output without fitting."""
        commands.append(command)
        if "src.models.train" in command and mode == "full":
            write_cv_summary(runs)
        return {"stage": name, "exit_code": 0, "duration_seconds": 0}

    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "training_runs_directory", lambda: runs)
    monkeypatch.setattr(pipeline, "run_stage", fake_stage)
    pipeline.run_pipeline(mode)

    train = next(c for c in commands if "src.models.train" in c)
    assert train[3:] == ["--mode", "validation", "--evaluation", "cv", "--search", search]
    comparison = [c for c in commands if "scripts.compare_experiments" in c]
    if mode == "full":
        assert len(comparison) == 1
        assert comparison[0][3:9] == ["--mode", "validation", "--evaluation", "cv", "--execution-id", RUN_ID]
    else:
        assert not comparison
    assert not any("src.models.promote" in c for c in commands)
    assert any("src.models.robustness" in c for c in commands)
    assert all(p.read_bytes() == b"unchanged" for p in reference.iterdir())


@pytest.mark.parametrize("overrides", [
    {"quick": False}, {"evaluation_protocol": "holdout"},
    {"scope": {"search": "first"}}, {"successful_candidates": 0},
    {"execution_id": "b" * 32}, {"scope": None},
])
def test_reject_incompatible_cv_summary(tmp_path, overrides):
    """A full grid comparison must use a successful current CV grid run."""
    write_cv_summary(tmp_path, **overrides)
    with pytest.raises(RuntimeError, match="CV grid execution"):
        pipeline.new_cv_execution_id(tmp_path, set())


def test_reject_missing_or_ambiguous_execution(tmp_path):
    """Do not accidentally select a historical or concurrent execution."""
    with pytest.raises(RuntimeError, match="exactly one"):
        pipeline.new_cv_execution_id(tmp_path, set())
    write_cv_summary(tmp_path)
    (tmp_path / ("b" * 32)).mkdir()
    with pytest.raises(RuntimeError, match="exactly one"):
        pipeline.new_cv_execution_id(tmp_path, set())
    assert pipeline.new_cv_execution_id(tmp_path, {"b" * 32}) == RUN_ID
