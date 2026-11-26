"""Exercise the adaptive search lifecycle using real Optuna and synthetic F1."""

import copy
from pathlib import Path

import optuna
from optuna.trial import TrialState
import pytest
import yaml

from src.models.search.optuna_search import CandidateEvaluationError, run_optuna_search


@pytest.fixture
def config():
    """Load the actual YAML search spaces, leaving the repository unchanged."""
    return yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/config.yml").read_text(encoding="utf-8"))


def make_study(config, **kwargs):
    """Create a seeded, maximizing study without accessing any model data."""
    return optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config["seed"], n_startup_trials=config["optuna"]["n_startup_trials"]),
        **kwargs,
    )


def constant_score(candidate):
    """Return a synthetic score for lifecycle tests."""
    return 0.72


@pytest.mark.parametrize("family", ["lightgbm", "xgboost", "catboost", "hist_gradient_boosting"])
def test_warm_start_and_effective_parameters(config, family):
    """Use each family's exact YAML warm start and merge fixed overrides."""
    before = copy.deepcopy(config)
    study = make_study(config)
    seen = []

    def evaluate(candidate):
        """Capture the complete sampled configuration without fitting a model."""
        seen.append(candidate)
        return 0.7

    result = run_optuna_search(study, config, family, config["variants"], evaluate, context_id="synthetic-v1", trials=1)
    model = config["models"][family]
    expected = model["optuna"]["enqueue"][0]
    assert seen[0].family == family
    assert seen[0].study_name == study.study_name
    assert seen[0].trial_number == 0
    assert seen[0].variant == expected["variant"]
    assert seen[0].sampled_params == expected["params"]
    assert seen[0].effective_params == {**model["fixed"], **expected["params"]}
    assert result.completed == result.attempted_this_run == 1
    assert result.best_value == 0.7 and result.best_trial_number == 0
    assert config == before


def test_excluded_warm_start_does_not_expand_scope(config):
    """Do not queue the interactions warm start when only baseline is selected."""
    study = make_study(config)
    result = run_optuna_search(study, config, "lightgbm", ["baseline"], constant_score, context_id="synthetic", trials=3)
    assert result.completed == 3
    assert all(trial.params["variant"] == "baseline" for trial in study.trials)
    assert all("fixed_params" not in trial.system_attrs for trial in study.trials)


def test_seeded_adaptive_search_reproducible(config):
    """Identical fresh studies and deterministic scores produce the same sequence."""
    studies = [make_study(config), make_study(config)]

    def evaluate(candidate):
        """Vary synthetic score with leaf count to provide feedback to TPE."""
        return 0.8 - abs(candidate.sampled_params["num_leaves"] - 25) / 200

    for study in studies:
        run_optuna_search(study, config, "lightgbm", config["variants"], evaluate, context_id="synthetic", trials=9)
    assert [trial.params for trial in studies[0].trials] == [trial.params for trial in studies[1].trials]
    assert [trial.value for trial in studies[0].trials] == [trial.value for trial in studies[1].trials]
    assert len({trial.params["num_leaves"] for trial in studies[0].trials}) > 1


def test_resume_total_budget_and_queued_starts(config, tmp_path):
    """Resume a persisted study without exceeding total attempts or requeuing starts."""
    storage = f"sqlite:///{(tmp_path / 'study.db').as_posix()}"
    study = make_study(config, storage=storage, study_name="hgb")
    family = "hist_gradient_boosting"
    first = run_optuna_search(study, config, family, config["variants"], constant_score, context_id="synthetic", trials=1)
    assert first.completed == 1 and first.waiting == 1
    study = optuna.load_study(study_name="hgb", storage=storage, sampler=optuna.samplers.TPESampler(seed=config["seed"], n_startup_trials=5))
    same = run_optuna_search(study, config, family, config["variants"], constant_score, context_id="synthetic", trials=1)
    assert same.attempted_this_run == 0 and same.waiting == 1
    expanded = run_optuna_search(study, config, family, config["variants"], constant_score, context_id="synthetic", trials=4)
    assert expanded.completed == 4 and expanded.attempted_this_run == 3 and expanded.waiting == 0
    assert len(study.trials) == 4
    second = config["models"][family]["optuna"]["enqueue"][1]
    assert study.trials[1].params == {"variant": second["variant"], **second["params"]}


@pytest.mark.parametrize("change", ["context", "variant", "family", "space", "fixed", "seed", "startup"])
def test_resume_rejects_incompatible_contract(config, change):
    """Refuse to mix incompatible search spaces or evaluation identities."""
    study = make_study(config)
    family, variants, context = "lightgbm", config["variants"], "v1"
    run_optuna_search(study, config, family, variants, constant_score, context_id=context, trials=1)
    if change == "context":
        context = "v2"
    elif change == "variant":
        variants = ["baseline"]
    elif change == "family":
        family = "xgboost"
    elif change == "space":
        config["models"][family]["optuna"]["space"]["num_leaves"]["high"] += 1
    elif change == "fixed":
        config["models"][family]["fixed"]["n_jobs"] = 2
    elif change == "seed":
        config["seed"] += 1
    elif change == "startup":
        config["optuna"]["n_startup_trials"] += 1
    with pytest.raises(ValueError, match="search contract differs"):
        run_optuna_search(study, config, family, variants, constant_score, context_id=context, trials=2)
    assert len(study.trials) == 1


def test_explicit_failure_consumes_budget(config):
    """Continue only on explicit recoverable errors and count failed attempts."""
    study = make_study(config)

    def evaluate(candidate):
        """Fail the first candidate and complete the next two."""
        if candidate.trial_number == 0:
            raise CandidateEvaluationError("synthetic fit failure")
        return 0.73

    result = run_optuna_search(study, config, "lightgbm", config["variants"], evaluate, context_id="v1", trials=3)
    assert result.failed == 1 and result.completed == 2 and result.attempted_this_run == 3
    assert study.trials[0].state == TrialState.FAIL
    assert result.best_value == 0.73


def test_all_failed_has_no_best(config):
    """Avoid asking Optuna for a winning configuration when every trial failed."""
    study = make_study(config)

    def evaluate(candidate):
        """Simulate recoverable failures for every candidate."""
        raise CandidateEvaluationError("synthetic failure")

    result = run_optuna_search(study, config, "lightgbm", config["variants"], evaluate, context_id="v1", trials=2)
    assert result.failed == 2 and result.completed == 0
    assert result.best_value is None and result.best_trial_number is None


def test_unexpected_exception_stops_search(config):
    """Do not suppress infrastructure failures as ordinary failed candidates."""
    study = make_study(config)

    def evaluate(candidate):
        """Simulate an unexpected failure in the caller's infrastructure."""
        raise OSError("tracking unavailable")

    with pytest.raises(OSError, match="tracking unavailable"):
        run_optuna_search(study, config, "lightgbm", config["variants"], evaluate, context_id="v1", trials=4)
    assert len(study.trials) == 1 and study.trials[0].state == TrialState.FAIL


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.01, 1.01, True, "0.7"])
def test_invalid_objective_stops_search(config, value):
    """Reject invalid F1 values instead of silently continuing optimization."""
    study = make_study(config)

    def evaluate(candidate):
        """Return an intentionally invalid score."""
        return value

    with pytest.raises(ValueError, match="finite F1"):
        run_optuna_search(study, config, "lightgbm", config["variants"], evaluate, context_id="v1", trials=3)
    assert study.trials[0].state == TrialState.FAIL


@pytest.mark.parametrize("trials", [0, -1, True, 1.5])
def test_invalid_trial_budget(config, trials):
    """Reject invalid CLI budget overrides before creating trial records."""
    study = make_study(config)
    with pytest.raises(ValueError, match="positive integer"):
        run_optuna_search(study, config, "lightgbm", config["variants"], constant_score, context_id="v1", trials=trials)
    assert not study.trials


@pytest.mark.parametrize("variants", [[], ["baseline", "baseline"], ["invalid"]])
def test_invalid_variants(config, variants):
    """Reject invalid, empty or duplicated variant selections."""
    study = make_study(config)
    with pytest.raises(ValueError, match="variants"):
        run_optuna_search(study, config, "lightgbm", variants, constant_score, context_id="v1", trials=1)


def test_missing_family_space(config):
    """Do not invent an Optuna search space for an unconfigured algorithm."""
    with pytest.raises(ValueError, match="No Optuna search space"):
        run_optuna_search(make_study(config), config, "logistic_regression", config["variants"], constant_score, context_id="v1", trials=1)


def test_default_budget(config):
    """Use the YAML budget when no explicit override is given."""
    config["models"]["lightgbm"]["optuna"]["trials"] = 2
    result = run_optuna_search(make_study(config), config, "lightgbm", config["variants"], constant_score, context_id="v1")
    assert result.budget == result.completed == 2


def test_running_trial_blocks_resume(config):
    """Fail closed if an earlier process left a study trial running."""
    study = make_study(config)
    run_optuna_search(study, config, "lightgbm", config["variants"], constant_score, context_id="v1", trials=1)
    study.ask()
    with pytest.raises(RuntimeError, match="RUNNING trial"):
        run_optuna_search(study, config, "lightgbm", config["variants"], constant_score, context_id="v1", trials=3)


def test_reject_unbound_populated_study(config):
    """Do not adopt historical trials whose evaluation contract is unknown."""
    study = make_study(config)
    trial = study.ask()
    study.tell(trial, 0.7)
    with pytest.raises(ValueError, match="without a search contract"):
        run_optuna_search(study, config, "lightgbm", config["variants"], constant_score, context_id="v1", trials=3)


def test_reject_minimization_study(config):
    """Require the objective direction to maximize F1."""
    study = optuna.create_study(direction="minimize")
    with pytest.raises(ValueError, match="maximizing objective"):
        run_optuna_search(study, config, "lightgbm", config["variants"], constant_score, context_id="v1", trials=1)


def test_pruned_trial_counts_towards_budget(config):
    """Account for pruned trials even though pruning is not enabled by our runner."""
    study = make_study(config)

    def evaluate(candidate):
        """Simulate a caller-requested prune in the first trial."""
        if candidate.trial_number == 0:
            raise optuna.TrialPruned()
        return 0.7

    result = run_optuna_search(study, config, "lightgbm", config["variants"], evaluate, context_id="v1", trials=2)
    assert result.pruned == 1 and result.completed == 1 and result.attempted_this_run == 2
