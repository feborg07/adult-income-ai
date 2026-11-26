"""Run adaptive, serial Optuna search through an injected evaluation callback.

The caller owns model fitting, tracking, study storage and evaluation provenance.
No training data, estimator plugins or MLflow objects are accessed here.
"""

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from numbers import Real
from typing import Callable

from optuna.study import Study, StudyDirection
from optuna.trial import Trial, TrialState

from .configuration import validate_optuna_config

_CONTRACT_KEY = "adult_income_search_contract_v1"
_TERMINAL_STATES = (TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED)


class CandidateEvaluationError(RuntimeError):
    """Identify a recoverable candidate failure; other errors stop the search."""


@dataclass(frozen=True)
class OptunaCandidate:
    """Describe one sampled configuration without constructing its estimator."""

    study_name: str
    trial_number: int
    family: str
    variant: str
    sampled_params: dict
    effective_params: dict


@dataclass(frozen=True)
class EvaluationOutcome:
    """Carry an objective and optional JSON-compatible evaluator provenance."""

    score: float
    metadata: dict


@dataclass(frozen=True)
class SearchSummary:
    """Report cumulative trial states and the number attempted this invocation."""

    study_name: str
    budget: int
    attempted_this_run: int
    completed: int
    failed: int
    pruned: int
    waiting: int
    best_trial_number: int | None
    best_value: float | None


def suggest_parameters(trial: Trial, space: dict) -> dict:
    """Translate validated YAML distributions into native Optuna suggestions."""
    sampled = {}
    for name, distribution in space.items():
        kind = distribution["type"]
        kwargs = {key: value for key, value in distribution.items() if key != "type"}
        if kind == "int":
            sampled[name] = trial.suggest_int(name, **kwargs)
        elif kind == "float":
            sampled[name] = trial.suggest_float(name, **kwargs)
        elif kind == "categorical":
            sampled[name] = trial.suggest_categorical(name, **kwargs)
        else:
            raise ValueError(f"Unsupported distribution type for {name}: {kind}")
    return sampled


def _bind_contract(study: Study, contract: dict) -> None:
    """Bind a study to its search scope and caller-supplied evaluation identity."""
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False)
    fingerprint = sha256(encoded.encode("utf-8")).hexdigest()
    stored = study.user_attrs.get(_CONTRACT_KEY)
    if stored is None:
        if study.get_trials(deepcopy=False):
            raise ValueError("Cannot adopt a populated study without a search contract.")
        study.set_user_attr(_CONTRACT_KEY, fingerprint)
    elif stored != fingerprint:
        raise ValueError("Study search contract differs; use a new study.")


def _summary(study: Study, budget: int, attempted_before: int) -> SearchSummary:
    """Count trial states and expose the Optuna F1-only best trial if present."""
    trials = study.get_trials(deepcopy=False)
    completed = sum(trial.state == TrialState.COMPLETE for trial in trials)
    failed = sum(trial.state == TrialState.FAIL for trial in trials)
    pruned = sum(trial.state == TrialState.PRUNED for trial in trials)
    waiting = sum(trial.state == TrialState.WAITING for trial in trials)
    best = study.best_trial if completed else None
    return SearchSummary(
        study_name=study.study_name, budget=budget,
        attempted_this_run=completed + failed + pruned - attempted_before,
        completed=completed, failed=failed, pruned=pruned, waiting=waiting,
        best_trial_number=best.number if best is not None else None,
        best_value=float(best.value) if best is not None else None,
    )


def run_optuna_search(
    study: Study,
    config: dict,
    family: str,
    variants: list[str],
    evaluate: Callable[[OptunaCandidate], float | EvaluationOutcome],
    *,
    context_id: str,
    trials: int | None = None,
) -> SearchSummary:
    """Optimize one family with a cumulative, attempted-trial budget.

    Warm starts are scoped to selected variants and count when executed. Failed
    and pruned trials also consume budget; waiting trials do not. A resumed study
    must have the same search contract and evaluation context. Only one process
    may write to a study at a time. The caller must encode data, CV, preprocessing
    and relevant code identity in context_id before enabling persistent resume.
    Expected candidate failures may continue; unexpected errors propagate.
    """
    # Work from a private snapshot so callbacks cannot change the search space.
    config = deepcopy(config)
    variants = list(variants)
    validate_optuna_config(config)
    if family not in config["models"] or "optuna" not in config["models"][family]:
        raise ValueError(f"No Optuna search space configured for {family}.")
    if not variants or len(set(variants)) != len(variants) or not set(variants) <= set(config["variants"]):
        raise ValueError("Optuna variants must be nonempty, unique and within configuration scope.")
    if not isinstance(context_id, str) or not context_id.strip():
        raise ValueError("Optuna requires a nonempty evaluation context_id.")
    if list(study.directions) != [StudyDirection.MAXIMIZE]:
        raise ValueError("Optuna search requires a single maximizing objective.")
    model = config["models"][family]
    section = model["optuna"]
    budget = section["trials"] if trials is None else trials
    if type(budget) is not int or budget < 1:
        raise ValueError("Optuna trial budget must be a positive integer.")
    queued = [entry for entry in section.get("enqueue", []) if entry["variant"] in variants]
    contract = {
        "family": family, "variants": variants, "fixed": model["fixed"],
        # Preserve distribution order because sampling order can affect seeded runs.
        "space_order": list(section["space"]), "space": section["space"],
        "enqueue": queued, "seed": config["seed"], "context_id": context_id,
        "sampler": config["optuna"]["sampler"],
        "n_startup_trials": config["optuna"]["n_startup_trials"],
    }
    _bind_contract(study, contract)
    existing = study.get_trials(deepcopy=False)
    if any(trial.state == TrialState.RUNNING for trial in existing):
        raise RuntimeError("Study has a RUNNING trial; resolve it before serial resumption.")
    attempted_before = sum(trial.state in _TERMINAL_STATES for trial in existing)
    remaining = max(0, budget - attempted_before)
    if remaining == 0:
        return _summary(study, budget, attempted_before)

    # Queue only in-scope warm starts; retries/resumption never duplicate them.
    for entry in queued:
        study.enqueue_trial({"variant": entry["variant"], **entry["params"]}, skip_if_exists=True)

    def objective(trial: Trial) -> float:
        """Sample a variant and estimator parameters, then delegate F1 evaluation."""
        variant = trial.suggest_categorical("variant", variants)
        sampled = suggest_parameters(trial, section["space"])
        # Optuna parameters override fixed values only for this candidate.
        candidate = OptunaCandidate(
            study_name=study.study_name, trial_number=trial.number,
            family=family, variant=variant, sampled_params=deepcopy(sampled),
            effective_params=deepcopy({**model["fixed"], **sampled}),
        )
        outcome = evaluate(candidate)
        score = outcome.score if isinstance(outcome, EvaluationOutcome) else outcome
        if isinstance(score, bool) or not isinstance(score, Real) or not isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Optuna evaluation must return a finite F1 score in [0, 1].")
        if isinstance(outcome, EvaluationOutcome):
            # Store tracking provenance before Optuna commits a completed trial.
            if not isinstance(outcome.metadata, dict):
                raise ValueError("Evaluation metadata must be a mapping.")
            metadata = json.loads(json.dumps(outcome.metadata, allow_nan=False))
            trial.set_user_attr("evaluation_metadata", metadata)
        return float(score)

    # Optuna records failed trials; only explicit recoverable failures continue.
    study.optimize(objective, n_trials=remaining, n_jobs=1,
                   catch=(CandidateEvaluationError,))
    return _summary(study, budget, attempted_before)
