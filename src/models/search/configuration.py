"""Validate Optuna settings without importing Optuna or constructing estimators."""

from math import isfinite


# These parameters belong to the shared engine or the study, not estimators.
_RESERVED_PARAMETERS = {"variant", "random_seed", "random_state"}


def _mapping(value, label: str) -> dict:
    """Require a mapping with nonempty string keys."""
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not key.strip() for key in value
    ):
        raise ValueError(f"{label} must be a mapping with nonempty string keys.")
    return value


def _keys(value: dict, required: set, optional: set, label: str) -> None:
    """Reject missing or unknown schema fields."""
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing or unknown:
        raise ValueError(f"{label}: missing keys {sorted(missing)}; unknown keys {sorted(unknown)}.")


def _number(value) -> bool:
    """Identify finite real numbers without accepting booleans."""
    return type(value) is int or (type(value) is float and isfinite(value))


def _integer(value, minimum: int, label: str) -> None:
    """Require an integer at or above the given minimum."""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}.")


def _same_choice(left, right) -> bool:
    """Match categorical choices without conflating booleans and numbers."""
    return type(left) is type(right) and left == right


def validate_distribution(spec, label: str) -> None:
    """Validate a bounded integer, float, or categorical distribution."""
    spec = _mapping(spec, label)
    kind = spec.get("type")
    if kind == "categorical":
        _keys(spec, {"type", "choices"}, set(), label)
        choices = spec["choices"]
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"{label}.choices must be a nonempty list.")
        for index, choice in enumerate(choices):
            if not (choice is None or type(choice) in (str, bool) or _number(choice)):
                raise ValueError(f"{label}.choices must contain finite scalar values.")
            # Optuna treats equal values as ambiguous, even across numeric types.
            if any(choice == other for other in choices[:index]):
                raise ValueError(f"{label}.choices must not contain duplicates.")
        return
    if kind not in ("int", "float"):
        raise ValueError(f"{label}.type must be int, float, or categorical.")
    _keys(spec, {"type", "low", "high"}, {"log"}, label)
    low, high = spec["low"], spec["high"]
    valid_bounds = (type(low) is int and type(high) is int) if kind == "int" else (_number(low) and _number(high))
    if not valid_bounds or low >= high:
        raise ValueError(f"{label}: bounds must be finite, correctly typed, and low < high.")
    log = spec.get("log", False)
    if type(log) is not bool or (log and low <= 0):
        raise ValueError(f"{label}.log requires a boolean and positive bounds when enabled.")


def _validate_seed_value(value, distribution: dict, label: str) -> None:
    """Require each queued trial value to lie within its declared distribution."""
    kind = distribution["type"]
    if kind == "categorical":
        if not any(_same_choice(value, choice) for choice in distribution["choices"]):
            raise ValueError(f"{label} is not a declared categorical choice.")
    elif (not _number(value) or (kind == "int" and type(value) is not int)
          or not distribution["low"] <= value <= distribution["high"]):
        raise ValueError(f"{label} is outside its distribution or has the wrong type.")


def validate_optuna_config(config: dict) -> None:
    """Validate optional study settings and per-family spaces without mutation.

    Old YAML files can omit Optuna altogether. Search-specific sampled values
    override matching fixed values only during future Optuna execution; the
    legacy fixed/grid mappings are never changed by this validator.
    """
    models = config["models"]
    if "optuna" not in config:
        if any("optuna" in model for model in models.values()):
            raise ValueError("optuna global settings are required for per-model Optuna spaces.")
        return

    settings = _mapping(config["optuna"], "optuna")
    _keys(settings, {"sampler", "n_startup_trials", "storage_path"}, set(), "optuna")
    if settings["sampler"] != "tpe":
        raise ValueError("optuna.sampler must be tpe.")
    _integer(settings["n_startup_trials"], 0, "optuna.n_startup_trials")
    storage = settings["storage_path"]
    if not isinstance(storage, str) or not storage.strip() or not storage.endswith(".db"):
        raise ValueError("optuna.storage_path must be a nonempty SQLite .db path.")

    for family, model in models.items():
        if "optuna" not in model:
            continue  # No default Optuna budget for this family.
        label = f"models.{family}.optuna"
        section = _mapping(model["optuna"], label)
        _keys(section, {"trials", "space"}, {"enqueue"}, label)
        _integer(section["trials"], 1, f"{label}.trials")
        space = _mapping(section["space"], f"{label}.space")
        if not space or _RESERVED_PARAMETERS & space.keys():
            raise ValueError(f"{label}.space must be nonempty and cannot sample reserved parameters.")
        for parameter, distribution in space.items():
            validate_distribution(distribution, f"{label}.space.{parameter}")

        # Complete queued configurations avoid silently randomizing unspecified values.
        queued = section.get("enqueue", [])
        if not isinstance(queued, list) or len(queued) > section["trials"]:
            raise ValueError(f"{label}.enqueue must be a list within the default trial budget.")
        seen = []
        for index, entry in enumerate(queued):
            entry_label = f"{label}.enqueue[{index}]"
            entry = _mapping(entry, entry_label)
            _keys(entry, {"variant", "params"}, set(), entry_label)
            # A valid warm start may be outside a narrowed YAML/CLI variant scope;
            # the future search runner must skip it rather than expand that scope.
            if entry["variant"] not in ("baseline", "robust", "interactions"):
                raise ValueError(f"{entry_label}.variant is unsupported.")
            params = _mapping(entry["params"], f"{entry_label}.params")
            _keys(params, set(space), set(), f"{entry_label}.params")
            for parameter, value in params.items():
                _validate_seed_value(value, space[parameter], f"{entry_label}.params.{parameter}")
            if entry in seen:
                raise ValueError(f"{entry_label} duplicates a queued configuration.")
            seen.append(entry)
