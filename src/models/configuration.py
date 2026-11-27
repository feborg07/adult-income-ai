"""Resolve trusted, repository-local algorithm configurations and experiment overrides."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re

import yaml


class UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _unique_mapping(loader, node):
    """Construct a mapping without silently discarding duplicate keys."""
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def read_mapping(path):
    """Read a nonempty YAML mapping and reject malformed input."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        value = yaml.load(handle, Loader=UniqueKeyLoader)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Expected a nonempty YAML mapping: {path}")
    return value


def merge_override(base, override, prefix=""):
    """Merge known mapping keys; replace lists and complete Optuna distributions."""
    result = deepcopy(base)
    for key, value in override.items():
        location = f"{prefix}.{key}" if prefix else key
        if key not in base:
            raise ValueError(f"Unknown experiment setting: {location}")
        # A distribution is atomic: do not retain stale keys when its type changes.
        if isinstance(base[key], dict) and isinstance(value, dict):
            result[key] = (deepcopy(value) if "type" in base[key] and "type" in value
                           else merge_override(base[key], value, location))
        else:
            result[key] = deepcopy(value)
    return result


def load_resolved_config(path, *, experiment_path=None):
    """Resolve legacy inline or modular YAML to the engine's existing mapping contract."""
    path = Path(path).resolve()
    config = read_mapping(path)
    if "algorithms_dir" in config:
        if "models" in config:
            raise ValueError("Use inline models or algorithms_dir, not both.")
        directory = config.pop("algorithms_dir")
        if not isinstance(directory, str) or not directory.strip():
            raise ValueError("algorithms_dir must be a nonempty relative path.")
        folder = (path.parent / directory).resolve()
        if Path(directory).is_absolute() or not folder.is_relative_to(path.parent) or not folder.is_dir():
            raise ValueError(f"Invalid algorithm configuration directory: {folder}")
        entries = []
        seen = set()
        for item in sorted([*folder.glob("*.yml"), *folder.glob("*.yaml")]):
            if not item.resolve().is_relative_to(folder):
                raise ValueError(f"Algorithm configuration escapes its directory: {item}")
            name = item.stem
            if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in seen:
                raise ValueError(f"Invalid or duplicate algorithm configuration: {item.name}")
            seen.add(name)
            spec = read_mapping(item)
            order = spec.pop("order", 1000)
            if type(order) is not int:
                raise ValueError(f"{name}: order must be an integer.")
            entries.append((order, name, spec))
        if not entries:
            raise ValueError("No algorithm YAML configurations discovered.")
        config["models"] = {name: spec for _, name, spec in sorted(entries)}
    if experiment_path is not None:
        override = read_mapping(experiment_path)
        # Identity and storage paths must not be redirected by an experiment profile.
        allowed = {"seed", "variants", "cv_representation", "cross_validation", "models"}
        if set(override) - allowed:
            raise ValueError(f"Unsupported experiment settings: {sorted(set(override) - allowed)}")
        config = merge_override(config, override)
    return config


def digest(value):
    """Hash JSON-compatible values in a deterministic canonical representation."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def algorithm_payload(spec, *, exclude_budget=False):
    """Preserve significant enumeration order and optionally omit the trial budget."""
    spec = deepcopy(spec)
    optuna = spec.get("optuna", {})
    if exclude_budget:
        optuna.pop("trials", None)
    return {"spec": spec, "grid_order": list(spec.get("grid", {})),
            "space_order": list(optuna.get("space", {}))}


def configuration_fingerprints(config):
    """Separate complete configuration, shared settings and per-algorithm identity."""
    models = config["models"]
    shared = {key: value for key, value in config.items() if key != "models"}
    full = {"shared": shared, "model_order": list(models),
            "models": {name: algorithm_payload(spec) for name, spec in models.items()}}
    study_shared = {key: deepcopy(shared[key]) for key in ("seed", "variants", "selection_metric",
                    "cv_representation", "cross_validation", "optuna") if key in shared}
    # Storage location is not a sampling or evaluation setting.
    study_shared.get("optuna", {}).pop("storage_path", None)
    return {"schema": "modular_configuration_v1", "complete_sha256": digest(full),
            "global_sha256": digest(shared),
            "algorithm_sha256": {name: digest(algorithm_payload(spec)) for name, spec in models.items()},
            "study_configuration_sha256": {name: digest({"shared": study_shared,
                "model": algorithm_payload(spec, exclude_budget=True)}) for name, spec in models.items()}}


def configuration_digest(config):
    """Identify the complete semantic configuration, including all algorithm files."""
    return configuration_fingerprints(config)["complete_sha256"]


def write_config_snapshot(output, config, scope):
    """Persist resolved YAML and independent fingerprints in the isolated run directory."""
    output = Path(output)
    # The full configuration includes all model cartridges, not just CLI-selected models.
    (output / "resolved_config.yml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    payload = {**configuration_fingerprints(config), "scope": scope}
    (output / "configuration_fingerprints.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
