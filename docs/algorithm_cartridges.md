# Algorithm cartridges

`configs/config.yml` contains shared training settings. `configs/algorithms/*.yml`
contains each algorithm's fixed parameters, grid, Optuna space and budget.
`src/models/algorithms/` contains the corresponding plugins. `tests/algorithms/`
contains automatically parametrized contract tests and family-specific tests.
Legacy grouped regression tests remain available to preserve existing coverage.

## Adding an algorithm

1. Add `<family>.py` under `src/models/algorithms/`. Provide one concrete
   `BaseAlgorithm` subclass, with `name` matching the module filename. Implement
   the required build/logging contract and declare its model dependencies via
   `required_packages` where applicable. Discovery imports trusted repository
   Python modules; do not place unreviewed code in this directory.
2. Add `<family>.yml` under `configs/algorithms/`, with `fixed`, `grid` and
   optional `optuna` sections. Optional YAML `order` and plugin
   `registration_order` must agree if specified (default 1000; ties sorted by
   identifier). Existing algorithms retain their original order.
3. Add `tests/algorithms/test_<family>.py` for unique behavior. The shared
   tests automatically discover the new cartridge. Update pinned dependencies
   only if required. No changes to the training loop or registry list are needed.

## Experiment profiles

An optional `--experiment-config configs/experiments/<profile>.yml` merges
known settings into the loaded configuration. Mappings merge recursively,
lists replace, and complete Optuna distributions replace atomically. Unknown
keys are rejected. When changing distributions, ensure existing queued starts
remain valid or override `enqueue` with a compatible list (possibly empty).
The CLI's explicit scope and budget take precedence. `--experiment` remains
reserved for the MLflow experiment name.

```bash
python -m src.models.train --mode validation --evaluation cv --search optuna --models xgboost --experiment-config configs/experiments/xgboost_budget_example.yml
```

Use that example only when you deliberately want a new or expanded XGBoost
study. There is no need to repeat completed boosting experiments for Phase 3.

## Experiment identities

Each run saves `resolved_config.yml` (all resolved defaults and profile
changes) and `configuration_fingerprints.json` (semantic hashes and actual CLI
scope, CV settings, storage paths and budgets). SHA-256 covers semantic settings
and significant candidate ordering, not YAML formatting. Model-specific study
identity also includes the selected variants, data and fold fingerprints,
preprocessing, relevant installed dependency versions, source hashes, tracking
identity and threshold policy. Trial budget changes do not reset a compatible
study. Changes to shared code or evaluation settings can reset multiple studies.
Changes to unrelated cartridges do not reset a compatible study when shared
code, dependency versions and evaluation settings are unchanged.

The new `cartridge_study_v2` identity deliberately does not adopt older study
keys automatically. Existing SQLite studies, MLflow runs and artifacts are
untouched; compare their valid completed results with the new runs instead of
rerunning the previous 350 trials. Legacy inline YAML configs remain supported.

After the remaining three algorithms have been run, use explicit execution IDs
with `scripts.compare_experiments --evaluation cv --execution-ids ...` to exclude
older incompatible pilots. The comparison checks data, fold, representation,
threshold and preprocessing identity rather than requiring identical model
configuration hashes. The independent validation and held-out evaluation lifecycle
is unchanged; test labels must not be used in this optimization phase.

## Verification

```bash
python -m pytest tests/algorithms/ -v
python -m pytest tests/ -q
```

Do not reset MLflow, Optuna, datasets or the reference model. This refactor is
not a model promotion. Complete local regression verification before launching
the remaining 150 trials. Training lifecycle terminology cleanup remains a
separate end-of-Phase-3 task.
