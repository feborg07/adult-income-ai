"""Check modular loading, overrides and independent configuration identities."""
from copy import deepcopy
import json

import pytest
import yaml

from src.models.configuration import (configuration_fingerprints, load_resolved_config,
                                      write_config_snapshot, read_mapping)


@pytest.fixture
def config_tree(tmp_path):
    """Create an isolated two-cartridge configuration with known search settings."""
    models = {'alpha': {'fixed': {}, 'grid': {'depth': [2, 3]}, 'optuna': {
        'trials': 10, 'space': {'depth': {'type': 'int', 'low': 2, 'high': 5}}, 'enqueue': []}},
        'beta': {'fixed': {}, 'grid': {'C': [1]}, 'optuna': {'trials': 10, 'space': {}, 'enqueue': []}}}
    folder = tmp_path / 'algorithms'
    folder.mkdir()
    global_settings = {'seed': 42, 'variants': ['baseline'], 'cv_representation': 'no_fnlwgt'}
    main = tmp_path / 'config.yml'
    main.write_text(yaml.safe_dump({**global_settings, 'algorithms_dir': 'algorithms'}), encoding='utf-8')
    for order, (name, spec) in enumerate(models.items()):
        (folder / f'{name}.yml').write_text(yaml.safe_dump({'order': order, **spec}, sort_keys=False), encoding='utf-8')
    return main, {**global_settings, 'models': models}


def test_resolved_configuration_matches_inline(config_tree):
    """The modular format preserves all effective model settings."""
    main, expected = config_tree
    assert load_resolved_config(main) == expected
    assert list(load_resolved_config(main)['models']) == list(expected['models'])


def test_override_isolated(config_tree, tmp_path):
    """A search change affects only the selected algorithm identity."""
    main, expected = config_tree
    override = tmp_path / 'override.yml'
    override.write_text('models:\n  alpha:\n    optuna:\n      space:\n        depth: {type: int, low: 3, high: 7}\n', encoding='utf-8')
    changed = load_resolved_config(main, experiment_path=override)
    original = configuration_fingerprints(expected)
    updated = configuration_fingerprints(changed)
    assert updated['algorithm_sha256']['alpha'] != original['algorithm_sha256']['alpha']
    assert updated['study_configuration_sha256']['beta'] == original['study_configuration_sha256']['beta']
    assert updated['global_sha256'] == original['global_sha256']
    assert load_resolved_config(main) == expected


def test_budget_does_not_change_study_configuration(config_tree):
    """The cumulative trial budget is logged but does not change search identity."""
    _, original = config_tree
    changed = deepcopy(original)
    changed['models']['alpha']['optuna']['trials'] = 20
    before, after = map(configuration_fingerprints, (original, changed))
    assert before['complete_sha256'] != after['complete_sha256']
    assert before['study_configuration_sha256'] == after['study_configuration_sha256']


def test_shared_seed_changes_all_study_configuration_ids(config_tree):
    """A changed training seed invalidates each study configuration."""
    _, original = config_tree
    changed = deepcopy(original)
    changed['seed'] = 99
    before, after = map(configuration_fingerprints, (original, changed))
    assert all(before['study_configuration_sha256'][name] != after['study_configuration_sha256'][name]
               for name in original['models'])


def test_comments_do_not_change_identity(config_tree):
    """Formatting-only edits do not alter semantic fingerprints."""
    main, _ = config_tree
    before = configuration_fingerprints(load_resolved_config(main))
    main.write_text('# comment\n' + main.read_text(), encoding='utf-8')
    assert configuration_fingerprints(load_resolved_config(main)) == before


def test_unknown_override_fails(config_tree, tmp_path):
    """Reject typographical errors instead of silently inventing settings."""
    main, _ = config_tree
    override = tmp_path / 'override.yml'
    override.write_text('models: {unknown: {fixed: {}}}', encoding='utf-8')
    with pytest.raises(ValueError, match='Unknown experiment setting'):
        load_resolved_config(main, experiment_path=override)


def test_duplicate_yaml_key_fails(tmp_path):
    """Reject duplicate YAML keys before loading an experiment."""
    path = tmp_path / 'duplicate.yml'
    path.write_text('seed: 1\nseed: 2\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate YAML key'):
        read_mapping(path)


def test_snapshot(config_tree, tmp_path):
    """Keep a complete reproducible configuration and scoped run identity."""
    main, expected = config_tree
    output = tmp_path / 'run'
    output.mkdir()
    write_config_snapshot(output, load_resolved_config(main), {'models': ['alpha']})
    assert read_mapping(output / 'resolved_config.yml') == expected
    assert json.loads((output / 'configuration_fingerprints.json').read_text())['scope']['models'] == ['alpha']
