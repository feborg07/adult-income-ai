"""Exercise the descriptive fairness audit with synthetic data only."""
import hashlib
import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from scripts import evaluate_fairness as audit
from src.models.tasks import binary_classification as binary


def test_confusion_counts_and_rates():
    """Verify the rates using a hand-calculated confusion matrix."""
    row = audit.subgroup_metrics([1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0])
    assert (row['tp'], row['tn'], row['fp'], row['fn']) == (2, 3, 1, 1)
    assert row['precision'] == pytest.approx(2/3)
    assert row['recall'] == pytest.approx(2/3)
    assert row['false_positive_rate'] == .25
    assert row['false_negative_rate'] == pytest.approx(1/3)
    assert row['f1'] == pytest.approx(2/3)
    assert row['selection_rate'] == pytest.approx(3/7)
    assert row['accuracy'] == pytest.approx(5/7)
    assert row['low_support'] is True


@pytest.mark.parametrize('y,p', [([0, 0], [0, 0]), ([1, 1], [1, 1])])
def test_undefined_rates(y, p):
    """Distinguish a measured zero from an undefined rate."""
    row = audit.subgroup_metrics(y, p)
    if y[0] == 0:
        assert row['recall'] is None and row['precision'] is None and row['f1'] is None
        assert row['false_positive_rate'] == 0
    else:
        assert row['false_positive_rate'] is None and row['recall'] == 1


@pytest.mark.parametrize('y,p', [([], []), ([0], [0, 1]), ([2], [1]), ([1], [np.nan])])
def test_invalid_inputs(y, p):
    """Reject empty, misaligned or nonbinary arrays."""
    with pytest.raises(ValueError):
        audit.subgroup_metrics(y, p)


def test_support_flag():
    """Flag sparse denominators as well as small groups."""
    assert audit.subgroup_metrics([0]*90 + [1]*10, [1]*100)['low_support']
    assert not audit.subgroup_metrics([0]*40 + [1]*40, [0]*40 + [1]*40)['low_support']


def test_group_alignment_and_missing():
    """Preserve positional alignment and missing demographic records."""
    X = pd.DataFrame({'sex': ['F', 'M', 'F', None], 'race': ['A', 'A', 'B', 'B']}, index=[7, 2, 9, 1])
    rows = audit.group_metrics(X, [1, 0, 0, 1], [1, 1, 0, 0], ('sex',))
    groups = {r['group']: r for r in rows}
    assert sum(r['rows'] for r in rows) == 4
    assert groups['F']['tp'] == groups['F']['tn'] == 1
    assert groups['M']['fp'] == 1
    assert groups['<missing>']['fn'] == 1
    assert len(audit.group_metrics(X, [1, 0, 0, 1], [1, 1, 0, 0], ('sex', 'race'))) == 4


def test_missing_demographic_column():
    """Fail rather than silently skip a requested attribute."""
    with pytest.raises(ValueError, match='Missing demographic'):
        audit.group_metrics(pd.DataFrame({'sex': ['F']}), [1], [1], ('race',))


def test_disparities():
    """Check demographic parity, equal opportunity and equalized odds formulas."""
    a = audit.subgroup_metrics([1, 1, 0, 0], [1, 1, 1, 0])
    b = audit.subgroup_metrics([1, 1, 0, 0], [1, 0, 0, 0])
    result = audit.disparity_summary([a, b])
    assert result['demographic_parity_difference'] == .5
    assert result['demographic_parity_ratio'] == pytest.approx(1/3)
    assert result['equal_opportunity_difference'] == .5
    assert result['equalized_odds_difference'] == .5
    assert result['false_negative_rate_gap'] == .5


def test_undefined_disparities():
    """Expose undefined gaps and denominator coverage."""
    a = audit.subgroup_metrics([0, 0], [0, 0])
    b = audit.subgroup_metrics([1, 0], [1, 0])
    result = audit.disparity_summary([a, b])
    assert result['recall_defined_groups'] == 1
    assert result['equal_opportunity_difference'] is None
    assert result['equalized_odds_difference'] is None
    assert audit.disparity_summary([a, a])['demographic_parity_ratio'] is None
    assert audit.disparity_summary([b])['demographic_parity_difference'] is None


@pytest.fixture
def synthetic_audit():
    """Supply demographic records, probabilities and expected aggregate metrics."""
    X = pd.DataFrame({'sex': ['F', 'F', 'M', 'M', 'F', 'M'], 'race': ['A', 'B', 'A', 'B', 'A', 'B']})
    y = pd.Series([0, 1, 0, 1, 1, 0])
    p = np.array([.1, .35, .4, .9, .2, .3])
    unseen = np.array([True, True, False, True, True, False])
    expected = {'candidate': 'xgboost'}
    for name, mask in [('test', np.ones(len(X), dtype=bool)), ('unseen_test', unseen)]:
        expected[f'{name}_rows'] = int(mask.sum())
        expected.update({f'{name}_{k}': v for k, v in binary.evaluate(y[mask], p[mask], .35).items()})
    return X, y, p, unseen, expected


def test_audit_frozen_threshold(synthetic_audit):
    """Reconcile subgroup confusion matrices and include threshold equality."""
    X, y, p, unseen, expected = synthetic_audit
    rows, gaps = audit.audit_predictions(X, y, p, .35, unseen, expected)
    assert len(gaps) == 6
    for prefix in ('test', 'unseen_test'):
        for grouping in ('sex', 'race', 'sex+race'):
            selected = [r for r in rows if r['partition'] == prefix and r['attribute'] == grouping]
            assert sum(r['rows'] for r in selected) == expected[f'{prefix}_rows']
            assert [[sum(r[k] for r in selected) for k in ('tn', 'fp')],
                    [sum(r[k] for r in selected) for k in ('fn', 'tp')]] == expected[f'{prefix}_confusion_matrix']


def test_reject_metric_mismatch(synthetic_audit):
    """Reject a change in previously recorded held-out results."""
    X, y, p, unseen, expected = synthetic_audit
    expected['test_f1'] = .99
    with pytest.raises(ValueError, match='Held-out metric mismatch'):
        audit.audit_predictions(X, y, p, .35, unseen, expected)


@pytest.fixture
def synthetic_report(synthetic_audit, monkeypatch):
    """Create a matching held-out report and frozen model identities."""
    X, y, p, unseen, expected = synthetic_audit
    e = audit.heldout
    monkeypatch.setattr(e, 'EXPECTED_ROWS', (10, 5, len(X)))
    monkeypatch.setattr(e, 'EXPECTED_UNSEEN', int(unseen.sum()))
    specs = {name: {'model_sha256': name, 'threshold': .35, 'representation': 'no_fnlwgt',
                    'run_id': name, 'family': name, 'variant': 'baseline'} for name in ('xgboost', 'reference')}
    rows = [dict(expected, candidate=name, **spec) for name, spec in specs.items()]
    identity = {'synthetic': True}
    context = e.EvaluationContext('a'*32, 'xgboost', {'reference': 'hash'})
    report = {'execution_id': 'synthetic-heldout', 'source_execution_id': context.run_id,
              'evaluation_protocol': 'frozen_heldout', 'selected_candidate': 'xgboost', 'partition_identity': identity,
              'frozen_metadata_content_sha256': context.anchors, 'preflight_approval_sha256': context.approval_sha256, 'test_accessed': True,
              'historical_test_previously_used': True, 'promotion_authorized': False,
              'test_rows': len(X), 'unseen_test_rows': int(unseen.sum()),
              'overlap_excluded_features': list(e.OVERLAP_EXCLUSIONS), 'candidates': rows,
              'performance_gate': e.performance_gate(rows, context.selected), 'raw_source_sha256': {'adult.data': 'a', 'adult.test': 'b'},
              'saved_test_sha256': {'baseline': {}}, 'unseen_mask_sha256': hashlib.sha256(unseen.astype('uint8').tobytes()).hexdigest()}
    return report, identity, specs, context


@pytest.mark.parametrize('mutation', ['identity', 'threshold', 'gate', 'duplicate', 'approval'])
def test_report_binding(synthetic_report, mutation):
    """Reject altered partition identity, threshold, gate or shortlist."""
    report, identity, specs, context = synthetic_report
    assert set(audit.verify_report(report, identity, specs, context)) == set(specs)
    if mutation == 'identity':
        report['partition_identity'] = {}
    elif mutation == 'threshold':
        report['candidates'][0]['threshold'] = .5
    elif mutation == 'gate':
        report['performance_gate']['passed'] = False
    elif mutation == 'approval':
        report['preflight_approval_sha256'] = 'wrong'
    else:
        report['candidates'].append(report['candidates'][0])
    with pytest.raises(ValueError):
        audit.verify_report(report, identity, specs, context)


def test_run_audit_writes_aggregate_reports(tmp_path, monkeypatch, synthetic_audit, synthetic_report):
    """Verify report generation and mocked tracking with no model fitting."""
    X, y, p, unseen, _ = synthetic_audit
    report, identity, specs, context = synthetic_report
    path = tmp_path / 'heldout_summary.json'
    path.write_text(json.dumps(report))
    models = {name: MagicMock() for name in specs}
    for spec in specs.values():
        spec['path'] = tmp_path / 'synthetic.joblib'
    e = audit.heldout
    monkeypatch.setattr(e, 'preflight', lambda root, run_id: (specs, models, identity, {}, (X, y, X, y, None), context))
    monkeypatch.setattr(e, 'verify_file', MagicMock())
    monkeypatch.setattr(e, 'reconstruct_partitions', lambda *args: (X, y))
    monkeypatch.setattr(e, 'verify_test_alignment', lambda *args: report['saved_test_sha256'])
    monkeypatch.setattr(e, 'unseen_mask', lambda *args: unseen)
    monkeypatch.setattr(audit.binary, 'positive_probabilities', lambda *args: p)
    monkeypatch.setattr(e.training, 'git_revision', lambda: 'synthetic')
    monkeypatch.setattr(e.training, 'load_training_config', lambda *args: {'paths': {'mlflow_db': 'db', 'mlflow_artifacts': 'artifacts'}, 'mlflow': {'experiment': 'test'}})
    monkeypatch.setattr(e.training, 'configure_mlflow', MagicMock())
    monkeypatch.setattr(e.training, 'mlflow', MagicMock())
    output = audit.run_audit(tmp_path, path)
    result = json.loads((output / 'fairness_summary.json').read_text())
    assert result['promotion_authorized'] is False
    assert result['fairness_review_required'] is True
    assert len(result['disparities']) == 12
    assert set(p.name for p in output.iterdir()) == {'fairness_summary.json', 'subgroup_metrics.csv', 'disparity_summary.csv'}
    assert len(pd.read_csv(output / 'subgroup_metrics.csv')) == len(result['subgroups'])
    assert e.training.mlflow.log_artifact.call_count == 3
    assert all(not model.fit.called for model in models.values())
