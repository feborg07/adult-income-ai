"""Explicitly install a frozen, reviewed candidate without fitting or test scoring.

Dry run by default. Confirmation requires the SHA-256 of the reviewed evidence.
The former full-training promotion path is intentionally no longer supported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from datetime import datetime, timezone
import uuid

import joblib

ROOT = Path(__file__).resolve().parents[2]
MODEL_FILENAME = 'best_model_pipeline.joblib'
METADATA_FILENAME = 'best_model_metadata.json'
SCENARIOS = {'control', 'mild', 'moderate', 'severe'}
MIN_VALIDATION_F1 = .65
MARGIN = .01


def require(condition, message):
    """Reject an invalid or incomplete promotion request."""
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    """Hash artifact bytes without loading or modifying them."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value):
    """Hash JSON content independently of formatting and line endings."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def load_json(path):
    """Load a JSON object, rejecting non-object records."""
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    require(isinstance(value, dict), f'Expected JSON object: {path}')
    return value


def validate_run_id(value):
    """Accept only safe execution identifiers."""
    require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{32}', value), 'Invalid execution ID')
    return value


def verify_hash(path, expected):
    """Require the exact expected byte hash."""
    require(isinstance(expected, str) and re.fullmatch(r'[0-9a-f]{64}', expected), 'Invalid SHA-256')
    require(sha256_file(path) == expected, f'SHA-256 mismatch: {path}')


def rate(value):
    """Validate a probability, metric or threshold in the closed unit interval."""
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
            'Invalid metric or threshold')
    return value


def evidence_row(report, name):
    """Return exactly one held-out model row."""
    rows = [r for r in report['candidates'] if r['candidate'] == name]
    require(len(rows) == 1, f'Missing or duplicate held-out candidate: {name}')
    return rows[0]


def read_evidence(root, run_id, heldout_id, fairness_id):
    """Read and cross-check the reviewed evidence, without executing evaluation."""
    for value in (run_id, heldout_id, fairness_id):
        validate_run_id(value)
    run = root / 'models/runs' / run_id
    ref = root / 'models/reference'
    heldout_path = root / 'reports/heldout' / heldout_id / 'heldout_summary.json'
    fairness_path = root / 'reports/fairness' / fairness_id / 'fairness_summary.json'
    robust = root / 'reports/robustness/phase4'
    heldout = load_json(heldout_path)
    fairness = load_json(fairness_path)
    summary = load_json(run / 'candidate_validation_summary.json')
    plan = load_json(run / 'frozen_plan.json')
    identity = load_json(run / 'partition_identity.json')
    previous = load_json(ref / METADATA_FILENAME)
    require(heldout['execution_id'] == heldout_id and heldout['source_execution_id'] == run_id
            and summary['execution_id'] == identity['execution_id'] == run_id, 'Execution identity mismatch')
    require(heldout['evaluation_protocol'] == 'frozen_heldout' and heldout['test_accessed'] is True
            and identity['test_accessed'] is False and summary['test_accessed'] is False,
            'Invalid evaluation protocol')
    require(heldout['partition_identity'] == identity and all(summary.get(k) == v for k,v in identity.items()),
            'Frozen partition identity mismatch')
    anchors = heldout['frozen_metadata_content_sha256']
    for name, data in (('candidate_validation_summary.json', summary), ('frozen_plan.json', plan),
                       ('partition_identity.json', identity), (METADATA_FILENAME, previous)):
        require(canonical_sha(data) == anchors[name], f'Frozen metadata changed: {name}')
    require(plan['representation'] == identity['representation'] == 'no_fnlwgt', 'Application representation mismatch')
    require(plan['raw_training_sha256'] == identity['raw_training_sha256'] and
            plan['fold_sha256'] == identity['fold_sha256'], 'Frozen plan identity mismatch')
    # Select solely from development metrics; never rank by test performance.
    rows = summary['candidates']
    require(len(rows) == len(plan['candidates']) and {r['candidate'] for r in rows} == set(plan['candidates']),
            'Frozen shortlist mismatch')
    name = heldout['selected_candidate']
    require(isinstance(name, str) and re.fullmatch(r'[a-zA-Z0-9_-]+', name) and name != 'reference',
            'Invalid candidate identifier')
    winners = [r for r in rows if rate(r['val_f1']) == max(rate(x['val_f1']) for x in rows)]
    require(len(winners) == 1 and winners[0]['candidate'] == name, 'Candidate was not the unique validation winner')
    candidate = winners[0]
    require(candidate == load_json(run / name / 'validation_result.json') and
            all(candidate.get(k) == v for k,v in plan['candidates'][name].items()), 'Frozen candidate mismatch')
    require(rate(candidate['val_f1']) >= MIN_VALIDATION_F1, 'Validation F1 gate failed')
    model_path = run / name / 'model_pipeline.joblib'
    verify_hash(model_path, candidate['model_sha256'])
    verify_hash(ref / MODEL_FILENAME, previous['model_sha256'])
    selected = evidence_row(heldout, name)
    reference = evidence_row(heldout, 'reference')
    for key in ('candidate', 'family', 'variant', 'model_sha256', 'run_id', 'threshold', 'val_f1'):
        require(selected[key] == candidate[key], f'Held-out candidate mismatch: {key}')
    require(reference['model_sha256'] == previous['model_sha256'] and
            reference['threshold'] == previous['decision_threshold'], 'Held-out reference mismatch')
    for row in (selected, reference):
        rate(row['threshold'])
        require(row['test_rows'] == heldout['test_rows'] and row['unseen_test_rows'] == heldout['unseen_test_rows'],
                'Held-out row count mismatch')
    gate = heldout['performance_gate']
    require(gate['passed'] is True and gate['selected_candidate'] == name and gate['margin'] == MARGIN,
            'Recorded performance gate failed')
    for field in ('test_f1', 'unseen_test_f1'):
        actual, baseline = rate(selected[field]), rate(reference[field])
        check = gate['checks'][field]
        require(actual >= baseline - MARGIN and check['passed'] is True and
                check['candidate'] == actual and check['reference'] == baseline and
                math.isclose(check['minimum'], baseline-MARGIN, abs_tol=1e-12), f'Performance gate failed: {field}')
    require(fairness['execution_id'] == fairness_id and fairness['heldout_execution_id'] == heldout_id
            and fairness['source_execution_id'] == run_id, 'Fairness execution mismatch')
    verify_hash(heldout_path, fairness['heldout_summary_sha256'])
    for key in ('raw_source_sha256', 'saved_test_sha256', 'unseen_mask_sha256', 'test_rows', 'unseen_test_rows'):
        require(fairness[key] == heldout[key], f'Fairness partition mismatch: {key}')
    require(fairness['heldout_evaluator_sha256'] == heldout['evaluator_sha256'], 'Fairness evaluator mismatch')
    robust_reports = {}
    for label, row in ((name, selected), ('reference', reference)):
        fair = fairness['models'][label]
        require(all(fair[k] == row[k] for k in ('model_sha256', 'threshold', 'representation')),
                f'Fairness artifact mismatch: {label}')
        for partition in ('test', 'unseen_test'):
            for attribute in ('sex', 'race', 'sex+race'):
                groups = [g for g in fairness['subgroups'] if (g['candidate'],g['partition'],g['attribute']) == (label,partition,attribute)]
                require(groups and sum(g['rows'] for g in groups) == row[partition+'_rows'] and
                        all(g['threshold'] == row['threshold'] for g in groups), 'Incomplete fairness subgroups')
                # Check aggregate classifications against the held-out evidence.
                matrix = [[sum(g['tn'] for g in groups), sum(g['fp'] for g in groups)],
                          [sum(g['fn'] for g in groups), sum(g['tp'] for g in groups)]]
                require(matrix == row[partition+'_confusion_matrix'], 'Fairness confusion matrix mismatch')
                require(len([g for g in fairness['disparities'] if (g['candidate'],g['partition'],g['attribute']) == (label,partition,attribute)]) == 1,
                        'Missing fairness disparity review')
        report = load_json(robust / label / 'robustness_report.json')
        meta = report['reference_model']
        require(meta['model_sha256'] == row['model_sha256'] and meta['decision_threshold'] == row['threshold'] and
                meta['model_family'] == row['family'] and meta['preprocessing_variant'] == row['variant'],
                f'Robustness artifact mismatch: {label}')
        require(set(report['scenarios']) == SCENARIOS, 'Missing robustness scenarios')
        for scenario in report['scenarios'].values():
            require(scenario['rows'] == heldout['test_rows'], 'Robustness row count mismatch')
            for field in ('f1','precision','recall','accuracy','roc_auc','pr_auc'):
                rate(scenario[field])
        control = report['scenarios']['control']
        for field in ('f1','precision','recall','accuracy','roc_auc','pr_auc','confusion_matrix'):
            require(control[field] == row['test_'+field], f'Robustness control mismatch: {label}/{field}')
        robust_reports[label] = report
    drift = load_json(root / 'data/drift/drift_metadata.json')
    require(drift['reference_verification']['verified'] is True and drift['reference_rows'] == heldout['test_rows']
            and set(drift['scenarios']) == SCENARIOS and all(s['rows'] == heldout['test_rows'] for s in drift['scenarios'].values()),
            'Drift provenance mismatch')
    # Hash source data and saved partitions only; no predictions or threshold search.
    for filename, digest in heldout['raw_source_sha256'].items():
        require(filename in ('adult.data','adult.test'), 'Unexpected raw data path')
        verify_hash(root / 'data/raw' / filename, digest)
    require(set(heldout['raw_source_sha256']) == {'adult.data','adult.test'}, 'Missing source data fingerprints')
    for variant, files in heldout['saved_test_sha256'].items():
        require(variant in ('baseline','robust','interactions') and set(files) == {'X_test.csv','y_test.csv'}, 'Invalid saved test paths')
        for filename, digest in files.items():
            verify_hash(root / 'data/processed' / variant / filename, digest)
    require(candidate['variant'] in heldout['saved_test_sha256'], 'Missing selected partition fingerprints')
    evidence = dict(summary=summary, plan=plan, identity=identity, previous_reference=previous,
                    heldout=heldout, fairness=fairness, robustness=robust_reports, drift=drift)
    return model_path, candidate, selected, evidence


def verify_pipeline(path, selected):
    """Inspect the trusted fitted pipeline and its exact application input schema."""
    model = joblib.load(path)  # Local, checksum-verified artifact only.
    require(callable(getattr(model, 'predict_proba', None)) and list(model.classes_) == [0,1], 'Invalid binary model')
    engineer = model.named_steps['preprocess'].named_steps['feature_engineering']
    required = list(engineer.required_features_)
    require(required == selected['required_features'] and 'fnlwgt' not in required and len(set(required)) == len(required),
            'Application feature schema mismatch')
    return required


def build_reference_metadata(candidate, selected, evidence, approval, required):
    """Build a self-contained reference record for later reproducibility and audit."""
    identity = evidence['identity']
    return dict(model_family=candidate['family'], preprocessing_variant=candidate['variant'],
                decision_threshold=candidate['threshold'], positive_class=1,
                validation_f1=candidate['val_f1'], validation_pr_auc=candidate['val_pr_auc'],
                validation_roc_auc=candidate['val_roc_auc'], mlflow_run_id=candidate['run_id'],
                execution_id=identity['execution_id'], git_commit=identity['git_commit'],
                dataset_sha256={'raw_train.csv':identity['raw_training_sha256'], 'raw_val.csv':identity['raw_validation_sha256'],
                                **evidence['heldout']['saved_test_sha256'][candidate['variant']]},
                effective_preprocessing_sha256=candidate['effective_preprocessing_sha256'],
                source_model_path=f"models/runs/{identity['execution_id']}/{candidate['candidate']}/model_pipeline.joblib",
                model_filename=MODEL_FILENAME, model_sha256=candidate['model_sha256'],
                representation=selected['representation'], required_features=required,
                parameters=candidate['parameters'], threshold_policy='training_only_group_calibration',
                promoted_at=datetime.now(timezone.utc).isoformat(), promotion_type='explicit_frozen',
                promotion_source=f"models/runs/{identity['execution_id']}/{candidate['candidate']}",
                approval_sha256=approval, educational_use_only=True,
                review_notes=['Fairness and robustness reviewed descriptively; no quantitative acceptance thresholds.',
                              'Historical test previously used; no claim of independent external validation.',
                              'Sensitive predictors and severe drift limitations remain. Not for high-stakes decisions.'],
                evidence=evidence)


# The standalone files are derived from the approved embedded evidence.
EVALUATION_FILES = {
    'validation.json': 'summary',
    'frozen_plan.json': 'plan',
    'partition_identity.json': 'identity',
    'heldout.json': 'heldout',
    'fairness.json': 'fairness',
    'robustness.json': 'robustness',
    'drift.json': 'drift',
    'previous_reference.json': 'previous_reference',
}


def evaluation_payloads(metadata):
    """Derive a release bundle only from the approved promotion record."""
    evidence = metadata['evidence']
    require(canonical_sha(evidence) == metadata['approval_sha256'], 'Approved evidence digest mismatch')
    return {filename: evidence[key] for filename, key in EVALUATION_FILES.items()}


def write_evaluation_bundle(directory, metadata):
    """Write evidence and a byte-level manifest to a new staging directory."""
    payloads = evaluation_payloads(metadata)
    directory.mkdir(exist_ok=False)
    for filename, value in payloads.items():
        (directory / filename).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    manifest = dict(schema_version=1, model_sha256=metadata['model_sha256'],
                    approval_sha256=metadata['approval_sha256'],
                    files={name: sha256_file(directory / name) for name in payloads})
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    verify_evaluation_bundle(directory, metadata)


def verify_evaluation_bundle(directory, metadata):
    """Check exact file membership, byte hashes and approved JSON contents."""
    payloads = evaluation_payloads(metadata)
    require(directory.is_dir() and not directory.is_symlink(), 'Missing or unsafe evaluation directory')
    require({p.name for p in directory.iterdir()} == set(payloads) | {'manifest.json'},
            'Unexpected evaluation bundle contents')
    require(all(p.is_file() and not p.is_symlink() for p in directory.iterdir()), 'Unsafe evaluation file')
    manifest = load_json(directory / 'manifest.json')
    require(manifest == dict(schema_version=1, model_sha256=metadata['model_sha256'],
                            approval_sha256=metadata['approval_sha256'],
                            files={name: sha256_file(directory / name) for name in payloads}),
            'Evaluation manifest mismatch')
    for filename, value in payloads.items():
        require(canonical_sha(load_json(directory / filename)) == canonical_sha(value),
                f'Approved evaluation content mismatch: {filename}')


def install_reference(root, source, metadata, previous):
    """Stage the new directory, retain the previous version and recover on exceptions."""
    reference = root / 'models/reference'
    history = root / 'models/reference_history'
    history.mkdir(parents=True, exist_ok=True)
    require(not history.is_symlink() and not getattr(history, 'is_junction', lambda: False)(),
            'Unsafe rollback history directory')
    lock = history / '.promotion.lock'
    lock.mkdir()  # Concurrent promotion or unresolved interruption fails closed.
    transaction = history / uuid.uuid4().hex
    backup = transaction / 'reference'
    staged = transaction / 'staged'
    keep_lock = False
    try:
        (lock / 'transaction').write_text(transaction.name, encoding='utf-8')
        # Finish publication and recovery of previous promotions first.
        require(not any(p != lock for p in history.iterdir()),
                'Pending rollback transaction; finalize the previous release first')
        # Recheck the old version before any replacement.
        require(load_json(reference / METADATA_FILENAME) == previous, 'Reference metadata changed during promotion')
        verify_hash(reference / MODEL_FILENAME, previous['model_sha256'])
        staged.mkdir(parents=True)
        shutil.copy2(source, staged / MODEL_FILENAME)
        verify_hash(staged / MODEL_FILENAME, metadata['model_sha256'])
        (staged / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
        write_evaluation_bundle(staged / 'evaluation', metadata)
        pointer = root / 'models/reference.dvc'
        require(pointer.is_file() and not pointer.is_symlink(), 'Missing or unsafe reference DVC pointer')
        shutil.copy2(pointer, transaction / 'reference.dvc')
        # Preserve the old directory intact; never write through DVC-linked files.
        try:
            os.rename(reference, backup)
            os.rename(staged, reference)
            verify_hash(reference / MODEL_FILENAME, metadata['model_sha256'])
            require(load_json(reference / METADATA_FILENAME) == metadata, 'Installed metadata mismatch')
            verify_evaluation_bundle(reference / 'evaluation', metadata)
        except BaseException:
            if backup.exists():
                # Keep the recovery lock if rollback itself cannot complete.
                keep_lock = True
                if reference.exists():
                    os.rename(reference, transaction / 'failed_reference')
                os.rename(backup, reference)
                keep_lock = False
            raise
        return backup
    finally:
        # Hard termination or failed rollback leaves a transaction marker for recovery.
        if not keep_lock:
            # A failed staging operation or successful rollback leaves no backup to publish.
            if transaction.exists() and not backup.exists() and reference.exists() and not (transaction / 'failed_reference').exists():
                shutil.rmtree(transaction)
            (lock / 'transaction').unlink(missing_ok=True)
            lock.rmdir()


def promote_candidate(run_id, heldout_id=None, fairness_id=None, *, confirm=False, approval_sha256=None, root=ROOT):
    """Validate the frozen evidence and install only after explicit approval of its digest."""
    require(heldout_id is not None and fairness_id is not None, 'Frozen held-out and fairness IDs required; legacy promotion disabled')
    root = Path(root)
    source, candidate, selected, evidence = read_evidence(root, run_id, heldout_id, fairness_id)
    approval = canonical_sha(evidence)
    # Confirm the independently reviewed evidence before deserializing the artifact.
    if confirm:
        require(approval_sha256 == approval, 'Approval SHA-256 mismatch; review the dry-run evidence first')
    required = verify_pipeline(source, selected)
    result = dict(execution_id=run_id, candidate=candidate['candidate'], candidate_sha256=candidate['model_sha256'],
                  previous_sha256=evidence['previous_reference']['model_sha256'], approval_sha256=approval,
                  validated=True, promoted=False)
    if confirm:
        metadata = build_reference_metadata(candidate, selected, evidence, approval, required)
        result['backup_path'] = str(install_reference(root, source, metadata, evidence['previous_reference']))
        result['promoted'] = True
    return result


def main():
    """Parse explicit promotion arguments; default to a non-mutating dry run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--heldout-id', required=True)
    parser.add_argument('--fairness-id', required=True)
    parser.add_argument('--approval-sha256')
    parser.add_argument('--confirm', action='store_true')
    args = parser.parse_args()
    try:
        result = promote_candidate(args.run_id, args.heldout_id, args.fairness_id, confirm=args.confirm,
                                   approval_sha256=args.approval_sha256)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        parser.exit(1, f'Promotion rejected: {exc}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
