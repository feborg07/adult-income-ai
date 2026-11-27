"""Tests for frozen evidence approval and recoverable reference replacement."""
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import pytest

from src.models import promote as p


class FittedModel:
    """Minimal serialized binary pipeline contract for promotion tests."""

    def __init__(self):
        """Expose the fitted input schema and class ordering."""
        self.classes_ = [0, 1]
        self.named_steps = {'preprocess': SimpleNamespace(named_steps={
            'feature_engineering': SimpleNamespace(required_features_=['age', 'sex'])})}

    def predict_proba(self, X):
        """Provide the inference interface; promotion must never call it."""
        raise AssertionError('Promotion must not predict or fit')


def write(path, value):
    """Write a synthetic JSON fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def modify(path, **changes):
    """Change a fixture record to exercise rejection paths."""
    record = p.load_json(path)
    record.update(changes)
    write(path, record)


@pytest.fixture
def env(tmp_path):
    """Create a self-contained frozen run, reference and reviewed evidence."""
    root = tmp_path
    run_id, held_id, fair_id = ('a'*32,'b'*32,'c'*32)
    run = root / 'models/runs' / run_id
    ref = root / 'models/reference'
    model = run / 'candidate/model_pipeline.joblib'
    model.parent.mkdir(parents=True)
    ref.mkdir(parents=True)
    joblib.dump(FittedModel(), model)
    (ref / p.MODEL_FILENAME).write_bytes(b'old reference')
    (root / 'models/reference.dvc').write_text('old DVC pointer')
    old = dict(model_sha256=p.sha256_file(ref / p.MODEL_FILENAME), decision_threshold=.4)
    write(ref / p.METADATA_FILENAME, old)
    candidate = dict(candidate='candidate', family='any_algorithm', variant='baseline', run_id='d'*32,
                     model_sha256=p.sha256_file(model), threshold=.35, val_f1=.72, val_pr_auc=.8,
                     val_roc_auc=.9, effective_preprocessing_sha256='1'*64, parameters={'depth': 4})
    identity = dict(execution_id=run_id, test_accessed=False, representation='no_fnlwgt',
                    raw_training_sha256='2'*64, raw_validation_sha256='3'*64, fold_sha256='4'*64,
                    git_commit='5'*40)
    summary = {**identity,'candidates':[candidate]}
    plan = dict(representation='no_fnlwgt', raw_training_sha256=identity['raw_training_sha256'],
                fold_sha256=identity['fold_sha256'], candidates={'candidate': {'family':'any_algorithm'}})
    write(run/'candidate_validation_summary.json', summary)
    write(run/'frozen_plan.json', plan)
    write(run/'partition_identity.json', identity)
    write(run/'candidate/validation_result.json', candidate)
    metrics = dict(f1=.72, precision=.72, recall=.72, accuracy=.72, roc_auc=.9, pr_auc=.8,
                   confusion_matrix=[[5,1],[1,3]])
    test = {f'{partition}_{k}':v for partition in ('test','unseen_test') for k,v in metrics.items()}
    common = dict(**test,test_rows=10,unseen_test_rows=10)
    selected = dict(**candidate,**common, representation='no_fnlwgt',required_features=['age','sex'])
    reference = dict(**common, candidate='reference', family='old_algorithm',variant='baseline',
                     representation='control', model_sha256=old['model_sha256'],threshold=.4)
    raw = {}
    for name in ('adult.data','adult.test'):
        path = root/'data/raw'/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(b'raw data')
        raw[name] = p.sha256_file(path)
    saved = {'baseline':{}}
    for name in ('X_test.csv','y_test.csv'):
        path = root/'data/processed/baseline'/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(b'test partition')
        saved['baseline'][name] = p.sha256_file(path)
    partition = dict(raw_source_sha256=raw,saved_test_sha256=saved,unseen_mask_sha256='6'*64,
                     test_rows=10,unseen_test_rows=10)
    gate = dict(selected_candidate='candidate', margin=.01, passed=True, checks={
        k:dict(candidate=.72,reference=.72,minimum=.71,passed=True) for k in ('test_f1','unseen_test_f1')})
    heldout = dict(**partition, execution_id=held_id, source_execution_id=run_id,
                   evaluation_protocol='frozen_heldout',test_accessed=True,selected_candidate='candidate',
                   partition_identity=identity, evaluator_sha256='7'*64, performance_gate=gate,
                   candidates=[selected,reference],frozen_metadata_content_sha256={
                       name:p.canonical_sha(value) for name,value in (
                           ('candidate_validation_summary.json',summary),('frozen_plan.json',plan),
                           ('partition_identity.json',identity),(p.METADATA_FILENAME,old))})
    hp=root/'reports/heldout'/held_id/'heldout_summary.json'
    write(hp,heldout)
    groups,disparities=[],[]
    for row in (selected,reference):
        for part in ('test','unseen_test'):
            for attr in ('sex','race','sex+race'):
                base=dict(candidate=row['candidate'],partition=part,attribute=attr)
                groups.append(dict(**base,rows=10,threshold=row['threshold'],tn=5,fp=1,fn=1,tp=3))
                disparities.append(base)
        write(root/'reports/robustness/phase4'/row['candidate']/'robustness_report.json',dict(
            reference_model=dict(model_family=row['family'],preprocessing_variant='baseline',
                                 model_sha256=row['model_sha256'],decision_threshold=row['threshold']),
            scenarios={s:dict(**metrics,rows=10) for s in p.SCENARIOS}))
    fp=root/'reports/fairness'/fair_id/'fairness_summary.json'
    write(fp,dict(**partition,execution_id=fair_id,heldout_execution_id=held_id,source_execution_id=run_id,
                  heldout_summary_sha256=p.sha256_file(hp),heldout_evaluator_sha256=heldout['evaluator_sha256'],
                  models={r['candidate']:{k:r[k] for k in ('model_sha256','threshold','representation')}
                          for r in (selected,reference)},subgroups=groups,disparities=disparities))
    write(root/'data/drift/drift_metadata.json',dict(reference_verification={'verified':True},reference_rows=10,
                                                     scenarios={s:{'rows':10} for s in p.SCENARIOS}))
    return SimpleNamespace(root=root, ids=(run_id,held_id,fair_id),model=model,ref=ref,old=old,heldout=hp,fairness=fp,run=run)


def run(env, **kwargs):
    """Invoke promotion with isolated fixture paths."""
    return p.promote_candidate(*env.ids,root=env.root,**kwargs)


def confirm(env):
    """Approve exactly the digest obtained from a dry run."""
    return run(env,confirm=True,approval_sha256=run(env)['approval_sha256'])


def test_dry_run(env):
    """A dry run validates without writing or creating history."""
    result=run(env)
    assert result['validated'] and not result['promoted']
    assert not (env.root/'models/reference_history').exists()
    assert p.load_json(env.ref/p.METADATA_FILENAME)==env.old


def test_confirmed_promotion(env):
    """Install identical candidate bytes and preserve old reference and pointer."""
    result=confirm(env)
    backup=Path(result['backup_path'])
    assert result['promoted']
    assert p.sha256_file(env.ref/p.MODEL_FILENAME)==p.sha256_file(env.model)
    assert p.load_json(backup/p.METADATA_FILENAME)==env.old
    assert p.sha256_file(backup/p.MODEL_FILENAME)==env.old['model_sha256']
    assert (backup.parent/'reference.dvc').read_text()=='old DVC pointer'
    assert (env.root/'models/reference.dvc').read_text()=='old DVC pointer'
    meta=p.load_json(env.ref/p.METADATA_FILENAME)
    assert meta['model_family']=='any_algorithm' and meta['decision_threshold']==.35
    assert meta['required_features']==['age','sex'] and meta['educational_use_only']
    assert meta['approval_sha256']==result['approval_sha256'] and 'fairness' in meta['evidence']
    assert set(x.name for x in env.ref.iterdir())=={p.MODEL_FILENAME,p.METADATA_FILENAME,'evaluation'}
    p.verify_evaluation_bundle(env.ref/'evaluation', meta)


@pytest.mark.parametrize('approval',[None,'0'*64])
def test_wrong_approval(env,approval):
    """Confirmation without the reviewed digest cannot modify the reference."""
    with pytest.raises(ValueError,match='Approval SHA-256'):
        run(env,confirm=True,approval_sha256=approval)
    assert p.load_json(env.ref/p.METADATA_FILENAME)==env.old


def test_legacy_rejected(env):
    """Legacy full training cannot bypass evidence requirements."""
    with pytest.raises(ValueError,match='legacy promotion disabled'):
        p.promote_candidate(env.ids[0],confirm=True,root=env.root)


@pytest.mark.parametrize('value',['../candidate','A'*32,'a'*31,None])
def test_invalid_id(value):
    """Reject unsafe execution IDs."""
    with pytest.raises(ValueError,match='Invalid execution ID'):
        p.validate_run_id(value)


def test_tampered_candidate(env):
    """Reject modified model bytes before loading."""
    env.model.write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA-256 mismatch'):
        run(env)


def test_tampered_metadata(env):
    """Reject changes to frozen candidate metadata."""
    modify(env.run/'candidate/validation_result.json',threshold=.9)
    with pytest.raises(ValueError,match='Frozen candidate mismatch'):
        run(env)


def test_wrong_winner(env):
    """Do not permit a different candidate to be selected after test access."""
    modify(env.heldout,selected_candidate='another')
    with pytest.raises(ValueError,match='validation winner'):
        run(env)


def test_failed_gate(env):
    """Reject a failed performance gate before promotion."""
    gate=p.load_json(env.heldout)['performance_gate']
    gate['passed']=False
    modify(env.heldout,performance_gate=gate)
    with pytest.raises(ValueError,match='performance gate failed'):
        run(env)


def test_recompute_gate(env):
    """A true gate flag cannot override a failing F1 margin."""
    rows=p.load_json(env.heldout)['candidates']
    rows[0]['test_f1']=.6
    modify(env.heldout,candidates=rows)
    with pytest.raises(ValueError,match='Performance gate failed'):
        run(env)


def test_mismatched_fairness(env):
    """Reject fairness reports tied to a different held-out execution."""
    modify(env.fairness,heldout_execution_id='e'*32)
    with pytest.raises(ValueError,match='Fairness execution mismatch'):
        run(env)


def test_missing_fairness_groups(env):
    """Require subgroup counts and prediction parity."""
    modify(env.fairness,subgroups=[])
    with pytest.raises(ValueError,match='Incomplete fairness'):
        run(env)


def test_missing_robustness(env):
    """Require all drift scenarios."""
    path=env.root/'reports/robustness/phase4/candidate/robustness_report.json'
    modify(path,scenarios={})
    with pytest.raises(ValueError,match='Missing robustness scenarios'):
        run(env)


def test_robustness_control_mismatch(env):
    """Require control scores to reproduce held-out metrics."""
    path=env.root/'reports/robustness/phase4/candidate/robustness_report.json'
    scenarios=p.load_json(path)['scenarios']
    scenarios['control']['f1']=.5
    modify(path,scenarios=scenarios)
    with pytest.raises(ValueError,match='Robustness control mismatch'):
        run(env)


def test_changed_evidence_after_dry_run(env):
    """Require fresh approval when reviewed descriptive evidence changes."""
    approval=run(env)['approval_sha256']
    modify(env.fairness,review_note='changed')
    with pytest.raises(ValueError,match='Approval SHA-256 mismatch'):
        run(env,confirm=True,approval_sha256=approval)


def test_changed_dataset(env):
    """Reject a changed source dataset without scoring it."""
    (env.root/'data/raw/adult.data').write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA-256 mismatch'):
        run(env)


def test_application_schema(env,monkeypatch):
    """Reject a model that requires the census weight."""
    model=FittedModel()
    model.named_steps['preprocess'].named_steps['feature_engineering'].required_features_.append('fnlwgt')
    monkeypatch.setattr(p.joblib,'load',lambda path:model)
    with pytest.raises(ValueError,match='Application feature schema mismatch'):
        run(env)


def test_rollback_on_install_failure(env,monkeypatch):
    """Restore the old directory if the staged directory cannot be installed."""
    original=p.os.rename
    def fail_staged(src,dst):
        """Inject a failure only when installing the new reference."""
        if Path(src).name=='staged':
            raise OSError('simulated install failure')
        original(src,dst)
    monkeypatch.setattr(p.os,'rename',fail_staged)
    with pytest.raises(OSError,match='simulated install failure'):
        confirm(env)
    assert p.load_json(env.ref/p.METADATA_FILENAME)==env.old
    assert p.sha256_file(env.ref/p.MODEL_FILENAME)==env.old['model_sha256']


def test_existing_lock(env):
    """Prevent overlapping or interrupted promotion transactions."""
    lock=env.root/'models/reference_history/.promotion.lock'
    lock.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        confirm(env)
    assert lock.exists() and p.load_json(env.ref/p.METADATA_FILENAME)==env.old


def test_repeat_promotion_rejected(env):
    """The same approval cannot overwrite the new reference again."""
    result=confirm(env)
    with pytest.raises(ValueError,match='Frozen metadata changed'):
        run(env,confirm=True,approval_sha256=result['approval_sha256'])


def test_failed_rollback_retains_lock(env,monkeypatch):
    """Preserve old reference and recovery marker if installation and rollback fail."""
    original=p.os.rename
    def fail_recovery(src,dst):
        """Simulate failures for staged installation and backup restoration."""
        if Path(dst)==env.ref:
            raise OSError('simulated rename failure')
        original(src,dst)
    monkeypatch.setattr(p.os,'rename',fail_recovery)
    with pytest.raises(OSError,match='simulated rename failure'):
        confirm(env)
    history=env.root/'models/reference_history'
    marker=history/'.promotion.lock/transaction'
    backup=history/marker.read_text()/'reference'
    assert p.sha256_file(backup/p.MODEL_FILENAME)==env.old['model_sha256']
    assert p.load_json(backup/p.METADATA_FILENAME)==env.old
    assert not env.ref.exists()


def test_staging_copy_failure(env,monkeypatch):
    """A staging failure leaves the original reference untouched."""
    def fail_copy(*args,**kwargs):
        """Simulate a copy failure before the original reference is moved."""
        raise OSError('simulated copy failure')
    monkeypatch.setattr(p.shutil,'copy2',fail_copy)
    with pytest.raises(OSError,match='simulated copy failure'):
        confirm(env)
    assert p.load_json(env.ref/p.METADATA_FILENAME)==env.old
    assert not (env.root/'models/reference_history/.promotion.lock').exists()


def test_pending_backup_blocks_new_promotion(env):
    """Require finalization of the prior release before replacing its successor."""
    pending = env.root / 'models/reference_history' / ('f' * 32) / 'reference'
    pending.mkdir(parents=True)
    with pytest.raises(ValueError, match='Pending rollback transaction'):
        confirm(env)
    assert pending.exists()
    assert p.load_json(env.ref / p.METADATA_FILENAME) == env.old


def test_missing_pointer_blocks_promotion(env):
    """Require a recoverable previous DVC pointer before reference replacement."""
    (env.root / 'models/reference.dvc').unlink()
    with pytest.raises(ValueError, match='reference DVC pointer'):
        confirm(env)
    assert p.load_json(env.ref / p.METADATA_FILENAME) == env.old
    assert not list((env.root / 'models/reference_history').iterdir())
