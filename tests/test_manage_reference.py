"""Test release bundling, remote tag recovery and guarded local backup removal."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts import manage_reference as r
from src.models import promote as p


TRANSACTION = 'a' * 32


def make_reference(path, *, legacy=False):
    """Create a synthetic reference with approved evidence or legacy metadata."""
    path.mkdir(parents=True)
    model = path / p.MODEL_FILENAME
    model.write_bytes(b'legacy' if legacy else b'current')
    metadata = {'model_sha256': p.sha256_file(model)}
    if not legacy:
        evidence = {key: {'record': key} for key in p.EVALUATION_FILES.values()}
        metadata.update(evidence=evidence, approval_sha256=p.canonical_sha(evidence))
    (path / p.METADATA_FILENAME).write_text(json.dumps(metadata), encoding='utf-8')
    return metadata


@pytest.fixture
def project(tmp_path):
    """Construct a current reference, pointer and previous rollback transaction."""
    root = tmp_path / 'project'
    ref = root / 'models/reference'
    make_reference(ref)
    (root / r.POINTER).write_bytes(b'current pointer\r\n')
    transaction = root / 'models/reference_history' / TRANSACTION
    make_reference(transaction / 'reference', legacy=True)
    (transaction / 'reference.dvc').write_bytes(b'old pointer\r\n')
    return root


def test_bundle_preserves_bytes(project):
    """Migration preserves the original serialized model and metadata bytes."""
    ref = project / 'models/reference'
    before = r.tree_hashes(ref)
    assert r.bundle_current(project)['created']
    after = r.tree_hashes(ref)
    assert all(after[name] == digest for name, digest in before.items())
    assert len(after) == 11
    assert not r.bundle_current(project)['created']
    assert r.tree_hashes(ref) == after


def test_invalid_approval(project):
    """Reject metadata with unapproved evidence and leave no partial bundle."""
    ref = project / 'models/reference'
    meta = p.load_json(ref / p.METADATA_FILENAME)
    meta['evidence']['summary'] = {'changed': True}
    (ref / p.METADATA_FILENAME).write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='Approved evidence digest'):
        r.bundle_current(project)
    assert not (ref / 'evaluation').exists()


def test_tampered_bundle(project):
    """Reject a modified evidence file even when model bytes still match."""
    r.bundle_current(project)
    (project / 'models/reference/evaluation/validation.json').write_text('{}')
    with pytest.raises(ValueError, match='manifest mismatch'):
        r.bundle_current(project)


def test_locked(project):
    """Do not interfere with an ongoing or interrupted promotion."""
    lock = project / 'models/reference_history/.promotion.lock'
    lock.mkdir()
    with pytest.raises(FileExistsError):
        r.bundle_current(project)
    assert lock.exists()


@pytest.fixture
def mocked_recovery(project, monkeypatch):
    """Simulate successful remote recovery while recording every checked tag."""
    r.bundle_current(project)
    calls = []

    def fake_git(root, *args):
        """Return the committed pointer with Git-normalized line endings."""
        assert args == ('show', f'HEAD:{r.POINTER}')
        return b'current pointer\n'

    def recover(root, tag, hashes, pointer, *, require_bundle=False):
        """Record the intended release without network access."""
        calls.append((tag, require_bundle))
        return '1' * 40

    monkeypatch.setattr(r, 'git', fake_git)
    monkeypatch.setattr(r, 'recover_tag', recover)
    return calls


def test_verify_then_remove(project, mocked_recovery):
    """Dry verification retains backups; explicit removal checks both releases."""
    backup = project / 'models/reference_history' / TRANSACTION
    tags = {TRANSACTION: 'model-v1.0.0'}
    assert r.verify_releases('model-v2.0.0', tags, root=project)['backups_removed'] == []
    assert backup.exists()
    result = r.verify_releases('model-v2.0.0', tags, root=project, remove_backups=True)
    assert result['backups_removed'] == [TRANSACTION]
    assert not backup.exists()
    assert mocked_recovery == [('model-v2.0.0', True), ('model-v1.0.0', False)] * 2
    assert (project / 'models/reference').exists()


def test_recovery_failure_retains_backup(project, mocked_recovery, monkeypatch):
    """An unavailable historical release must prevent any deletion."""
    def recover(root, tag, *args, **kwargs):
        """Fail only the old release download."""
        if tag == 'model-v1.0.0':
            raise subprocess.CalledProcessError(1, ['dvc', 'pull'])
        return '1' * 40

    monkeypatch.setattr(r, 'recover_tag', recover)
    with pytest.raises(subprocess.CalledProcessError):
        r.verify_releases('model-v2.0.0', {TRANSACTION: 'model-v1.0.0'}, root=project, remove_backups=True)
    assert (project / 'models/reference_history' / TRANSACTION).exists()


def test_missing_mapping(project, mocked_recovery):
    """Reject deletion when a rollback transaction has no published tag."""
    with pytest.raises(ValueError, match='exactly one'):
        r.verify_releases('model-v2.0.0', {}, root=project, remove_backups=True)
    assert not mocked_recovery


def test_uncommitted_pointer(project, mocked_recovery):
    """Reject a pointer that has not been committed to Git."""
    (project / r.POINTER).write_bytes(b'changed')
    with pytest.raises(ValueError, match='Commit the current'):
        r.verify_releases('model-v2.0.0', {TRANSACTION: 'model-v1.0.0'}, root=project)
    assert not mocked_recovery


def test_unknown_transaction(project, mocked_recovery):
    """Never remove unrecognized or interrupted transaction directories."""
    (project / 'models/reference_history/unknown').mkdir()
    with pytest.raises(ValueError, match='Unexpected rollback'):
        r.verify_releases('model-v2.0.0', {TRANSACTION: 'model-v1.0.0'}, root=project, remove_backups=True)
    assert not mocked_recovery


def test_invalid_mapping():
    """Reject invalid tag names and duplicate transaction mappings."""
    for args in ([f'{TRANSACTION}=main'], [f'{TRANSACTION}=model-v1.0.0'] * 2):
        with pytest.raises(ValueError, match='Invalid or duplicate'):
            r.parse_backup_tags(args)


def test_recover_tag_real_git(project, tmp_path, monkeypatch):
    """Clone a real tagged Git origin and simulate only the DVC remote transfer."""
    r.bundle_current(project)
    origin = tmp_path / 'origin'
    origin.mkdir()
    subprocess.run(['git', 'init', '-q', str(origin)], check=True)
    (origin / 'models').mkdir()
    (origin / r.POINTER).write_bytes(b'current pointer\n')
    r.git(origin, 'add', '.')
    r.git(origin, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'release')
    r.git(origin, 'tag', 'model-v2.0.0')
    subprocess.run(['git', 'init', '-q', str(project)], check=True)
    r.git(project, 'remote', 'add', 'origin', str(origin))
    original = subprocess.run
    dvc_calls = []
    source = project / 'models/reference'

    def run(command, **kwargs):
        """Simulate DVC configuration and transfer while running real Git."""
        if command[1:3] != ['-m', 'dvc']:
            return original(command, **kwargs)
        dvc_calls.append(command[3:])
        if command[3] == 'pull':
            shutil.copytree(source, Path(kwargs['cwd']) / 'models/reference')
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(r.subprocess, 'run', run)
    assert r.recover_tag(project, 'model-v2.0.0', r.tree_hashes(source), b'current pointer\r\n', require_bundle=True)
    assert dvc_calls[0][:3] == ['config', '--local', 'cache.dir']
    assert dvc_calls[-1] == ['pull', '--remote', 'adult-income-ai', r.POINTER]
    with pytest.raises(ValueError, match='DVC pointer mismatch'):
        r.recover_tag(project, 'model-v2.0.0', r.tree_hashes(source), b'wrong pointer')


def test_reference_changed_during_recovery(project, mocked_recovery, monkeypatch):
    """Keep backups if the local reference changes during remote checks."""
    def recover(root, tag, *args, **kwargs):
        """Simulate an out-of-band write to the current model."""
        (root / 'models/reference' / p.MODEL_FILENAME).write_bytes(b'changed')
        return '1' * 40

    monkeypatch.setattr(r, 'recover_tag', recover)
    with pytest.raises(ValueError, match='Active reference changed'):
        r.verify_releases('model-v2.0.0', {TRANSACTION: 'model-v1.0.0'}, root=project, remove_backups=True)
    assert (project / 'models/reference_history' / TRANSACTION).exists()
