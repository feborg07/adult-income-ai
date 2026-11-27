"""Verify cleanup selection, evidence preservation and fail-closed safeguards."""
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import zipfile

import pytest

from scripts import clean_project as cp


@pytest.fixture
def project(tmp_path):
    """Create an isolated Git checkout containing protected and generated files."""
    root = tmp_path / 'project'
    root.mkdir()
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    for name in ('models/reference.dvc', 'data/raw.dvc', 'src/module.py', 'requirements.txt'):
        put(root, name, 'tracked')
    for name, output in cp.KNOWN_POINTERS.items():
        put(root, name, f'outs:\n- path: {output}\n')
    subprocess.run(['git', '-C', str(root), 'add', '.'], check=True)
    for name in ('models/reference/best_model_pipeline.joblib', 'data/raw/adult.data',
                 '.dvc/cache/file', '.venv/lib/__pycache__/x.pyc',
                 'models/reference_history/old/model.joblib', 'reports/archive/keep.json',
                 'models/runs/' + 'a' * 32 + '/xgboost/model_pipeline.joblib',
                 'models/runs/' + 'b' * 32 + '/cv_leaderboard.csv',
                 'models/runs/' + 'b' * 32 + '/fold.joblib', 'mlartifacts/fold.joblib',
                 'mlflow.db', 'models/optimization/optuna.db', 'reports/heldout/summary.json',
                 'data/processed/X.csv', 'models/preprocessors/baseline.joblib',
                 'data/drift/control/X.csv', 'src/__pycache__/x.pyc', '.pytest_cache/nodeids'):
        put(root, name, name)
    return root


def put(root, name, contents):
    """Write a test fixture at the specified relative path."""
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding='utf-8')
    return path


def mock_release(monkeypatch):
    """Supply verified release metadata for isolated cleanup-only tests."""
    monkeypatch.setattr(cp, 'release_metadata', lambda root: {'execution_id': 'a' * 32, 'model_sha256': 'b' * 64})
    original = cp.git_output
    monkeypatch.setattr(cp, 'git_output', lambda root, *args: b'commit\n' if args == ('rev-parse', 'HEAD') else original(root, *args))


def test_dry_run_never_deletes(project, tmp_path):
    """Preview all categories without writing an archive or deleting anything."""
    archive = tmp_path / 'evidence.zip'
    before = set(project.rglob('*'))
    assert cp.clean_project(cp.CATEGORIES, dry_run=True, yes=True, archive=archive, root=project) == []
    assert before == set(project.rglob('*'))
    assert not archive.exists()


def test_cache_processed_and_drift(project):
    """Delete regenerable categories without touching permanent assets."""
    cp.clean_project({'cache', 'processed', 'drift'}, yes=True, root=project)
    for name in ('data/processed', 'data/drift', 'models/preprocessors', 'src/__pycache__', '.pytest_cache'):
        assert not (project / name).exists()
    for name in ('.git', '.dvc/cache/file', '.venv/lib/__pycache__/x.pyc', 'data/raw/adult.data',
                 'models/reference/best_model_pipeline.joblib', 'models/reference_history/old/model.joblib',
                 'src/module.py', 'mlflow.db', 'reports/archive/keep.json'):
        assert (project / name).exists()


def test_archive_then_clean(project, tmp_path, monkeypatch):
    """Preserve the frozen shortlist, summaries and reports with verified hashes."""
    mock_release(monkeypatch)
    archive = tmp_path / 'evidence.zip'
    cp.clean_project({'experiments', 'reports'}, yes=True, archive=archive, root=project)
    for name in ('mlflow.db', 'mlartifacts', 'models/runs', 'models/optimization', 'reports/heldout'):
        assert not (project / name).exists()
    assert (project / 'reports/archive/keep.json').exists()
    assert (project / 'models/reference/best_model_pipeline.joblib').exists()
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('cleanup_manifest.json'))
        names = set(manifest['files'])
        assert 'models/runs/' + 'a' * 32 + '/xgboost/model_pipeline.joblib' in names
        assert 'models/runs/' + 'b' * 32 + '/cv_leaderboard.csv' in names
        assert 'models/runs/' + 'b' * 32 + '/fold.joblib' not in names
        assert 'reports/heldout/summary.json' in names
        assert 'mlflow.db' not in names
        assert 'models/reference.dvc' in names
        for name, checksum in manifest['files'].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == checksum


def test_archive_required(project):
    """Refuse experimental deletion without an evidence destination."""
    with pytest.raises(ValueError, match='--archive'):
        cp.clean_project({'experiments'}, yes=True, root=project)
    assert (project / 'mlflow.db').exists()


def test_pending_backup_blocks_cleanup(project, tmp_path):
    """Do not erase experiments while a reference rollback is unresolved."""
    with pytest.raises(ValueError, match='Pending reference'):
        cp.clean_project({'experiments'}, yes=True, archive=tmp_path / 'e.zip', root=project)
    assert (project / 'mlflow.db').exists()


@pytest.mark.parametrize('location', ['inside', 'existing'])
def test_bad_archive_destination(project, tmp_path, monkeypatch, location):
    """Reject archives inside the project or overwriting an existing archive."""
    mock_release(monkeypatch)
    destination = (project if location == 'inside' else tmp_path) / 'e.zip'
    if location == 'existing':
        destination.write_text('keep')
    with pytest.raises(ValueError, match='outside|already exists'):
        cp.clean_project({'experiments'}, yes=True, archive=destination, root=project)
    assert (project / 'mlflow.db').exists()


def test_tracked_file_blocks_cleanup(project):
    """Never delete a target containing a tracked file."""
    subprocess.run(['git', '-C', str(project), 'add', 'data/processed/X.csv'], check=True)
    with pytest.raises(ValueError, match='tracked files'):
        cp.clean_project({'processed'}, yes=True, root=project)
    assert (project / 'data/processed/X.csv').exists()


def test_unexpected_dvc_pointer_blocks_cleanup(project):
    """Require review if a new tracked DVC output is introduced."""
    put(project, 'new.dvc', 'pointer')
    subprocess.run(['git', '-C', str(project), 'add', 'new.dvc'], check=True)
    with pytest.raises(ValueError, match='DVC configuration'):
        cp.clean_project({'cache'}, yes=True, root=project)


def test_nested_dvc_pointer_blocks_cleanup(project):
    """Reject even untracked nested DVC metadata inside deletion targets."""
    put(project, 'data/processed/hidden.dvc', 'pointer')
    with pytest.raises(ValueError, match='Protected Git/DVC'):
        cp.clean_project({'processed'}, yes=True, root=project)


def test_link_blocks_cleanup(project, tmp_path):
    """Do not follow linked descendants into external directories."""
    external = tmp_path / 'outside'
    external.mkdir()
    put(external, 'keep.txt', 'keep')
    try:
        (project / 'data/processed/link').symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip('Symlinks unavailable on this platform')
    with pytest.raises(ValueError, match='Unsafe link'):
        cp.clean_project({'processed'}, yes=True, root=project)
    assert (external / 'keep.txt').exists()


def test_cancel(project, monkeypatch):
    """Require affirmative confirmation unless explicitly bypassed."""
    monkeypatch.setattr('builtins.input', lambda _: 'n')
    assert cp.clean_project({'cache'}, root=project) == []
    assert (project / '.pytest_cache').exists()


def test_archive_failure_keeps_originals(project, tmp_path, monkeypatch):
    """Do not delete any targets if evidence archival fails."""
    mock_release(monkeypatch)
    def fail(*args):
        """Simulate an archive write failure."""
        raise OSError('disk full')
    monkeypatch.setattr(cp, 'write_archive', fail)
    with pytest.raises(OSError, match='disk full'):
        cp.clean_project({'experiments', 'processed'}, yes=True, archive=tmp_path / 'e.zip', root=project)
    assert (project / 'mlflow.db').exists()
    assert (project / 'data/processed/X.csv').exists()


def test_idempotent(project):
    """Repeated cleanup of regenerable outputs is harmless."""
    cp.clean_project({'cache'}, yes=True, root=project)
    assert cp.clean_project({'cache'}, yes=True, root=project) == []


def make_released_reference(project):
    """Build real approved evidence and commit the synthetic reference pointer."""
    from src.models import promote as p
    shutil.rmtree(project / 'models/reference_history')
    reference = project / 'models/reference'
    evidence = {key: {'record': key} for key in p.EVALUATION_FILES.values()}
    metadata = {'execution_id': 'a' * 32, 'model_sha256': p.sha256_file(reference / p.MODEL_FILENAME),
                'evidence': evidence, 'approval_sha256': p.canonical_sha(evidence)}
    put(project, 'models/reference/' + p.METADATA_FILENAME, json.dumps(metadata))
    p.write_evaluation_bundle(reference / 'evaluation', metadata)
    subprocess.run(['git', '-C', str(project), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'Release pointer'], check=True)
    return metadata


def test_real_release_bundle(project, tmp_path):
    """Validate the real bundle and committed pointer before evidence deletion."""
    metadata = make_released_reference(project)
    archive = tmp_path / 'verified.zip'
    cp.clean_project({'experiments'}, yes=True, archive=archive, root=project)
    assert cp.release_metadata(project) == metadata
    with zipfile.ZipFile(archive) as z:
        assert 'models/reference/evaluation/manifest.json' in z.namelist()
        assert 'data/raw.dvc' in z.namelist()


def test_tampered_reference_blocks_cleanup(project, tmp_path):
    """Reject evidence deletion if a release file has been changed."""
    make_released_reference(project)
    put(project, 'models/reference/evaluation/validation.json', '{}')
    with pytest.raises(ValueError, match='manifest mismatch'):
        cp.clean_project({'experiments'}, yes=True, archive=tmp_path / 'e.zip', root=project)
    assert (project / 'mlflow.db').exists()


def test_changed_pointer_output(project):
    """Reject a known pointer redirected to a generated output directory."""
    put(project, 'models/reference.dvc', 'outs:\n- path: runs\n')
    with pytest.raises(ValueError, match='Unexpected DVC output'):
        cp.clean_project({'cache'}, yes=True, root=project)
