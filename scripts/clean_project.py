"""Clean generated project files using an explicit allowlist and evidence archive.

Stop training, MLflow, notebooks and other project writers before running.
No Git/DVC history, remote objects, raw data or reference releases are deleted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {
    'cache': ('.pytest_cache', '.mypy_cache', '.ruff_cache', '.coverage', 'htmlcov'),
    'experiments': ('mlflow.db', 'mlflow.db-shm', 'mlflow.db-wal', 'mlflow.db-journal',
                    'mlruns', 'mlartifacts', 'models/optimization', 'models/runs',
                    'models/training', 'models/best_model_pipeline.joblib',
                    'models/best_model_metadata.json'),
    'processed': ('data/processed', 'models/preprocessors'),
    'drift': ('data/drift',),
    'reports': ('reports/drift', 'reports/robustness', 'reports/heldout', 'reports/fairness',
                'reports/mlflow_comparison.csv', 'reports/optuna_final_comparison.csv',
                'reports/external_partition_overlap.json', 'reports/external_partition_overlap_v2.json'),
}
CACHE_ROOTS = ('src', 'scripts', 'tests', 'notebooks')
CACHE_NAMES = {'__pycache__', '.ipynb_checkpoints', '.pytest_cache', '.mypy_cache', '.ruff_cache'}
EVIDENCE_CATEGORIES = {'experiments', 'reports'}
KNOWN_POINTERS = {'data/raw.dvc': 'raw', 'models/reference.dvc': 'reference'}
PROTECTED_NAMES = {'.git', '.dvc', 'dvc.yaml', 'dvc.yml', 'dvc.lock'}
TEXT_SUFFIXES = {'.json', '.csv', '.yaml', '.yml', '.txt', '.md', '.tsv'}


def require(condition, message):
    """Fail closed when cleanup cannot be proven safe."""
    if not condition:
        raise ValueError(message)


def exists(path):
    """Detect broken links as well as regular files."""
    return os.path.lexists(path)


def safe_entry(path):
    """Reject symlinks, Windows junctions, and special filesystem entries."""
    info = path.lstat()
    reparse = getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
    require(not stat.S_ISLNK(info.st_mode) and not reparse,
            f'Unsafe link or reparse point: {path}')
    require(stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode),
            f'Unexpected filesystem entry: {path}')


def safe_path(root, path):
    """Check every existing component without following linked ancestors."""
    relative = path.relative_to(root)
    require(relative.parts, 'Cannot clean project root')
    current = root
    for part in relative.parts:
        current = current / part
        if exists(current):
            safe_entry(current)


def walk_tree(path):
    """Yield a regular tree without traversing links or nested repositories."""
    safe_entry(path)
    require(path.name not in PROTECTED_NAMES and not path.name.endswith('.dvc'),
            f'Protected Git/DVC entry: {path}')
    yield path
    if path.is_dir():
        for child in sorted(path.iterdir()):
            yield from walk_tree(child)


def git_output(root, *args):
    """Run a read-only Git command in the project checkout."""
    return subprocess.check_output(['git', '-C', str(root), *args])


def tracked_paths(root):
    """Require the actual checkout root and read all tracked paths."""
    top = Path(os.fsdecode(git_output(root, 'rev-parse', '--show-toplevel')).strip()).resolve()
    require(top == root, 'Cleanup must run in the project Git root')
    tracked = {os.fsdecode(p) for p in git_output(root, 'ls-files', '-z').split(b'\0') if p}
    # A new DVC output layout requires an explicit safety review.
    pointers = {p for p in tracked if p.endswith('.dvc')}
    require(pointers <= KNOWN_POINTERS.keys() and not any(Path(p).name in {'dvc.yaml', 'dvc.yml', 'dvc.lock'} for p in tracked),
            'Unexpected tracked DVC configuration; review cleanup allowlist first')
    # Check output locations, not only the pointer filenames.
    import yaml

    for name in pointers:
        pointer = root / name
        safe_path(root, pointer)
        try:
            data = yaml.safe_load(pointer.read_text(encoding='utf-8'))
            outputs = data['outs']
            require(isinstance(outputs, list) and len(outputs) == 1 and
                    outputs[0]['path'] == KNOWN_POINTERS[name],
                    f'Unexpected DVC output: {name}')
        except (yaml.YAMLError, TypeError, KeyError, IndexError) as error:
            raise ValueError(f'Invalid DVC pointer: {name}') from error
    return tracked


def cache_paths(root):
    """Find named caches only below the known code and notebook trees."""
    def visit(directory):
        """Visit ordinary directories, skipping unrelated links and repositories."""
        for child in sorted(directory.iterdir()):
            if child.name in CACHE_NAMES:
                yield child
            elif child.is_dir() and child.name not in PROTECTED_NAMES:
                try:
                    safe_entry(child)
                except ValueError:
                    continue
                yield from visit(child)
    for name in CACHE_ROOTS:
        path = root / name
        if exists(path):
            safe_path(root, path)
            if path.is_dir():
                yield from visit(path)
    yield from sorted(root.glob('.coverage.*'))
    yield root / '__pycache__'


def build_plan(root, categories):
    """Return validated existing deletion targets and their regular files."""
    require(categories and set(categories) <= CATEGORIES.keys(), 'Choose valid cleanup categories')
    tracked = tracked_paths(root)
    paths = {root / name for category in categories for name in CATEGORIES[category]}
    if 'cache' in categories:
        paths.update(cache_paths(root))
    plan = {}
    for path in sorted(paths):
        safe_path(root, path)
        if not exists(path):
            continue
        relative = path.relative_to(root).as_posix()
        # Case-insensitive matching also protects Windows checkouts.
        key = relative.casefold()
        require(not any(t.casefold() == key or t.casefold().startswith(key + '/') for t in tracked),
                f'Cleanup target contains tracked files: {relative}')
        entries = list(walk_tree(path))
        plan[path] = [entry for entry in entries if entry.is_file()]
    return plan


def sha256_stream(stream):
    """Compute a SHA-256 digest without loading large files into memory."""
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()


def sha256_file(path):
    """Fingerprint a regular local file."""
    with path.open('rb') as stream:
        return sha256_stream(stream)


def release_metadata(root):
    """Require a verified local reference bundle and no pending rollback transaction."""
    from scripts.manage_reference import reference_metadata

    reference = root / 'models/reference'
    history = root / 'models/reference_history'
    safe_path(root, reference)
    safe_path(root, history)
    if exists(history):
        require(history.is_dir() and not any(history.iterdir()),
                'Pending reference backup or promotion lock; resolve it with manage_reference first')
    # Validate every reference path before importing its metadata.
    list(walk_tree(reference))
    metadata = reference_metadata(reference, require_bundle=True)
    pointer = root / 'models/reference.dvc'
    safe_path(root, pointer)
    require(pointer.read_bytes().replace(b'\r\n', b'\n') ==
            git_output(root, 'show', 'HEAD:models/reference.dvc').replace(b'\r\n', b'\n'),
            'Commit the current reference DVC pointer before cleanup')
    return metadata


def evidence_paths(root, plan, metadata):
    """Collect report files, run records, frozen shortlist and release provenance."""
    result = set()
    run_root = root / 'models/runs'
    execution = metadata.get('execution_id')
    require(isinstance(execution, str) and re.fullmatch(r'[0-9a-f]{32}', execution),
            'Reference has no valid frozen execution ID')
    frozen = run_root / execution
    for target, files in plan.items():
        if target == run_root or target == root / 'models/training':
            for path in files:
                if frozen in path.parents or path.suffix.lower() in TEXT_SUFFIXES:
                    result.add(path)
        elif target == root / 'models/best_model_metadata.json':
            result.update(files)
        elif target.parent == root / 'reports':
            result.update(files)
    # Preserve provenance and current config even for reports-only cleanup.
    for name in ('configs', 'requirements.txt', 'models/reference/best_model_metadata.json',
                 'models/reference/evaluation'):
        path = root / name
        safe_path(root, path)
        if exists(path):
            result.update(p for p in walk_tree(path) if p.is_file())
    for name in KNOWN_POINTERS:
        pointer = root / name
        safe_path(root, pointer)
        if exists(pointer):
            result.add(pointer)
    # Archive the shortlist if present even when only reports are selected.
    safe_path(root, frozen)
    if exists(frozen):
        result.update(p for p in walk_tree(frozen) if p.is_file())
    return sorted(result)


def write_archive(root, destination, files, metadata):
    """Create an external ZIP exclusively, then verify each archived SHA-256."""
    destination = destination.resolve()
    require(destination.suffix.lower() == '.zip' and not destination.is_relative_to(root),
            'Evidence archive must be a ZIP outside the project directory')
    require(destination.parent.is_dir(), 'Archive parent must already exist')
    require(not exists(destination), f'Archive already exists: {destination}')
    manifest = {'schema_version': 1, 'git_commit': git_output(root, 'rev-parse', 'HEAD').decode().strip(),
                'model_sha256': metadata['model_sha256'], 'files': {}}
    # Exclusive mode prevents overwriting an earlier archive.
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            name = path.relative_to(root).as_posix()
            manifest['files'][name] = sha256_file(path)
            archive.write(path, name)
        archive.writestr('cleanup_manifest.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(destination) as archive:
        require(archive.testzip() is None, 'Archive CRC verification failed')
        for path in files:
            name = path.relative_to(root).as_posix()
            with archive.open(name) as stream:
                require(sha256_stream(stream) == manifest['files'][name] == sha256_file(path),
                        f'Archive/source checksum mismatch: {name}')
    return destination


def clean_project(categories, *, dry_run=False, yes=False, archive=None, root=ROOT):
    """Preview or clean the selected categories after validation and archiving."""
    root = Path(root).resolve()
    categories = set(categories)
    plan = build_plan(root, categories)
    print('Cleanup targets:')
    for path, files in plan.items():
        print(f'  {path.relative_to(root)} ({len(files)} files)')
    print(f'Total: {len(plan)} targets, {sum(map(len, plan.values()))} files')
    if not plan or dry_run:
        print('No files deleted.' if dry_run else 'Nothing to clean.')
        if dry_run and categories & EVIDENCE_CATEGORIES:
            print('Actual cleanup of experiments/reports requires a verified reference and --archive outside the project.')
        return []
    needs_archive = any(path.relative_to(root).as_posix() in CATEGORIES[c] for path in plan
                        for c in categories & EVIDENCE_CATEGORIES)
    if needs_archive:
        require(archive is not None, 'Use --archive PATH.zip before deleting experiments or reports')
        metadata = release_metadata(root)
        files = evidence_paths(root, plan, metadata)
    else:
        require(archive is None, '--archive requires existing experiment or report targets')
    if not yes and input('Delete the listed generated files? [y/N]: ').strip().lower() != 'y':
        print('Cleanup cancelled.')
        return []
    if needs_archive:
        destination = write_archive(root, Path(archive), files, metadata)
        print(f'Verified evidence archive: {destination}')
    # Recheck targets after confirmation and archive generation. Do not run concurrently.
    require(build_plan(root, categories) == plan, 'Cleanup targets changed; stop project writers and retry')
    removed = []
    for path in plan:
        safe_path(root, path)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(path.relative_to(root).as_posix())
    print(f'Cleanup completed: {len(removed)} targets removed.')
    return removed


def main():
    """Parse cleanup flags, require a category and report failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    for category in CATEGORIES:
        parser.add_argument(f'--{category}', action='store_true')
    parser.add_argument('--all', action='store_true', help='Select every cleanup category')
    parser.add_argument('--dry-run', action='store_true', help='Preview without modifying files')
    parser.add_argument('--yes', action='store_true', help='Skip interactive confirmation')
    parser.add_argument('--archive', type=Path, help='New external ZIP for experimental evidence')
    args = parser.parse_args()
    categories = set(CATEGORIES) if args.all else {c for c in CATEGORIES if getattr(args, c)}
    if not categories:
        parser.error('Select a cleanup category or --all; use --dry-run to preview')
    try:
        clean_project(categories, dry_run=args.dry_run, yes=args.yes, archive=args.archive)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
        parser.exit(1, f'Cleanup failed: {error}\n')


if __name__ == '__main__':
    main()
