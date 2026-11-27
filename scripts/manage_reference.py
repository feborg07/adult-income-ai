"""Bundle current reference evidence and verify published releases before backup removal.

Run from the project environment with Git and DVC installed. Never deletes S3 objects.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager

from src.models import promote as p

ROOT = p.ROOT
POINTER = 'models/reference.dvc'
TAG = re.compile(r'model-v\d+\.\d+\.\d+')


@contextmanager
def release_lock(root):
    """Share promotion's lock; do not operate during an unfinished transaction."""
    history = root / 'models/reference_history'
    history.mkdir(parents=True, exist_ok=True)
    p.require(not history.is_symlink() and not getattr(history, 'is_junction', lambda: False)(),
              'Unsafe rollback history directory')
    lock = history / '.promotion.lock'
    lock.mkdir()
    try:
        yield history
    finally:
        lock.rmdir()


def reference_metadata(directory, *, require_bundle=False):
    """Verify model integrity and any bundled evidence without deserializing it."""
    p.require(directory.is_dir() and not directory.is_symlink()
              and not getattr(directory, 'is_junction', lambda: False)(), 'Unsafe reference directory')
    for name in (p.MODEL_FILENAME, p.METADATA_FILENAME):
        path = directory / name
        p.require(path.is_file() and not path.is_symlink(), f'Missing or unsafe reference file: {name}')
    metadata = p.load_json(directory / p.METADATA_FILENAME)
    p.verify_hash(directory / p.MODEL_FILENAME, metadata['model_sha256'])
    if require_bundle or (directory / 'evaluation').exists():
        p.verify_evaluation_bundle(directory / 'evaluation', metadata)
    return metadata


def bundle_current(root=ROOT):
    """Add the evidence directory atomically, leaving existing DVC files unchanged."""
    root = Path(root)
    reference = root / 'models/reference'
    with release_lock(root) as history:
        metadata = reference_metadata(reference)
        target = reference / 'evaluation'
        if target.exists():
            return {'bundled': True, 'created': False, 'model_sha256': metadata['model_sha256']}
        # Stage on the same filesystem; never edit DVC-linked model or metadata.
        with tempfile.TemporaryDirectory(prefix='bundle-', dir=history) as tmp:
            staged = Path(tmp) / 'evaluation'
            p.write_evaluation_bundle(staged, metadata)
            p.require(reference_metadata(reference) == metadata, 'Reference changed during bundling')
            os.rename(staged, target)
        p.verify_evaluation_bundle(target, metadata)
    return {'bundled': True, 'created': True, 'model_sha256': metadata['model_sha256']}


def git(root, *args):
    """Execute Git in the specified checkout and return its stdout bytes."""
    return subprocess.run(['git', '-C', str(root), *args], check=True, stdout=subprocess.PIPE).stdout


def tree_hashes(directory):
    """Fingerprint every regular file; reject links and unexpected file types."""
    p.require(directory.is_dir() and not directory.is_symlink()
              and not getattr(directory, 'is_junction', lambda: False)(), 'Unsafe reference tree')
    hashes = {}
    for path in sorted(directory.rglob('*')):
        is_junction = getattr(path, 'is_junction', lambda: False)
        p.require(not path.is_symlink() and not is_junction(), f'Unsafe reference entry: {path}')
        if path.is_file():
            hashes[path.relative_to(directory).as_posix()] = p.sha256_file(path)
        else:
            p.require(path.is_dir(), f'Unexpected reference entry: {path}')
    return hashes


def normalize_pointer(data):
    """Ignore Git's Windows checkout line-ending conversion, not pointer content."""
    return data.replace(b'\r\n', b'\n')


def recover_tag(root, tag, expected_reference, expected_pointer, *, require_bundle=False):
    """Pull the exact published tag into an isolated checkout with a new DVC cache."""
    p.require(TAG.fullmatch(tag) is not None, f'Invalid release tag: {tag}')
    origin = git(root, 'remote', 'get-url', 'origin').decode().strip()
    with tempfile.TemporaryDirectory(prefix='reference-recovery-') as tmp:
        clone = Path(tmp) / 'checkout'
        # Clone origin, not the local repo or its DVC cache.
        subprocess.run(['git', 'clone', '--branch', tag, '--single-branch', '--depth', '1',
                        '--', origin, str(clone)], check=True, cwd=root)
        p.require(git(clone, 'rev-parse', f'refs/tags/{tag}^{{commit}}') == git(clone, 'rev-parse', 'HEAD'),
                  f'Published tag mismatch: {tag}')
        p.require(normalize_pointer((clone / POINTER).read_bytes()) == normalize_pointer(expected_pointer), f'DVC pointer mismatch: {tag}')
        # Override any shared/global cache so recovery actually fetches from the remote.
        cache = Path(tmp) / 'empty-cache'
        dvc = [sys.executable, '-m', 'dvc']
        subprocess.run([*dvc, 'config', '--local', 'cache.dir', str(cache)], cwd=clone, check=True)
        subprocess.run([*dvc, 'config', '--local', 'cache.type', 'copy'], cwd=clone, check=True)
        subprocess.run([*dvc, 'pull', '--remote', 'adult-income-ai', POINTER], cwd=clone, check=True)
        recovered = clone / 'models/reference'
        reference_metadata(recovered, require_bundle=require_bundle)
        p.require(tree_hashes(recovered) == expected_reference, f'Recovered reference differs: {tag}')
        return git(clone, 'rev-parse', 'HEAD').decode().strip()


def verify_releases(current_tag, backup_tags, *, remove_backups=False, root=ROOT):
    """Verify the active and every rollback release before optionally deleting backups."""
    root = Path(root)
    reference = root / 'models/reference'
    with release_lock(root) as history:
        metadata = reference_metadata(reference, require_bundle=True)
        current_tree = tree_hashes(reference)
        pointer = (root / POINTER).read_bytes()
        p.require(normalize_pointer(git(root, 'show', f'HEAD:{POINTER}')) == normalize_pointer(pointer), 'Commit the current DVC pointer first')
        transactions = {}
        # Unknown or interrupted transaction directories must be inspected manually.
        for path in history.iterdir():
            if path.name == '.promotion.lock':
                continue
            p.require(re.fullmatch(r'[0-9a-f]{32}', path.name) is not None and path.is_dir()
                      and not path.is_symlink() and not getattr(path, 'is_junction', lambda: False)(),
                      f'Unexpected rollback entry: {path}')
            p.require({entry.name for entry in path.iterdir()} == {'reference', 'reference.dvc'},
                      f'Incomplete rollback transaction: {path}')
            p.require((path / 'reference.dvc').is_file() and not (path / 'reference.dvc').is_symlink(),
                      f'Unsafe rollback pointer: {path}')
            reference_metadata(path / 'reference')
            transactions[path.name] = (path, tree_hashes(path / 'reference'), (path / 'reference.dvc').read_bytes())
        p.require(set(backup_tags) == set(transactions), 'Provide exactly one published tag for every rollback transaction')
        p.require(len({current_tag, *backup_tags.values()}) == 1 + len(backup_tags), 'Release tags must be distinct')
        recovered = {current_tag: recover_tag(root, current_tag, current_tree, pointer, require_bundle=True)}
        for name, (path, hashes, old_pointer) in transactions.items():
            tag = backup_tags[name]
            recovered[tag] = recover_tag(root, tag, hashes, old_pointer)
        # Check all local bytes again before removing anything. No S3 deletion.
        p.require(tree_hashes(reference) == current_tree and (root / POINTER).read_bytes() == pointer,
                  'Active reference changed during recovery verification')
        for path, hashes, old_pointer in transactions.values():
            p.require(tree_hashes(path / 'reference') == hashes and (path / 'reference.dvc').read_bytes() == old_pointer,
                      'Rollback backup changed during recovery verification')
        if remove_backups:
            for path, _, _ in transactions.values():
                shutil.rmtree(path)
        return dict(verified=True, model_sha256=metadata['model_sha256'], releases=recovered,
                    backups_removed=list(transactions) if remove_backups else [])


def parse_backup_tags(values):
    """Parse explicit transaction-to-tag mappings, rejecting duplicate IDs."""
    mappings = {}
    for value in values:
        name, sep, tag = value.partition('=')
        p.require(sep and re.fullmatch(r'[0-9a-f]{32}', name) and TAG.fullmatch(tag) and name not in mappings,
                  f'Invalid or duplicate backup mapping: {value}')
        mappings[name] = tag
    return mappings


def main():
    """Expose evidence migration and safe verification/finalization commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('bundle', help='Bundle already approved evidence without fitting')
    verify = sub.add_parser('verify', help='Verify published tags with fresh DVC downloads')
    verify.add_argument('--current-tag', required=True)
    verify.add_argument('--backup-tag', action='append', default=[], metavar='TRANSACTION=TAG')
    verify.add_argument('--remove-backups', action='store_true', help='Remove verified local backups only')
    args = parser.parse_args()
    try:
        if args.action == 'bundle':
            result = bundle_current()
        else:
            result = verify_releases(args.current_tag, parse_backup_tags(args.backup_tag),
                                     remove_backups=args.remove_backups)
    except (ValueError, KeyError, OSError, TypeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'Reference release operation failed: {exc}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
