"""Provide a local single-writer guard for persistent Optuna studies."""

from contextlib import contextmanager
from hashlib import sha256
import os
from pathlib import Path


@contextmanager
def exclusive_study_lock(database: Path, study_name: str):
    """Prevent concurrent local writers to one study using an exclusive lockfile.

    An interrupted process may leave a lock. Remove it manually only after
    verifying that the original process is no longer running and inspecting
    any RUNNING trials. This is not a distributed locking mechanism.
    """
    key = sha256(study_name.encode("utf-8")).hexdigest()
    path = database.with_name(f"{database.name}.{key}.lock")
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Study is locked: {path}. Verify the existing process before removing the lock.") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as lock:
            lock.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)
