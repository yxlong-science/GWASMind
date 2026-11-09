"""Bounded, cancellable OS file locks shared by application instances."""
from __future__ import annotations

import errno
import math
import os
import time
from concurrent.futures import CancelledError
from contextlib import contextmanager
from pathlib import Path


class FileLockTimeout(TimeoutError):
    """The lock could not be acquired within the caller's bounded allowance."""


@contextmanager
def exclusive_file_lock(path, timeout_seconds, cancel_event=None):
    """Lock one persistent file; OS ownership releases even when its process exits.

    Never delete these lock files: another process may already have opened the
    same inode. The lock protects the caller's entire context, not only writes.
    """
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("File lock timeout must be a finite positive number.")

    def check():
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError("Waiting for the shared file lock was cancelled.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise FileLockTimeout("The shared file lock acquisition timed out.")
        return remaining

    deadline = time.monotonic() + timeout_seconds
    check()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    acquired = False
    try:
        # Windows can lock byte ranges beyond EOF. Do not initialise an empty
        # file before locking: a second opener may already own that byte.
        while not acquired:
            remaining = check()
            try:
                if os.name == "nt":
                    import msvcrt
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                delay = min(0.05, remaining)
                if cancel_event is not None:
                    cancel_event.wait(delay)
                else:
                    time.sleep(delay)
        check()
        yield
    finally:
        try:
            if acquired:
                if os.name == "nt":
                    import msvcrt
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
