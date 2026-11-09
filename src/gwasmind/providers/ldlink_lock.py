"""One interruptible LDlink request slot shared by GWASMind processes.

The caller must keep the context open until the complete response is consumed.
This module never receives, names locks after, or persists an API credential.
"""
from __future__ import annotations

import errno
import math
import os
import tempfile
import time
from concurrent.futures import CancelledError
from contextlib import contextmanager
from pathlib import Path


_WINDOWS_MUTEX_NAME = r"Global\GWASMind.LDlink.Request.v1"
_POSIX_LOCK_NAME = "gwasmind-ldlink-request-v1.lock"
_POLL_SECONDS = 0.05


def _lock_error():
    error = RuntimeError("The shared LDlink request lock is unavailable.")
    error.reason_code = "ld_serialization_unavailable"
    return error


def _remaining(deadline, cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("Waiting for the shared LDlink request was cancelled.")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        error = TimeoutError("The LDlink request deadline expired while waiting for the shared request slot.")
        error.reason_code = "total_budget_exhausted"
        raise error
    return remaining


@contextmanager
def _windows_slot(deadline, cancel_event):
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.ReleaseMutex.argtypes = (wintypes.HANDLE,)
    kernel.ReleaseMutex.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateMutexW(None, False, _WINDOWS_MUTEX_NAME)
    if not handle:
        raise _lock_error()
    acquired = False
    try:
        while not acquired:
            remaining = _remaining(deadline, cancel_event)
            result = kernel.WaitForSingleObject(handle, max(1, math.ceil(min(_POLL_SECONDS, remaining) * 1000)))
            if result in {0x00000000, 0x00000080}:  # Acquired, or abandoned by a terminated owner.
                acquired = True
            elif result != 0x00000102:  # WAIT_TIMEOUT simply rechecks cancellation/deadline.
                raise _lock_error()
        _remaining(deadline, cancel_event)
        yield
    finally:
        try:
            if acquired and not kernel.ReleaseMutex(handle):
                raise _lock_error()
        finally:
            kernel.CloseHandle(handle)


@contextmanager
def _posix_slot(deadline, cancel_event):
    import fcntl

    # Never unlink this file: existing waiters must keep referring to one inode.
    path = Path(tempfile.gettempdir()) / _POSIX_LOCK_NAME
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    except OSError:
        raise _lock_error() from None
    acquired = False
    try:
        while not acquired:
            remaining = _remaining(deadline, cancel_event)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError as exc:
                if exc.errno == errno.EINTR:
                    continue
                if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise _lock_error() from None
                if cancel_event is not None:
                    cancel_event.wait(min(_POLL_SECONDS, remaining))
                else:
                    time.sleep(min(_POLL_SECONDS, remaining))
        _remaining(deadline, cancel_event)
        yield
    finally:
        try:
            if acquired:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError:
                    raise _lock_error() from None
        finally:
            os.close(descriptor)


@contextmanager
def serial_ldlink_request(*, deadline, cancel_event=None):
    """Serialize a complete request using an absolute monotonic deadline.

    Cancellation and the deadline are checked while waiting and immediately
    before entering. Network timeout/response consumption remain the caller's
    responsibility inside the context. OS ownership is released on process exit.
    """
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
        raise ValueError("LDlink request deadline must be a finite monotonic timestamp.")
    _remaining(deadline, cancel_event)
    if os.name == "nt":
        slot = _windows_slot
    elif os.name == "posix":
        slot = _posix_slot
    else:
        raise _lock_error()
    with slot(deadline, cancel_event):
        yield
