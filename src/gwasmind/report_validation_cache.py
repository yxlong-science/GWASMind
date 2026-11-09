"""Call-local reuse of pure final-report derivations, never validation outcomes.

The native result is shallowly mutable. Reuse is therefore restricted to the
synchronous local phases of one run_hybrid_final call, with no external callback
allowed inside a phase. A model callback clears and suspends the memo. Callers
must not concurrently mutate the report's nested evidence during validation.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from threading import get_ident


def _shape(value):
    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        return type(value), id(value), len(value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return type(value), value
    return type(value), id(value)


def _context_shape(context):
    # Detect container replacement/resize and top-level result-field edits.
    # This is not a deep-mutation detector or a proof of deep immutability.
    return (tuple((name, _shape(getattr(context, name))) for name in (
        'result', 'source_observations', 'facts', 'sections', 'references',
        'safety_review', 'biological_context', 'intent', 'source_trait',
        'deterministic_answer')),
        tuple((key, _shape(value)) for key, value in context.result.items()))


class _ReportMemo:
    def __init__(self, context):
        self.context = context
        self.thread_id = get_ident()
        self.shape = None
        self.values = {}

    def clear(self):
        self.values.clear()
        self.shape = None


_CURRENT: ContextVar[_ReportMemo | None] = ContextVar('gwasmind_final_report_derivations', default=None)


@contextmanager
def report_validation_session(context):
    """Own one report's memo; nested calls and cancellation restore their parent."""
    memo = _ReportMemo(context)
    token = _CURRENT.set(memo)
    try:
        yield
    finally:
        memo.clear()
        _CURRENT.reset(token)


@contextmanager
def report_validation_callback():
    """External code may update context inputs; never reuse pre-callback values."""
    memo = _CURRENT.get()
    if memo is not None and memo.thread_id == get_ident():
        memo.clear()
    token = _CURRENT.set(None)
    try:
        yield
    finally:
        _CURRENT.reset(token)
        if memo is not None and memo.thread_id == get_ident():
            memo.clear()


def memoized_report_derivation(function):
    """Reuse a pure context/result derivation only in its owning report phase."""
    @wraps(function)
    def derive(owner, *args, **kwargs):
        memo = _CURRENT.get()
        if (memo is None or memo.thread_id != get_ident()
                or not (owner is memo.context or owner is memo.context.result)):
            return function(owner, *args, **kwargs)
        shape = _context_shape(memo.context)
        if shape != memo.shape:
            memo.clear()
            memo.shape = shape
        key = (function, id(owner), tuple(_shape(arg) for arg in args),
               tuple((name, _shape(value)) for name, value in kwargs.items()))
        if key not in memo.values:
            memo.values[key] = function(owner, *args, **kwargs)
        return memo.values[key]
    return derive
