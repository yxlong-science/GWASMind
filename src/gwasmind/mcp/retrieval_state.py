"""A single typed presentation of requested retrieval outcomes."""
from enum import StrEnum

from .contracts import ToolStatus


class RetrievalState(StrEnum):
    SUCCESS_DATA = "SUCCESS_DATA"
    SUCCESS_NO_DATA = "SUCCESS_NO_DATA"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    NOT_REQUESTED = "NOT_REQUESTED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


def retrieval_state(envelope):
    gaps = any(getattr(envelope.coverage, key) for key in ("failed", "unavailable", "blocked", "truncated"))
    if envelope.status in {ToolStatus.ERROR, ToolStatus.INVALID_INPUT, ToolStatus.BLOCKED, ToolStatus.UNAVAILABLE}:
        return RetrievalState.PARTIAL if envelope.records else RetrievalState.FAILED
    if envelope.status is ToolStatus.PARTIAL or gaps or envelope.next_cursor:
        return RetrievalState.PARTIAL
    if envelope.status in {ToolStatus.SUCCESS, ToolStatus.NO_DATA}:
        return RetrievalState.SUCCESS_DATA if envelope.records else RetrievalState.SUCCESS_NO_DATA
    return RetrievalState.FAILED
