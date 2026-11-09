"""Explicit LD backend selection; credentials never enter scientific configuration."""
from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from ..configuration import parameter
from .ld import LDReferenceProvider


class LDProviderConfigurationError(RuntimeError):
    reason_code = "ld_credential_unavailable"


def ld_configuration_issue(*, scope: str) -> dict[str, Any]:
    """A missing optional LD credential is operational, not invalid source data."""
    activity = "Optional LD expansion was not performed; original source variants remain available." if scope == "ld_expansion" else (
        "The LD-based prior-report search could not be completed; novelty remains unresolved.")
    return {"code": "LD_CONFIGURATION_UNAVAILABLE", "category": "configuration", "scope": scope,
            "blocking": False, "reason_codes": [LDProviderConfigurationError.reason_code],
            "message": "LDlink is selected, but its API credential is unavailable. " + activity}


def _provider_name(value: str | None) -> str:
    selected = parameter("ld.provider") if value is None else value
    if selected not in {"ldlink", "ensembl"}:
        raise ValueError("LD provider must be explicitly selected as ldlink or ensembl.")
    return selected


def create_ld_provider(*, provider_name: str | None = None,
                       token_getter: Callable[[], str | None] | None = None, **kwargs: Any):
    """Create the configured provider; missing LDlink credentials never change it."""
    selected = _provider_name(provider_name)
    if selected == "ensembl":
        return LDReferenceProvider(**kwargs)
    if token_getter is None:
        from ..credentials import get_ldlink_token
        token_getter = get_ldlink_token
    try:
        token = token_getter()
    except Exception:
        raise LDProviderConfigurationError(
            "LDlink is selected but its API credential could not be loaded. Configure LDlink credentials before requesting LD."
        ) from None
    if not isinstance(token, str) or not token.strip():
        raise LDProviderConfigurationError(
            "LDlink is selected but its API credential is not configured. Configure LDlink credentials before requesting LD."
        )
    from .ldlink import LDLinkReferenceProvider
    return LDLinkReferenceProvider(token=token, **kwargs)


def ld_provider_metadata(provider=None) -> dict[str, str]:
    """Public backend identity, never credentials or transport instance state.

    Without an instance this describes the selected reference configuration.
    Explicit fixture backends retain their own identity instead of being labelled
    as the configured live service.
    """
    if provider is None:
        name = _provider_name(None)
        version = "1.0.0" if name == "ldlink" else "ensembl-rest-v1"
    else:
        supplied = getattr(provider, "provider_name", None)
        name = supplied.casefold() if isinstance(supplied, str) else (
            "ensembl" if isinstance(provider, LDReferenceProvider) else "injected")
        if name not in {"ldlink", "ensembl", "injected"}:
            name = "injected"
        version = getattr(provider, "provider_version", "ensembl-rest-v1" if name == "ensembl" else "unversioned")
    if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", version):
        version = "unversioned"
    return {"reference_provider": name, "provider_version": version}
