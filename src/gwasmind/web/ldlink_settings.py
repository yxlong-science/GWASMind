"""Local LDlink credential settings, independent of the model configuration."""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlencode

from .. import credentials
from ..providers.ldlink import LDLinkReferenceProvider, _check_error
from ..providers.ldlink_lock import serial_ldlink_request

_TOKEN_ENVIRONMENTS = ("GWASMIND_LDLINK_TOKEN", "LDLINK_TOKEN")
_TEST_VARIANTS = ("rs6983267", "rs10505477")
_TEST_SECONDS = 12.0


def _fingerprint(token):
    # Only an in-memory comparison; never returned or written to disk.
    return hashlib.sha256(token.encode()).digest()


def _validate_connection(token: str, *, fetch=None) -> None:
    """One fixed public LDpair request; no selected/user variants or model call."""
    provider = LDLinkReferenceProvider(token=token, ldlink_fetch=fetch, total_timeout_seconds=_TEST_SECONDS)
    params = {"var1": _TEST_VARIANTS[0], "var2": _TEST_VARIANTS[1],
              "pop": "EUR", "genome_build": "grch38", "json_out": "true"}
    # Use the provider's authentication/redaction/no-redirect boundary and the
    # same cross-process slot as analyses, without cache or progress artifacts.
    with serial_ldlink_request(deadline=provider.deadline):
        text, _ = provider._transport(provider.ldlink + "/ldpair?" + urlencode(params),
                                      timeout=provider._remaining())
        provider._remaining()
    value = json.loads(text)
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, dict):
        raise ValueError("Invalid diagnostic response")
    _check_error(value)
    for side, expected in zip(("snp1", "snp2"), _TEST_VARIANTS, strict=True):
        if not isinstance(value.get(side), dict) or value[side].get("rsnum") != expected:
            raise ValueError("Invalid diagnostic identity")
    stats = value.get("statistics")
    if not isinstance(stats, dict):
        raise ValueError("Invalid diagnostic statistics")
    for name in ("r2", "d_prime"):
        raw = stats.get(name)
        number = float(raw)
        if isinstance(raw, bool) or not math.isfinite(number) or not 0 <= number <= 1:
            raise ValueError("Invalid diagnostic statistics")


class LDLinkSettings:
    """DPAPI persistence with verification valid only for the current credential."""

    def __init__(self, *, validator=None):
        self._validator = validator or _validate_connection
        self._verified = None
        self._lock = threading.RLock()
        self._test_lock = threading.Lock()

    @staticmethod
    def _environment_managed():
        return any(name in os.environ for name in _TOKEN_ENVIRONMENTS)

    def status(self):
        environment = self._environment_managed()
        try:
            token = credentials.get_ldlink_token()
        except credentials.CredentialError:
            if not environment and not credentials.ldlink_credential_path().exists():
                return {"configured": False, "verified": False, "credential_state": "missing",
                        "storage": "windows_dpapi", "editable": os.name == "nt", "reason_code": None}
            return {"configured": False, "verified": False,
                    "credential_state": "invalid_environment" if environment else "unreadable",
                    "storage": "environment" if environment else "windows_dpapi",
                    "editable": not environment and os.name == "nt",
                    "reason_code": "ld_credential_unreadable"}
        with self._lock:
            verified = bool(token and self._verified == _fingerprint(token))
        return {"configured": bool(token), "verified": verified,
                "credential_state": "available" if token else "missing",
                "storage": "environment" if environment else "windows_dpapi",
                "editable": not environment and os.name == "nt", "reason_code": None}

    def _require_editable(self):
        if self._environment_managed():
            raise credentials.CredentialError("LDlink is managed by the launch environment; its environment credential takes precedence.")
        if os.name != "nt":
            raise credentials.CredentialError("Encrypted LDlink settings require Windows; configure LDLINK_TOKEN on this system.")

    def save(self, api_key):
        self._require_editable()
        with self._lock:
            credentials.save_ldlink_token(api_key)
            self._verified = None
        return {"success": True, "response": "LDlink key saved with Windows encryption. Test the connection to verify access.",
                "ldlink": self.status()}

    def clear(self):
        self._require_editable()
        with self._lock:
            try:
                credentials.ldlink_credential_path().unlink(missing_ok=True)
            except OSError:
                raise credentials.CredentialError("The encrypted LDlink credential could not be removed.") from None
            self._verified = None
        return {"success": True, "response": "Saved LDlink key removed.", "ldlink": self.status()}

    def test(self, api_key=None):
        if not self._test_lock.acquire(blocking=False):
            return {"success": False, "response": "An LDlink connection test is already running.",
                    "reason_code": "ld_test_busy", "ldlink": self.status()}
        try:
            if api_key is not None:
                self.save(api_key)
            token = credentials.get_ldlink_token()
            if not token:
                raise credentials.CredentialError("An LDlink key is required before testing the connection.")
            started = time.monotonic()
            with self._lock:
                self._verified = None
            try:
                self._validator(token)
            except Exception as error:
                reason = getattr(error, "reason_code", None)
                code = error.code if isinstance(error, HTTPError) else None
                reason = ("authentication_failed" if reason == "authentication_failed" or code in {401, 403}
                          else "rate_limited" if reason == "rate_limited" or code == 429
                          else "request_timeout" if isinstance(error, TimeoutError) or reason == "total_budget_exhausted"
                          else "connection_unverified")
                messages = {"authentication_failed": "LDlink did not accept this API key.",
                            "rate_limited": "LDlink is busy or rate limited. The saved key was kept; retry later.",
                            "request_timeout": "The LDlink connection test timed out. The saved key was kept; retry later.",
                            "connection_unverified": "LDlink access could not be verified. The saved key was kept; retry later."}
                # Provider bodies, URLs and exception text may contain secrets.
                return {"success": False, "response": messages[reason], "reason_code": reason,
                        "ldlink": self.status()}
            with self._lock:
                self._verified = _fingerprint(token)
            status = self.status()
            return {"success": status["verified"], "response": "LDlink connection verified." if status["verified"] else
                    "The credential changed during the test. Test the current key again.",
                    "latency_ms": round((time.monotonic() - started) * 1000), "ldlink": status}
        finally:
            self._test_lock.release()
