"""User-authorized local model configuration protected by Windows DPAPI."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from uuid import uuid4

from .credentials import CredentialError, _dpapi

_FIELDS = {"api_key", "base_url", "model_id", "thinking_mode", "reasoning_effort", "model_provider"}


def model_configuration_path() -> Path:
    return Path(__file__).resolve().parents[2] / ".gwasmind/credentials/model.dpapi"


def _configuration(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != _FIELDS:
        raise CredentialError("The saved model configuration has invalid fields.")
    if any(not isinstance(item, str) or not item or len(item) > 4096 for item in value.values()):
        raise CredentialError("The saved model configuration has invalid values.")
    if any(character.isspace() for character in value["api_key"]):
        raise CredentialError("The saved model credential is invalid.")
    if value["thinking_mode"] not in {"enabled", "disabled", "auto"}:
        raise CredentialError("The saved thinking mode is invalid.")
    if not value["base_url"].startswith(("https://", "http://127.0.0.1", "http://localhost")):
        raise CredentialError("The saved model endpoint is invalid.")
    return dict(value)


def save_model_configuration(value: dict[str, str], path: Path | None = None) -> None:
    payload = json.dumps(_configuration(value), sort_keys=True, separators=(",", ":")).encode()
    try:
        cipher = _dpapi(payload, decrypt=False)
    except CredentialError:
        raise CredentialError("The model configuration could not be encrypted.") from None
    target = path or model_configuration_path()
    temporary = target.with_name(target.name + "." + uuid4().hex + ".tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps({"format": "gwasmind-model-dpapi-v1",
            "encrypted_configuration": base64.b64encode(cipher).decode("ascii")}), encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def load_model_configuration(path: Path | None = None) -> dict[str, str]:
    target = path or model_configuration_path()
    try:
        if target.stat().st_size > 32768:
            raise ValueError()
        saved = json.loads(target.read_text(encoding="utf-8"))
        if saved.get("format") != "gwasmind-model-dpapi-v1":
            raise ValueError()
        cipher = base64.b64decode(saved["encrypted_configuration"], validate=True)
        return _configuration(json.loads(_dpapi(cipher, decrypt=True)))
    except (OSError, ValueError, KeyError, TypeError, AttributeError, CredentialError):
        raise CredentialError("The saved model configuration could not be opened by this Windows user.") from None
