"""Local credentials, kept outside scientific configuration and run artifacts."""
from __future__ import annotations

import base64
import ctypes
import json
import os
from pathlib import Path
from uuid import uuid4


class CredentialError(RuntimeError):
    """A credential cannot be used; messages never include its contents."""


def _token(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise CredentialError("The LDlink credential is empty or invalid.")
    value = value.strip()
    if any(character.isspace() or ord(character) < 32 for character in value):
        raise CredentialError("The LDlink credential contains invalid characters.")
    return value


def _dpapi(value: bytes, *, decrypt: bool) -> bytes:
    if os.name != "nt":
        raise CredentialError("This encrypted credential requires its Windows user account; configure LDLINK_TOKEN on other systems.")
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]

    buffer = ctypes.create_string_buffer(value)
    source = Blob(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    target = Blob()
    library = ctypes.WinDLL("crypt32", use_last_error=True)
    function = library.CryptUnprotectData if decrypt else library.CryptProtectData
    function.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.POINTER(Blob),
                         ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    function.restype = wintypes.BOOL
    if not function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(target)):
        raise CredentialError("The local LDlink credential could not be protected or opened.")
    try:
        return ctypes.string_at(target.data, target.size)
    finally:
        release = ctypes.WinDLL("kernel32", use_last_error=True).LocalFree
        release.argtypes = [ctypes.c_void_p]
        release.restype = ctypes.c_void_p
        release(target.data)


def ldlink_credential_path() -> Path:
    """Resolve the local store independently of the launcher's working directory."""
    root = Path(__file__).resolve().parents[2]
    explicit = os.environ.get("GWASMIND_LDLINK_CREDENTIAL_FILE")
    path = Path(explicit).expanduser() if explicit else root / ".gwasmind/credentials/ldlink.dpapi"
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def save_ldlink_token(value: str, path: Path | None = None) -> None:
    """Protect a user-supplied key with user-scoped Windows DPAPI."""
    protected = _dpapi(_token(value).encode("utf-8"), decrypt=False)
    document = {"format": "gwasmind-windows-dpapi-v1",
                "encrypted_token": base64.b64encode(protected).decode("ascii")}
    path = Path(path) if path is not None else ldlink_credential_path()
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(document), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        raise CredentialError("The encrypted LDlink credential could not be saved.") from None
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _default_credential_paths() -> list[Path]:
    return [ldlink_credential_path()]


def get_ldlink_token() -> str | None:
    for name in ("GWASMIND_LDLINK_TOKEN", "LDLINK_TOKEN"):
        value = os.environ.get(name)
        if value is not None:
            return _token(value)
    explicit = os.environ.get("GWASMIND_LDLINK_CREDENTIAL_FILE")
    paths = [ldlink_credential_path()] if explicit else _default_credential_paths()
    for path in paths:
        if not path.is_file():
            if explicit:
                raise CredentialError("The configured LDlink credential file is unavailable.")
            continue
        try:
            if path.stat().st_size > 32768:
                raise ValueError()
            document = json.loads(path.read_text(encoding="utf-8"))
            if document.get("format") != "gwasmind-windows-dpapi-v1":
                raise ValueError()
            protected = base64.b64decode(document["encrypted_token"], validate=True)
            return _token(_dpapi(protected, decrypt=True).decode("utf-8"))
        except (OSError, ValueError, KeyError, TypeError, AttributeError, CredentialError):
            raise CredentialError("The configured LDlink credential could not be opened.") from None
    return None
