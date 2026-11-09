"""Bind a supervised CLI child to its selected source package before importing it.

This file is executed by absolute path, so importing the bootstrap itself does
not depend on an editable installation, the working directory, or PYTHONPATH.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import runpy
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path


def _verify_loaded_sources(package: Path) -> None:
    for name, module in tuple(sys.modules.items()):
        if module is None or not (name == "gwasmind" or name.startswith("gwasmind.")):
            continue
        origin = getattr(module, "__file__", None)
        if not origin or not Path(origin).resolve().is_relative_to(package):
            raise RuntimeError(f"The worker resolved {name} outside its selected source package.")


def bind_worker_package(package_directory: str | Path) -> Path:
    """Require the selected package and CLI; never fall back to another checkout."""
    package = Path(package_directory).resolve()
    expected = {"gwasmind": package / "__init__.py", "gwasmind.cli": package / "cli.py"}
    if package.name != "gwasmind" or not all(path.is_file() for path in expected.values()):
        raise RuntimeError(f"The selected GWASMind worker source is incomplete: {package}")
    _verify_loaded_sources(package)
    # PYTHONPATH alone does not outrank cwd for -m, or site customizations that
    # prepend another checkout. Bind again inside the actual child interpreter.
    sys.path.insert(0, str(package.parent))
    importlib.invalidate_caches()
    for name, path in expected.items():
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.origin or Path(spec.origin).resolve() != path.resolve():
            raise RuntimeError(f"The worker resolved {name} outside its selected source package.")
    _verify_loaded_sources(package)
    return package


def _write_source_receipt(destination: str, package: Path) -> None:
    """Persist only code origins and hashes, never environment or credentials."""
    def source(path: Path) -> dict[str, str]:
        path = path.resolve()
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    value = {
        "schema_version": 1,
        "verified_at": datetime.now(UTC).isoformat(),
        "verification": "package and CLI import origins verified before execution",
        "selected_package": str(package),
        "controller_bootstrap": source(Path(__file__)),
        "modules": {"gwasmind": source(package / "__init__.py"), "gwasmind.cli": source(package / "cli.py")},
    }
    target = Path(destination)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         prefix=target.name, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, ensure_ascii=False)
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    if len(sys.argv) < 2:
        print("A selected GWASMind worker source package is required.", file=sys.stderr)
        return 2
    package = sys.argv.pop(1)
    receipt = None
    if len(sys.argv) > 1 and sys.argv[1] == "--source-binding-receipt":
        if len(sys.argv) < 3:
            print("A worker source receipt path is required.", file=sys.stderr)
            return 2
        del sys.argv[1]
        receipt = sys.argv.pop(1)
    try:
        selected = bind_worker_package(package)
        if receipt is not None:
            _write_source_receipt(receipt, selected)
    except (OSError, ImportError, ValueError, RuntimeError) as exc:
        print(f"GWASMind worker source binding failed: {exc}", file=sys.stderr)
        return 2
    # Remaining argv contains exactly the ordinary CLI arguments. run_module
    # supplies normal __main__/argv[0] semantics without importing another copy.
    runpy.run_module("gwasmind.cli", run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
