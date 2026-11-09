"""Loopback-only local web interface for GWASMind."""

from typing import Any

__all__ = ["GWASMindHTTPServer", "JobManager", "create_server"]


def __getattr__(name: str) -> Any:
    if name == "JobManager":
        from .jobs import JobManager

        return JobManager
    if name in {"GWASMindHTTPServer", "create_server"}:
        from .server import GWASMindHTTPServer, create_server

        return {
            "GWASMindHTTPServer": GWASMindHTTPServer,
            "create_server": create_server,
        }[name]
    raise AttributeError(name)
