"""Optional short-lived LD response index with immutable raw JSON objects.

Only successful public responses persist across calls. Index entries bind the
exact request and reference configuration; raw objects remain content-addressed
so an older run's receipt stays verifiable after the index is refreshed.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
import json
import os
from pathlib import Path
import threading
import time
from uuid import uuid4
import weakref
from urllib.parse import parse_qs, urlsplit

from ..configuration import parameter

_LOCKS = weakref.WeakValueDictionary()
_LOCK_GUARD = threading.Lock()


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class LDResponseCache:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        (self.directory / "responses").mkdir(parents=True, exist_ok=True)
        (self.directory / "requests").mkdir(exist_ok=True)
        (self.directory / "variations").mkdir(exist_ok=True)
        self._responses = OrderedDict()
        self._response_lock = threading.RLock()

    @staticmethod
    def request(url, body):
        return {"url": url, "method": "POST" if body is not None else "GET", "body": body,
                "reference_build": parameter("novelty.reference_build"),
                "reference_panel": parameter("novelty.reference_panel"), "release": "live"}

    @staticmethod
    def key(request):
        return hashlib.sha256(encode(request)).hexdigest()

    def lock(self, key):
        identity = (str(self.directory), key)
        with _LOCK_GUARD:
            lock = _LOCKS.get(identity)
            if lock is None:
                lock = threading.RLock()
                _LOCKS[identity] = lock
            return lock

    @staticmethod
    def _write(path, raw):
        temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
        try:
            temporary.write_bytes(raw)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _read(path, limit):
        with path.open("rb") as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("Saved LD response exceeds its bounded envelope")
        return raw

    def get(self, request):
        key = self.key(request)
        path = self.directory / "requests" / (key + ".json")
        if not path.exists():
            return None
        metadata = json.loads(self._read(path, parameter("transport.graphql_max_request_bytes")))
        if metadata.get("schema") != 1 or metadata.get("request") != request or metadata.get("request_key") != key:
            raise ValueError("Saved LD request identity is invalid")
        if metadata.get("reusable") is False:
            return None
        result = self._response(metadata)
        if result is None:
            return None
        value, receipt = result
        path = urlsplit(request["url"]).path
        if request["method"] == "POST" and path.endswith("/variation/human"):
            if not isinstance(value, dict) or not all(isinstance(value.get(rsid), dict) and value[rsid].get("name") == rsid
                for rsid in (request.get("body") or {}).get("ids", [])):
                return None
        elif request["method"] == "GET" and "/variation/human/" in path:
            identifier = path.rsplit("/", 1)[-1]
            if not isinstance(value, dict) or value.get("name", identifier) != identifier:
                return None
        return value, receipt

    def _response(self, metadata):
        stored_at = metadata.get("stored_at")
        if not isinstance(stored_at, (int, float)) or isinstance(stored_at, bool):
            raise ValueError("Saved LD response date is invalid")
        age = time.time() - stored_at
        if not 0 <= age <= parameter("retrieval.ld_cache_ttl_seconds"):
            return None
        digest = metadata.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Saved LD response digest is invalid")
        raw_path = self.directory / "responses" / (digest + ".json")
        with self._response_lock:
            loaded = self._responses.get(digest)
            if loaded is None:
                raw = self._read(raw_path, parameter("transport.graphql_max_response_bytes"))
                if hashlib.sha256(raw).hexdigest() != digest or len(raw) != metadata.get("bytes"):
                    raise ValueError("Saved LD response content failed verification")
                value = json.loads(raw)
                loaded = (value, len(raw))
                self._responses[digest] = loaded
                while len(self._responses) > parameter("retrieval.entity_cache_memory_items"):
                    self._responses.popitem(last=False)
            self._responses.move_to_end(digest)
            value, byte_count = loaded
            if byte_count != metadata.get("bytes"):
                raise ValueError("Saved LD response length is inconsistent")
        return value, {**metadata, "response_path": str(raw_path)}

    @staticmethod
    def _part(endpoint, rsid, populations):
        return {"endpoint": endpoint, "rsid": rsid, "populations_requested": populations,
                "reference_build": parameter("novelty.reference_build"),
                "reference_panel": parameter("novelty.reference_panel"), "release": "live"}

    def get_variation(self, endpoint, rsid, *, populations=True):
        part = self._part(endpoint, rsid, populations)
        path = self.directory / "variations" / (self.key(part) + ".json")
        if not path.exists():
            return None
        value = json.loads(self._read(path, parameter("transport.graphql_max_request_bytes")))
        metadata = value.get("parent")
        parent = metadata.get("request") if isinstance(metadata, dict) else None
        if (value.get("part") != part or not isinstance(metadata, dict) or
                not isinstance(parent, dict) or metadata.get("schema") != 1 or self.key(parent) != metadata.get("request_key") or
                parent.get("method") != "POST" or not parent.get("url", "").startswith(endpoint + "/variation/human?") or
                (parse_qs(urlsplit(parent["url"]).query).get("pops") == ["1"]) is not populations or
                any(parent.get(key) != part[key] for key in ("reference_build", "reference_panel", "release")) or
                rsid not in (parent.get("body") or {}).get("ids", [])):
            raise ValueError("Saved LD variant-to-parent binding is invalid")
        response = self._response(metadata)
        if response is None:
            return None
        raw, receipt = response
        if not isinstance(raw, dict) or not isinstance(raw.get(rsid), dict) or raw[rsid].get("name") != rsid:
            raise ValueError("Saved LD parent response omitted its bound variant")
        return raw[rsid], receipt

    def save(self, request, raw, *, retrieved_at, reusable=True):
        if type(reusable) is not bool:
            raise ValueError("LD response reuse flag must be boolean")
        digest = hashlib.sha256(raw).hexdigest()
        key = self.key(request)
        raw_path = self.directory / "responses" / (digest + ".json")
        if not raw_path.exists() or hashlib.sha256(self._read(raw_path, parameter("transport.graphql_max_response_bytes"))).hexdigest() != digest:
            self._write(raw_path, raw)
        metadata = {"schema": 1, "request": request, "request_key": key,
                    "sha256": digest, "bytes": len(raw), "retrieved_at": retrieved_at, "stored_at": time.time()}
        parsed = urlsplit(request["url"])
        if request["method"] == "POST" and parsed.path.endswith("/variation/human"):
            endpoint = request["url"].split("/variation/human?", 1)[0]
            populations = parse_qs(parsed.query).get("pops") == ["1"]
            response = json.loads(raw)
            metadata["reusable"] = isinstance(response, dict) and all(
                isinstance(response.get(rsid), dict) and response[rsid].get("name") == rsid
                for rsid in (request.get("body") or {}).get("ids", []))
            if isinstance(response, dict):
                for rsid in (request.get("body") or {}).get("ids", []):
                    if not isinstance(response.get(rsid), dict) or response[rsid].get("name") != rsid:
                        continue
                    part = self._part(endpoint, rsid, populations)
                    self._write(self.directory / "variations" / (self.key(part) + ".json"),
                                encode({"part": part, "parent": metadata}))
        elif request["method"] == "GET" and "/variation/human/" in parsed.path:
            response = json.loads(raw)
            identifier = parsed.path.rsplit("/", 1)[-1]
            metadata["reusable"] = isinstance(response, dict) and response.get("name", identifier) == identifier
        if not reusable:
            metadata["reusable"] = False
        self._write(self.directory / "requests" / (key + ".json"), encode(metadata))
        return {**metadata, "response_path": str(raw_path)}
