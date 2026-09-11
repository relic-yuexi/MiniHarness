"""Single-owner, durable append-only session event storage.

Only a successful fsync publishes an event to memory. A write error poisons the
writer because the disk commit status is unknown; close and reopen to reconcile.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from filelock import FileLock, Timeout


class StorageError(RuntimeError):
    pass


class CorruptLog(StorageError):
    pass


class SessionBusy(StorageError):
    pass


class EventConflict(StorageError):
    pass


class PoisonedStore(StorageError):
    pass


def canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(event: dict) -> str:
    return hashlib.sha256(canonical({k: v for k, v in event.items() if k != "hash"})).hexdigest()


def _strict_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("Duplicate JSON key")
        obj[key] = value
    return obj


class SessionStore:
    def __init__(
        self,
        root: Path,
        session_id: str | None = None,
        user_id: str = "default",
        *,
        repair: bool = True,
    ):
        self.session_id = (
            session_id or hashlib.sha256((user_id + "\0" + uuid4().hex).encode()).hexdigest()
        )
        if not re.fullmatch(r"[0-9a-f]{64}", self.session_id):
            raise ValueError("session_id must be 64 lowercase hex characters")
        self.root = Path(root).resolve()
        self.path = self.root / self.session_id
        self.user_id = user_id
        self.repair = repair
        self.events: list[dict] = []
        self._ids: dict[str, dict] = {}
        self._mutex = threading.RLock()
        self._file = None
        self._lock = None
        self._poisoned = False

    def open(self) -> SessionStore:
        with self._mutex:
            if self._file is not None:
                if self._poisoned:
                    raise PoisonedStore("Close and reopen the poisoned writer")
                return self
            self.path.mkdir(parents=True, exist_ok=True)
            if self.path.resolve().parent != self.root:
                raise StorageError("Session directory escapes root")
            self._lock = FileLock(self.path / "session.lock", timeout=0)
            try:
                self._lock.acquire()
            except Timeout as exc:
                self._lock = None
                raise SessionBusy("Session already has an owner") from exc
            try:
                log = self.path / "session.jsonl"
                if log.is_symlink():
                    raise StorageError("Session log must not be a symbolic link")
                self._file = log.open("a+b")
                self._file.seek(0)
                data = self._file.read()
                boundary = data.rfind(b"\n") + 1
                complete, tail = data[:boundary], data[boundary:]
                # Validate before repairing: a complete corrupt line is never hidden.
                events = self._validate(complete)
                if tail:
                    if not self.repair:
                        raise CorruptLog("Unterminated tail requires repair")
                    diagnostic = self.path / f"session.torn-{uuid4().hex}.bin"
                    with diagnostic.open("xb") as backup:
                        backup.write(tail)
                        backup.flush()
                        os.fsync(backup.fileno())
                    self._file.seek(boundary)
                    self._file.truncate()
                    self._file.flush()
                    os.fsync(self._file.fileno())
                self.events = events
                self._ids = {event["event_id"]: event for event in events}
                self._poisoned = False
                self._file.seek(0, os.SEEK_END)
                if not events:
                    self.append("session.created", {"user_id": self.user_id})
                return self
            except BaseException:
                self.close()
                raise

    def _validate(self, data: bytes) -> list[dict]:
        events = []
        seen = set()
        previous = "0" * 64
        required = {
            "schema_version",
            "event_id",
            "seq",
            "timestamp",
            "session_id",
            "turn_id",
            "step_id",
            "action_id",
            "type",
            "causation_id",
            "payload",
            "prehash",
            "hash",
        }
        for seq, line in enumerate(data.splitlines(), 1):
            try:
                event = json.loads(line, object_pairs_hook=_strict_object)
                if not isinstance(event, dict) or set(event) != required:
                    raise ValueError("Invalid envelope")
                if (
                    type(event["seq"]) is not int
                    or event["seq"] != seq
                    or event["schema_version"] != 1
                    or event["session_id"] != self.session_id
                    or event["prehash"] != previous
                    or event["hash"] != _digest(event)
                    or not isinstance(event["payload"], dict)
                    or not isinstance(event["event_id"], str)
                    or not event["event_id"]
                    or event["event_id"] in seen
                ):
                    raise ValueError("Invalid event or hash chain")
                if seq == 1 and (
                    event["type"] != "session.created"
                    or event["payload"].get("user_id") != self.user_id
                ):
                    raise ValueError("Session owner mismatch or missing creation event")
                seen.add(event["event_id"])
                previous = event["hash"]
                events.append(event)
            except (ValueError, TypeError, KeyError, UnicodeError) as exc:
                raise CorruptLog(f"Invalid session log at line {seq}: {exc}") from exc
        return events

    def append(
        self,
        event_type: str,
        payload: dict,
        *,
        turn_id=None,
        step_id=None,
        action_id=None,
        causation_id=None,
        event_id=None,
    ) -> dict:
        with self._mutex:
            if self._file is None:
                raise StorageError("Session is not open")
            if self._poisoned:
                raise PoisonedStore("Write status unknown; close and reopen before continuing")
            if not isinstance(payload, dict) or not isinstance(event_type, str) or not event_type:
                raise ValueError("Event requires nonempty type and object payload")
            if event_id is not None and (not isinstance(event_id, str) or not event_id):
                raise ValueError("event_id must be a nonempty string")
            semantic = {
                "type": event_type,
                "payload": payload,
                "turn_id": turn_id,
                "step_id": step_id,
                "action_id": action_id,
                "causation_id": causation_id,
            }
            # Freeze caller-owned nested containers and validate before touching disk.
            semantic = json.loads(canonical(semantic))
            if event_id in self._ids:
                prior = self._ids[event_id]
                if canonical({key: prior[key] for key in semantic}) != canonical(semantic):
                    raise EventConflict(f"Conflicting event_id: {event_id}")
                return json.loads(canonical(prior))
            event = dict(
                schema_version=1,
                event_id=event_id or uuid4().hex,
                seq=len(self.events) + 1,
                timestamp=datetime.now(UTC).isoformat(),
                session_id=self.session_id,
                **semantic,
                prehash=self.events[-1]["hash"] if self.events else "0" * 64,
            )
            event["hash"] = _digest(event)
            encoded = canonical(event) + b"\n"
            try:
                if self._file.write(encoded) != len(encoded):
                    raise OSError("Short event write")
                self._file.flush()
                os.fsync(self._file.fileno())
            except BaseException:
                self._poisoned = True
                raise
            self.events.append(event)
            self._ids[event["event_id"]] = event
            return json.loads(canonical(event))

    def close(self):
        with self._mutex:
            try:
                if self._file is not None:
                    self._file.close()
            finally:
                self._file = None
                if self._lock is not None:
                    self._lock.release()
                    self._lock = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()
