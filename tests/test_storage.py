import concurrent.futures
import json
import os
import subprocess
import sys

import pytest

from miniharness.storage import (
    CorruptLog,
    EventConflict,
    PoisonedStore,
    SessionBusy,
    SessionStore,
    StorageError,
)


def test_roundtrip_isolation_and_ids(tmp_path):
    with SessionStore(tmp_path, user_id="alice") as first:
        sid = first.session_id
        payload = {"nested": ["中文"]}
        event = first.append("test", payload, event_id="stable", turn_id="turn")
        payload["nested"].append("changed")
        event["payload"]["nested"].append("also changed")
        assert first.events[-1]["payload"] == {"nested": ["中文"]}
        assert first.events[-1]["prehash"] == first.events[0]["hash"]
        assert (
            first.append("test", {"nested": ["中文"]}, event_id="stable", turn_id="turn")["seq"]
            == 2
        )
        with pytest.raises(EventConflict):
            first.append("test", {}, event_id="stable")
        with SessionStore(tmp_path, user_id="alice") as second:
            assert second.session_id != sid
            assert len(second.events) == 1
    with SessionStore(tmp_path, sid, "alice") as recovered:
        assert len(recovered.events) == 2
        recovered.append("test", {"nested": ["中文"]}, event_id="stable", turn_id="turn")
        assert len(recovered.events) == 2
    with pytest.raises(CorruptLog, match="owner"):
        SessionStore(tmp_path, sid, "bob").open()
    with SessionStore(tmp_path, sid, "alice"):
        pass  # Failed recovery released the lock.


@pytest.mark.parametrize("tail", [b'{"seq":', b'{"valid":"json"}', b"\xff\xfe"])
def test_torn_tail_preserved_and_repaired(tmp_path, tail):
    with SessionStore(tmp_path) as store:
        sid, path = store.session_id, store.path / "session.jsonl"
    original = path.read_bytes()
    path.write_bytes(original + tail)
    with pytest.raises(CorruptLog, match="tail"):
        SessionStore(tmp_path, sid, repair=False).open()
    assert path.read_bytes() == original + tail
    with SessionStore(tmp_path, sid) as restored:
        assert len(restored.events) == 1
        assert path.read_bytes() == original
        assert next(restored.path.glob("session.torn-*.bin")).read_bytes() == tail


@pytest.mark.parametrize("corrupt", [b"no json\n", b"{}\n", b"\n"])
def test_complete_corrupt_tail_never_repaired(tmp_path, corrupt):
    with SessionStore(tmp_path) as store:
        sid, path = store.session_id, store.path / "session.jsonl"
    damaged = path.read_bytes() + corrupt
    path.write_bytes(damaged)
    with pytest.raises(CorruptLog):
        SessionStore(tmp_path, sid).open()
    assert path.read_bytes() == damaged


def test_hash_tamper(tmp_path):
    with SessionStore(tmp_path) as store:
        sid, path = store.session_id, store.path / "session.jsonl"
    event = json.loads(path.read_bytes())
    event["payload"]["user_id"] = "tampered"
    path.write_text(json.dumps(event) + "\n", encoding="utf-8")
    with pytest.raises(CorruptLog):
        SessionStore(tmp_path, sid).open()


def test_unknown_write_poisons_then_reconciles(tmp_path, monkeypatch):
    with SessionStore(tmp_path) as store:
        sid = store.session_id
        with monkeypatch.context() as patch:
            patch.setattr(os, "fsync", lambda _: (_ for _ in ()).throw(OSError("disk")))
            with pytest.raises(OSError):
                store.append("test", {"a": 1}, event_id="request")
            assert len(store.events) == 1
            with pytest.raises(PoisonedStore):
                store.append("another", {})
        with pytest.raises(PoisonedStore):
            store.open()
    with SessionStore(tmp_path, sid) as recovered:
        assert len(recovered.events) == 2
        recovered.append("test", {"a": 1}, event_id="request")
        assert len(recovered.events) == 2


def test_validation_errors_do_not_poison(tmp_path):
    with SessionStore(tmp_path) as store:
        for value in [float("nan"), float("inf"), object()]:
            with pytest.raises((ValueError, TypeError)):
                store.append("test", {"bad": value})
        store.append("valid", {})
        assert len(store.events) == 2


def test_thread_serialization(tmp_path):
    with SessionStore(tmp_path) as store:
        sid = store.session_id
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda n: store.append("test", {"n": n}), range(40)))
        assert [e["seq"] for e in store.events] == list(range(1, 42))
    with SessionStore(tmp_path, sid) as recovered:
        assert len(recovered.events) == 41


def test_process_and_same_process_lock(tmp_path):
    with SessionStore(tmp_path) as store:
        with pytest.raises(SessionBusy):
            SessionStore(tmp_path, store.session_id).open()
        script = (
            "from pathlib import Path; from miniharness.storage import SessionStore, SessionBusy; "
            "import sys\ntry:\n SessionStore(Path(sys.argv[1]),sys.argv[2]).open()\n"
            "except SessionBusy:\n sys.exit(23)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), store.session_id],
            capture_output=True,
            timeout=10,
            check=False,
        )
        assert result.returncode == 23, result.stderr


@pytest.mark.parametrize("sid", ["../escape", "a" * 63, "A" * 64, "x" * 64])
def test_path_rejection(tmp_path, sid):
    with pytest.raises(ValueError):
        SessionStore(tmp_path, sid)


def test_append_requires_open(tmp_path):
    store = SessionStore(tmp_path)
    with pytest.raises(StorageError):
        store.append("test", {})
    store.close()


def test_creation_fsync_failure_releases_lock(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", lambda _: (_ for _ in ()).throw(OSError("disk")))
        with pytest.raises(OSError):
            store.open()
    with SessionStore(tmp_path, store.session_id) as recovered:
        assert len(recovered.events) == 1


def test_corruption_before_tail_is_not_silently_repaired(tmp_path):
    with SessionStore(tmp_path) as store:
        sid, path = store.session_id, store.path / "session.jsonl"
    damaged = path.read_bytes() + b"invalid\npartial"
    path.write_bytes(damaged)
    with pytest.raises(CorruptLog):
        SessionStore(tmp_path, sid).open()
    assert path.read_bytes() == damaged
    assert not list(store.path.glob("session.torn-*.bin"))
