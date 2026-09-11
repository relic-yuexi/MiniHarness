import asyncio
import threading
from pathlib import Path

import pytest

import miniharness.tools as tools
from miniharness.tools import ToolContext, ToolRegistry, atomic_write, run_bash_job


@pytest.mark.asyncio
async def test_cancelled_sync_tool_settles_before_owner_can_continue(tmp_path):
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    registry = ToolRegistry()

    def slow_write(args, context):
        started.set()
        release.wait(3)
        (context.workspace / "settled.txt").write_text("done")
        finished.set()
        return tools.result("written", effect="applied")

    registry.register("slow", "test", {"type": "object"}, slow_write)
    worker = asyncio.create_task(
        registry.execute("slow", {}, ToolContext(tmp_path, tmp_path / "artifacts"))
    )
    try:
        await asyncio.to_thread(started.wait, 2)
        worker.cancel()
        await asyncio.sleep(0.01)
        worker.cancel()  # Repeated cancellation must not bypass settling.
        await asyncio.sleep(0.01)
        assert not worker.done()
        assert not finished.is_set()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await worker
    assert finished.is_set()
    assert (tmp_path / "settled.txt").read_text() == "done"


def test_cleanup_failure_after_publication_is_not_failed_write(tmp_path, monkeypatch, caplog):
    original = Path.unlink

    def broken_cleanup(path, *args, **kwargs):
        if path.name.startswith(".miniharness-"):
            raise PermissionError("cleanup denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", broken_cleanup)
    target = tmp_path / "created.txt"
    atomic_write(target, b"published", create_only=True)
    assert target.read_bytes() == b"published"
    assert "cleanup failed" in caplog.text


class HangingPipe:
    def __init__(self):
        self.first = True

    async def read(self, _):
        if self.first:
            self.first = False
            return b"captured before timeout"
        await asyncio.Event().wait()


class DeadParent:
    pid = 99999999
    returncode = 0

    def __init__(self):
        self.stdout, self.stderr = HangingPipe(), HangingPipe()

    async def wait(self):
        return 0

    def kill(self):
        pass


@pytest.mark.asyncio
async def test_bash_dead_parent_with_open_descendant_pipes_is_bounded(tmp_path, monkeypatch):
    process = DeadParent()

    async def spawn(*args, **kwargs):
        return process

    monkeypatch.setattr(tools.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(tools, "CLEANUP_TIMEOUT", 0.02)
    context = ToolContext(tmp_path, tmp_path / "artifacts")
    descriptor = {
        "executable": "fake-bash",
        "cwd": str(tmp_path),
        "command": "test",
        "timeout_seconds": 0.02,
    }
    result = await asyncio.wait_for(run_bash_job(descriptor, context), 1)
    assert result["effect_status"] == "unknown"
    assert result["data"]["cleanup_incomplete"]
    assert result["data"]["output_partial"]
    assert "captured before timeout" in result["preview_content"]
    assert result["error"]["code"] == "TIMEOUT"


@pytest.mark.asyncio
async def test_bash_cancellation_cleanup_is_bounded(tmp_path, monkeypatch):
    process = DeadParent()

    async def spawn(*args, **kwargs):
        return process

    monkeypatch.setattr(tools.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(tools, "CLEANUP_TIMEOUT", 0.02)
    context = ToolContext(tmp_path, tmp_path / "artifacts")
    descriptor = {
        "executable": "fake-bash",
        "cwd": str(tmp_path),
        "command": "test",
        "timeout_seconds": 10,
    }
    worker = asyncio.create_task(run_bash_job(descriptor, context))
    await asyncio.sleep(0.01)
    worker.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(worker, 1)
