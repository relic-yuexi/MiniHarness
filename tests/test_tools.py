import asyncio
import copy
import hashlib
import shutil

import pytest
from jsonschema.exceptions import SchemaError

from miniharness.tools import ToolContext, ToolRegistry, builtin_registry, run_bash_job


@pytest.fixture
def context(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return ToolContext(workspace, tmp_path / "artifacts", bash_executable=shutil.which("bash"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args",
    [
        '{"expression":"1","expression":"2"}',
        '{"expression":NaN}',
        "[]",
        '{"expression":"1"} extra',
        {"expression": "1", "extra": True},
        {"expression": 2},
    ],
)
async def test_invalid_arguments(context, args):
    assert (await builtin_registry().execute("calculator", args, context))["error"][
        "code"
    ] == "INVALID_ARGUMENT"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expression", ["__import__('os')", "x.y", "2**10000", "1/0", "True", "(-1)**0.5", "1e309"]
)
async def test_calculator_rejects(context, expression):
    assert not (
        await builtin_registry().execute("calculator", {"expression": expression}, context)
    )["ok"]


@pytest.mark.asyncio
async def test_registry_and_calculator(context):
    registry = builtin_registry()
    assert len(registry.specs()) == 7
    assert (await registry.execute("calculator", {"expression": "(12 + 3) * 2"}, context))["data"][
        "value"
    ] == "30"
    assert (await registry.execute("no", {}, context))["error"]["code"] == "UNKNOWN_TOOL"
    mutable = registry.specs()
    mutable[0]["name"] = "oops"
    assert registry.specs()[0]["name"] != "oops"
    with pytest.raises(ValueError):
        registry.register("read", "x", {}, lambda: None)
    with pytest.raises(SchemaError):
        ToolRegistry().register("a", "x", {"type": "nonsense"}, lambda: None)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [0, 32767, 32768, 32769, 1048600])
async def test_read_preview_and_snapshot(context, size):
    path = context.workspace / "sample.txt"
    raw = ("😀中" * (size // 7) + "x" * (size % 7)).encode()
    path.write_bytes(raw)
    registry = builtin_registry()
    result = await registry.execute("read", {"path": "sample.txt"}, context)
    assert result["ok"]
    assert len(result["preview_content"].encode()) <= 32768
    ref = result["ref_contents"][0]
    assert (context.artifacts / ref["artifact_id"]).read_bytes() == raw
    path.write_text("changed", encoding="utf-8")
    again = await registry.execute("read", {"artifact_id": ref["artifact_id"]}, context)
    assert again["data"]["source_sha256"] == hashlib.sha256(raw).hexdigest()
    if size > 32768:
        assert "这里被省略了" in result["preview_content"]
        assert result["preview_content"].endswith(raw[-100:].decode(errors="ignore"))


@pytest.mark.asyncio
async def test_read_ranges(context):
    raw = "😀中文\r\nhello".encode()
    (context.workspace / "a").write_bytes(raw)
    registry = builtin_registry()
    first = await registry.execute("read", {"path": "a", "offset": 2, "limit": 4}, context)
    assert first["data"]["byte_start"] == 0
    assert first["data"]["next_offset"] == 4
    assert first["preview_content"].endswith("😀")
    offset = 0
    parts = []
    while offset < len(raw):
        current = await registry.execute(
            "read", {"path": "a", "offset": offset, "limit": 4}, context
        )
        parts.append((context.artifacts / current["ref_contents"][0]["artifact_id"]).read_bytes())
        offset = current["data"]["next_offset"]
    assert b"".join(parts) == raw
    for args in [
        {"path": "a", "offset": 1},
        {"path": "a", "limit": 4},
        {"path": "a", "offset": 100, "limit": 4},
        {"path": "a", "artifact_id": "a" * 64},
    ]:
        assert not (await registry.execute("read", args, context))["ok"]


@pytest.mark.asyncio
async def test_write_edit_conflicts(context):
    registry = builtin_registry()
    initial = await registry.execute("write", {"path": "a", "content": "foo foo\r\n"}, context)
    sha = initial["data"]["sha256"]
    assert initial["effect_status"] == "applied"
    assert not (await registry.execute("write", {"path": "a", "content": "bad"}, context))["ok"]
    assert not (
        await registry.execute(
            "write", {"path": "a", "content": "bad", "create_only": False}, context
        )
    )["ok"]
    args = {"path": "a", "old_string": "foo", "new_string": "bar", "expected_sha256": sha}
    assert not (await registry.execute("edit", args, context))["ok"]
    changed = await registry.execute("edit", args | {"expected_matches": 2}, context)
    assert changed["ok"]
    assert (context.workspace / "a").read_bytes() == b"bar bar\r\n"
    assert not (await registry.execute("edit", args | {"expected_matches": 2}, context))["ok"]
    assert not list(context.workspace.glob(".miniharness-*"))


@pytest.mark.asyncio
async def test_concurrent_create_and_edit(context):
    registry = builtin_registry()
    created = await asyncio.gather(
        *(registry.execute("write", {"path": "a", "content": str(i)}, context) for i in range(6))
    )
    assert sum(x["ok"] for x in created) == 1
    raw = (context.workspace / "a").read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    changed = await asyncio.gather(
        *(
            registry.execute(
                "edit",
                {
                    "path": "a",
                    "old_string": raw.decode(),
                    "new_string": "changed",
                    "expected_sha256": sha,
                },
                context,
            )
            for _ in range(6)
        )
    )
    assert sum(x["ok"] for x in changed) == 1


@pytest.mark.asyncio
async def test_path_and_encoding(context):
    registry = builtin_registry()
    assert not (await registry.execute("read", {"path": "../outside"}, context))["ok"]
    (context.workspace / "binary").write_bytes(b"\xff")
    assert (await registry.execute("read", {"path": "binary"}, context))["error"][
        "code"
    ] == "UNSUPPORTED_FILE"
    context.protected_root = context.workspace / "sessions"
    assert (await registry.execute("write", {"path": "sessions/a", "content": "x"}, context))[
        "error"
    ]["code"] == "OUT_OF_SCOPE"


@pytest.mark.asyncio
async def test_todo_is_proposal_not_mutation(context):
    registry = builtin_registry()
    original = copy.deepcopy(context.todo)
    args = {
        "operation": "replace",
        "expected_revision": 0,
        "items": [{"id": "1", "text": "task", "status": "pending"}],
    }
    changed = await registry.execute("todo", args, context)
    assert changed["_todo"]["revision"] == 1
    assert context.todo == original
    context.todo = changed["_todo"]
    assert (await registry.execute("todo", args, context))["error"]["code"] == "CONFLICT"
    assert not (
        await registry.execute(
            "todo",
            {"operation": "replace", "expected_revision": 1, "items": args["items"] * 2},
            context,
        )
    )["ok"]
    assert not (await registry.execute("todo", {"operation": "list", "items": []}, context))["ok"]
    assert not (await registry.execute("todo", {"operation": "replace"}, context))["ok"]
    assert (await registry.execute("todo", {"operation": "list"}, context))["data"][
        "items"
    ] == args["items"]


@pytest.mark.asyncio
async def test_search_explicitly_mock(context):
    answer = await builtin_registry().execute("search", {"query": "agent"}, context)
    assert answer["data"]["mock"] is True
    assert answer["data"]["results"]


@pytest.mark.asyncio
async def test_bash_missing(context):
    context.bash_executable = "/does/not/exist"
    assert not (await builtin_registry().execute("bash", {"command": "echo x"}, context))["ok"]


@pytest.mark.asyncio
async def test_bash_foreground_background_timeout(context):
    if not context.bash_executable:
        pytest.skip("Bash unavailable")
    registry = builtin_registry()
    normal = await registry.execute("bash", {"command": "printf hello; printf error >&2"}, context)
    assert normal["ok"], normal
    assert normal["data"]["stdout"] == "hello"
    assert normal["data"]["stderr"] == "error"
    error = await registry.execute("bash", {"command": "exit 3"}, context)
    assert error["data"]["exit_code"] == 3
    assert not error["ok"]
    accepted = await registry.execute(
        "bash", {"command": "printf done", "background": True}, context
    )
    assert accepted["data"]["status"] == "accepted"
    complete = await run_bash_job(accepted["_job"], context)
    assert complete["data"]["stdout"] == "done"
    timeout = await registry.execute("bash", {"command": "sleep 10", "timeout_seconds": 1}, context)
    assert timeout["data"]["timed_out"]
    assert timeout["effect_status"] == "unknown"


@pytest.mark.asyncio
async def test_atomic_replace_failure_preserves_original(context, monkeypatch):
    import miniharness.tools as module

    path = context.workspace / "original"
    path.write_bytes(b"original")

    def fail(*args):
        raise PermissionError("simulated sharing violation")

    monkeypatch.setattr(module.os, "replace", fail)
    answer = await builtin_registry().execute(
        "write",
        {
            "path": "original",
            "content": "replacement",
            "create_only": False,
            "expected_sha256": hashlib.sha256(b"original").hexdigest(),
        },
        context,
    )
    assert answer["error"]["code"] == "PERMISSION_DENIED"
    assert path.read_bytes() == b"original"
    assert not list(context.workspace.glob(".miniharness-*"))


@pytest.mark.asyncio
async def test_read_invalid_utf8_crosses_chunk(context):
    (context.workspace / "bad").write_bytes(b"a" * 65535 + b"\xf0\x80")
    answer = await builtin_registry().execute("read", {"path": "bad"}, context)
    assert answer["error"]["code"] == "UNSUPPORTED_FILE"
    assert not list(context.artifacts.glob(".snapshot-*"))


@pytest.mark.asyncio
async def test_shell_output_limit_drains_both_streams(context, monkeypatch):
    import miniharness.tools as module

    if not context.bash_executable:
        pytest.skip("Bash unavailable")
    monkeypatch.setattr(module, "OUTPUT_LIMIT", 1000)
    answer = await builtin_registry().execute(
        "bash",
        {"command": "for ((i=0; i<2000; i++)); do printf abc; printf def >&2; done"},
        context,
    )
    assert answer["ok"]
    assert answer["data"]["truncated"]
    assert answer["data"]["dropped_bytes"] == 10000
    assert len(answer["preview_content"].encode()) <= 32768


@pytest.mark.asyncio
async def test_background_does_not_launch_until_dispatched(context):
    if not context.bash_executable:
        pytest.skip("Bash unavailable")
    accepted = await builtin_registry().execute(
        "bash", {"command": "printf x > marker", "background": True}, context
    )
    assert accepted["ok"]
    assert not (context.workspace / "marker").exists()
    completed = await run_bash_job(accepted["_job"], context)
    assert completed["ok"]
    assert (context.workspace / "marker").read_text() == "x"
