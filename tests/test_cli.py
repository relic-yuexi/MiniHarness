import asyncio
import json
from types import SimpleNamespace

import pytest

from miniharness import cli
from miniharness.config import Config
from miniharness.models import Completion, ProviderConfig
from miniharness.runtime import Runtime
from miniharness.storage import CorruptLog, SessionStore


@pytest.fixture
def config(tmp_path):
    return Config(
        provider=ProviderConfig(model="test"),
        workspace=tmp_path / "workspace",
        session_root=tmp_path / "sessions",
        stream=False,
    )


class Provider:
    def __init__(self, fail=False):
        self.fail = fail

    async def complete(self, messages, tools, system, **kwargs):
        await asyncio.sleep(0.005)
        if self.fail:
            raise ValueError("simulated failure")
        return Completion(text="answer", usage={"input_tokens_total": 1, "output_tokens": 1})

    async def aclose(self):
        pass


def test_parser_and_missing_config(capsys):
    assert cli.execute([]) == 0
    assert "MiniHarness" in capsys.readouterr().out
    assert cli.execute(["--config", "missing-unique-config.toml", "doctor"]) == 1
    assert "Configuration not found" in capsys.readouterr().err
    with pytest.raises(SystemExit) as result:
        cli.execute(["--version"])
    assert result.value.code == 0


def test_doctor_does_not_leak_key(config, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda _: config)
    monkeypatch.setenv("OPENAI_API_KEY", "secret-value-do-not-show")
    assert cli.execute(["doctor"]) == 0
    output = capsys.readouterr().out
    assert "secret-value-do-not-show" not in output
    assert json.loads(output)["api_key_present"] is True
    assert not config.session_root.exists()


def test_inspection_is_read_only_even_when_owned(config):
    with SessionStore(config.session_root, user_id="alice") as store:
        store.append(
            "usage.recorded",
            {"attempt_id": "a", "usage": {"input_tokens_total": 10, "output_tokens": 5}},
        )
        path = store.path / "session.jsonl"
        before = path.read_bytes()
        files = sorted(p.name for p in store.path.iterdir())
        info = cli.inspect_session(config, store.session_id)
        assert info["user_id"] == "alice"
        assert info["hash_chain"] == "valid"
        assert info["total_tokens"] == 15
        assert info["last_seq"] == 2
        assert path.read_bytes() == before
        assert sorted(p.name for p in store.path.iterdir()) == files


def test_inspection_torn_log_never_repairs(config):
    with SessionStore(config.session_root) as store:
        session_id = store.session_id
        path = store.path / "session.jsonl"
    with path.open("ab") as file:
        file.write(b'{"partial":')
    before = path.read_bytes()
    with pytest.raises(CorruptLog):
        cli.inspect_session(config, session_id)
    assert path.read_bytes() == before
    assert not list(path.parent.glob("session.torn-*"))


def test_sessions_and_inspect(config, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda _: config)
    assert cli.execute(["sessions"]) == 0
    assert json.loads(capsys.readouterr().out) == []
    assert not config.session_root.exists()
    with SessionStore(config.session_root) as store:
        session_id = store.session_id
    assert cli.execute(["sessions"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["session_id"] == session_id
    assert cli.execute(["inspect", session_id]) == 0
    assert json.loads(capsys.readouterr().out)["hash_chain"] == "valid"
    assert cli.execute(["inspect", "../escape"]) == 1


@pytest.mark.parametrize(
    "protocol,event,expected",
    [
        (
            "openai_chat",
            {"choices": [{"index": 0, "delta": {"content": "a", "reasoning_content": "secret"}}]},
            "a",
        ),
        (
            "anthropic_messages",
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "b"}},
            "b",
        ),
        (
            "anthropic_messages",
            {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "secret"},
            },
            "",
        ),
        ("openai_responses", {"type": "response.output_text.delta", "delta": "c"}, "c"),
        (
            "openai_responses",
            {"type": "response.reasoning_summary_text.delta", "delta": "secret"},
            "",
        ),
    ],
)
def test_visible_delta(protocol, event, expected):
    assert cli.visible_delta({"protocol": protocol, "event": event}) == expected
    assert cli.visible_delta({"protocol": protocol, "event": event, "maintenance": True}) == ""


@pytest.mark.asyncio
async def test_console_no_duplicates_and_jobs(capsys):
    console = cli.Console()
    await console.event(
        {
            "type": "provider_delta",
            "protocol": "openai_responses",
            "event": {"type": "response.output_text.delta", "delta": "answer"},
        }
    )
    final = {"type": "turn.ended", "turn_id": "a", "status": "completed", "text": "answer"}
    await console.event(final)
    console.complete(final)
    assert capsys.readouterr().out.count("answer") == 1
    await console.event(
        {
            "type": "job.completed",
            "job_id": "j",
            "status": "completed",
            "result": {"preview_content": "job output"},
        }
    )
    assert "job output" in capsys.readouterr().out


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_run_once_exit_and_output(config, monkeypatch, capsys, fail):
    monkeypatch.setattr(cli, "Runtime", lambda *a, **kw: Runtime(*a, provider=Provider(fail), **kw))
    status = await cli.run_once(
        config, SimpleNamespace(session=None, user="default", prompt="hello")
    )
    assert status == (1 if fail else 0)
    output = capsys.readouterr()
    assert output.out.count("answer") == (0 if fail else 1)
    assert "Session:" in output.err


@pytest.mark.asyncio
async def test_chat_commands_and_nonblocking_queue(config, monkeypatch, capsys):
    instances = []

    def create(*args, **kwargs):
        runtime = Runtime(*args, provider=Provider(), **kwargs)
        instances.append(runtime)
        return runtime

    monkeypatch.setattr(cli, "Runtime", create)
    inputs = iter(
        [
            "/help",
            "/session",
            "/new",
            "/todo",
            "hello",
            "/followup again",
            "/steer use details",
            "/unknown",
            "/quit",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    assert await cli.chat(config, SimpleNamespace(session=None, user="default")) == 0
    assert len(instances[0].state.requests) == 3
    assert all(request["status"] == "completed" for request in instances[0].state.requests.values())
    output = capsys.readouterr()
    assert "independent session" in output.out
    assert "Unknown or incomplete" in output.err


@pytest.mark.asyncio
async def test_chat_compact_cancel_and_eof(config, monkeypatch, capsys):
    monkeypatch.setattr(cli, "Runtime", lambda *a, **kw: Runtime(*a, provider=Provider(), **kw))
    lines = iter(["/compact", "/cancel"])

    def read(_):
        try:
            return next(lines)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", read)
    assert await cli.chat(config, SimpleNamespace(session=None, user="default")) == 0
    assert "Cancellation requested" in capsys.readouterr().out


def test_inspection_detects_tampered_hash_without_writing(config):
    with SessionStore(config.session_root) as store:
        path = store.path / "session.jsonl"
        session_id = store.session_id
    raw = path.read_bytes().replace(b'"user_id":"default"', b'"user_id":"changed"')
    path.write_bytes(raw)
    with pytest.raises(CorruptLog):
        cli.inspect_session(config, session_id)
    assert path.read_bytes() == raw
