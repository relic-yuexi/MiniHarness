"""End-to-end loop tests with deterministic model decisions and race barriers."""

import asyncio
import json
from copy import deepcopy

import pytest

from miniharness.config import Config
from miniharness.context import fingerprint, validate_pairing
from miniharness.models import Completion, ProviderConfig, ToolCall
from miniharness.runtime import Runtime
from miniharness.storage import SessionStore
from miniharness.tools import ToolRegistry, result


class FakeProvider:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = False

    async def complete(self, messages, tools, system, **kwargs):
        self.calls.append(
            {"messages": deepcopy(messages), "tools": deepcopy(tools), "system": system, **kwargs}
        )
        validate_pairing(messages)
        if not self.responses:
            raise AssertionError("Unexpected model call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return await response(messages, tools, system, **kwargs)
        return response

    async def aclose(self):
        self.closed = True


@pytest.fixture
def config(tmp_path):
    return Config(
        provider=ProviderConfig(model="fake"),
        session_root=tmp_path / "sessions",
        workspace=tmp_path / "workspace",
        compact_target_tokens=500,
    )


def tool(name, args, call_id="call-1"):
    return Completion(tool_calls=[ToolCall(call_id, name, json.dumps(args))])


@pytest.mark.asyncio
async def test_direct_followup_preserves_cache_prefix_and_usage(config):
    fake = FakeProvider(
        Completion(
            text="You said red",
            usage={"input_tokens_total": 10, "output_tokens": 4, "cache_read_input_tokens": 5},
        ),
        Completion(text="Your favorite color is red"),
    )
    async with Runtime(config, provider=fake) as runtime:
        first = await runtime.ask("My favorite color is red")
        prefix = deepcopy(runtime.state.messages)
        second = await runtime.ask("What color did I say?")
        assert first["text"] == "You said red"
        assert second["text"] == "Your favorite color is red"
        assert first["turn_id"] != second["turn_id"]
        assert runtime.state.messages[: len(prefix)] == prefix
        assert fake.calls[1]["messages"][: len(prefix)] == [
            {key: value for key, value in message.items() if not key.startswith("_")}
            for message in prefix
        ]
        assert runtime.state.total_tokens == 14
        assert len(runtime.state.usage) == 2
    assert fake.closed


@pytest.mark.asyncio
async def test_tools_then_tool_followup(config):
    fake = FakeProvider(
        tool("calculator", {"expression": "6*7"}),
        Completion(text="42"),
        tool("calculator", {"expression": "42+8"}, "call-2"),
        Completion(text="50"),
    )
    async with Runtime(config, provider=fake) as runtime:
        assert (await runtime.ask("6 times 7?"))["text"] == "42"
        assert (await runtime.ask("Add 8 to that"))["text"] == "50"
        validate_pairing(runtime.state.messages)
        results = [m for m in runtime.state.messages if m["role"] == "tool"]
        assert len(results) == 2
        assert "42" in results[0]["content"]
        assert "50" in results[1]["content"]
        assert len(runtime.state.actions) == 2


@pytest.mark.asyncio
async def test_session_resume_isolation_and_request_idempotency(config):
    async with Runtime(config, provider=FakeProvider(Completion(text="remembered"))) as first:
        sid = first.session_id
        original = await first.ask("Remember kiwi", request_id="request-1")
        assert await first.ask("Remember kiwi", request_id="request-1") == original
        with pytest.raises(ValueError):
            await first.ask("Different input", request_id="request-1")
    fake = FakeProvider(Completion(text="kiwi"))
    async with Runtime(config, provider=fake, session_id=sid) as resumed:
        assert await resumed.ask("Remember kiwi", request_id="request-1") == original
        assert not fake.calls
        assert (await resumed.ask("What fruit?"))["text"] == "kiwi"
        assert any("Remember kiwi" in m["content"] for m in fake.calls[0]["messages"])
    isolated = FakeProvider(Completion(text="I do not know"))
    async with Runtime(config, provider=isolated) as second:
        assert second.session_id != sid
        await second.ask("What fruit?")
        assert not any("kiwi" in m["content"] for m in isolated.calls[0]["messages"])


@pytest.mark.asyncio
async def test_todo_atomic_state_restores(config):
    items = [{"id": "a", "text": "Implement tests", "status": "pending"}]
    fake = FakeProvider(
        tool("todo", {"operation": "replace", "expected_revision": 0, "items": items}),
        Completion(text="saved"),
    )
    async with Runtime(config, provider=fake) as runtime:
        sid = runtime.session_id
        await runtime.ask("Track tests")
        assert runtime.state.todo == {"revision": 1, "items": items}
        updates = [e for e in runtime.store.events if e["type"] == "action.completed"]
        assert updates[0]["payload"]["todo"] == runtime.state.todo
    async with Runtime(config, provider=FakeProvider(), session_id=sid) as restored:
        assert restored.state.todo == {"revision": 1, "items": items}


@pytest.mark.asyncio
async def test_step_limit_closes_tool_batch(config):
    config.max_steps_per_turn = 1
    fake = FakeProvider(tool("calculator", {"expression": "1+1"}))
    async with Runtime(config, provider=fake) as runtime:
        answer = await runtime.ask("Keep calculating")
        assert answer["status"] == "limited"
        assert len(fake.calls) == 1
        validate_pairing(runtime.state.messages)
        assert not runtime.state.current_turn


@pytest.mark.asyncio
async def test_unknown_and_invalid_tools_always_get_results(config):
    fake = FakeProvider(
        Completion(
            tool_calls=[
                ToolCall("bad-1", "missing", "{}"),
                ToolCall("bad-2", "calculator", "{broken"),
            ]
        ),
        Completion(text="Cannot do that"),
    )
    async with Runtime(config, provider=fake) as runtime:
        await runtime.ask("Use invalid tools")
        validate_pairing(runtime.state.messages)
        results = [m for m in runtime.state.messages if m["role"] == "tool"]
        assert [m["tool_call_id"] for m in results] == ["bad-1", "bad-2"]
        assert all(json.loads(m["content"])["ok"] is False for m in results)


@pytest.mark.asyncio
async def test_steer_waits_for_full_tool_batch_followup_new_turn(config):
    entered, release = asyncio.Event(), asyncio.Event()
    registry = ToolRegistry()

    async def blocked(args, ctx):
        entered.set()
        await release.wait()
        return result("done")

    registry.register(
        "blocked",
        "Barrier tool",
        {"type": "object", "properties": {}, "additionalProperties": False},
        blocked,
    )
    fake = FakeProvider(
        tool("blocked", {}), Completion(text="Steered answer"), Completion(text="Followup answer")
    )
    async with Runtime(config, provider=fake, registry=registry) as runtime:
        original = await runtime.submit("Start")
        await asyncio.wait_for(entered.wait(), 5)
        steering = await runtime.steer("Actually focus on tests")
        later = await runtime.followup("Then summarize")
        assert len(fake.calls) == 1
        release.set()
        a, b, c = await asyncio.wait_for(
            asyncio.gather(original.wait(), steering.wait(), later.wait()), 5
        )
        assert a["turn_id"] == b["turn_id"] != c["turn_id"]
        assert [m["role"] for m in fake.calls[1]["messages"]] == [
            "user",
            "assistant",
            "tool",
            "user",
        ]
        assert fake.calls[1]["messages"][-1]["content"] == "Actually focus on tests"
        assert fake.calls[2]["messages"][-1]["content"] == "Then summarize"


@pytest.mark.asyncio
async def test_partial_stream_failure_never_commits_or_executes(config):
    async def broken(messages, tools, system, **kwargs):
        if kwargs.get("on_delta"):
            await kwargs["on_delta"]({"type": "text", "text": "partial"})
        raise RuntimeError("stream disconnected")

    async with Runtime(
        config, provider=FakeProvider(broken, Completion(text="recovered"))
    ) as runtime:
        failed = await runtime.ask("Start")
        assert failed["status"] == "failed"
        assert not any(m["role"] == "assistant" for m in runtime.state.messages)
        assert not runtime.state.actions
        assert (await runtime.ask("Try next"))["text"] == "recovered"


@pytest.mark.asyncio
async def test_manual_compact_atomic_failure_then_success(config):
    config.keep_recent_turns = 0
    fake = FakeProvider(
        Completion(text="remember red"),
        RuntimeError("summary failure"),
        Completion(text="[用户任务]\nRemember the user's color is red.\n[工作状态]\nColor saved."),
        Completion(text="red"),
    )
    async with Runtime(config, provider=fake) as runtime:
        await runtime.ask("My color is red")
        original = deepcopy(runtime.state.messages)
        try:
            await runtime.compact()
        except RuntimeError:
            pass
        assert runtime.state.messages == original
        assert runtime.state.epoch == 0
        await runtime.compact()
        assert runtime.state.epoch == 1
        assert "red" in json.dumps(runtime.state.messages)
        assert fake.calls[2]["tools"] == []
        assert (await runtime.ask("Which color?"))["text"] == "red"
        validate_pairing(runtime.state.messages)


@pytest.mark.asyncio
async def test_session_token_budget_stops_new_model_calls(config):
    config.max_session_tokens = 10
    fake = FakeProvider(
        Completion(text="done", usage={"input_tokens_total": 8, "output_tokens": 3})
    )
    async with Runtime(config, provider=fake) as runtime:
        await runtime.ask("First")
        limited = await runtime.ask("Second")
        assert limited["status"] == "limited"
        assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_steer_during_direct_response_continues_current_turn(config):
    entered, release = asyncio.Event(), asyncio.Event()

    async def pending(messages, tools, system, **kwargs):
        entered.set()
        await release.wait()
        return Completion(text="First answer")

    fake = FakeProvider(pending, Completion(text="Updated answer"))
    async with Runtime(config, provider=fake) as runtime:
        original = await runtime.submit("Start")
        await asyncio.wait_for(entered.wait(), 5)
        steer = await runtime.steer("One more requirement")
        release.set()
        a, b = await asyncio.wait_for(asyncio.gather(original.wait(), steer.wait()), 5)
        assert a["turn_id"] == b["turn_id"]
        assert a["text"] == "Updated answer"
        assert [m["role"] for m in fake.calls[1]["messages"]] == ["user", "assistant", "user"]


@pytest.mark.asyncio
async def test_pending_duplicate_requests_share_one_execution(config):
    entered, release = asyncio.Event(), asyncio.Event()

    async def pending(messages, tools, system, **kwargs):
        entered.set()
        await release.wait()
        return Completion(text="once")

    fake = FakeProvider(pending)
    async with Runtime(config, provider=fake) as runtime:
        first = await runtime.submit("Do once", request_id="same")
        await asyncio.wait_for(entered.wait(), 5)
        duplicate = await runtime.submit("Do once", request_id="same")
        release.set()
        a, b = await asyncio.wait_for(asyncio.gather(first.wait(), duplicate.wait()), 5)
        assert a == b
        assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_queue_capacity_rejects_before_ack(config):
    config.queue_capacity = 1
    entered, release = asyncio.Event(), asyncio.Event()

    async def pending(messages, tools, system, **kwargs):
        entered.set()
        await release.wait()
        return Completion(text="first")

    fake = FakeProvider(pending, Completion(text="second"))
    async with Runtime(config, provider=fake) as runtime:
        first = await runtime.submit("First")
        await asyncio.wait_for(entered.wait(), 5)
        second = await runtime.followup("Second", request_id="queued")
        try:
            with pytest.raises((ValueError, RuntimeError)):
                await runtime.followup("Overflow", request_id="overflow")
            assert "overflow" not in runtime.state.requests
        finally:
            release.set()
        await asyncio.wait_for(asyncio.gather(first.wait(), second.wait()), 5)


@pytest.mark.asyncio
async def test_oversized_input_is_not_accepted(config):
    config.max_input_bytes = 8
    async with Runtime(config, provider=FakeProvider()) as runtime:
        with pytest.raises((ValueError, RuntimeError)):
            await runtime.submit("过长的中文输入", request_id="oversized")
        assert "oversized" not in runtime.state.requests


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["proposed", "started", "completed"])
async def test_recovery_closes_protocol_without_repeating_side_effect(config, boundary):
    executions = []
    registry = ToolRegistry()

    async def effect(args, ctx):
        executions.append("executed")
        return result("actual effect")

    registry.register("effect", "Side effect", {"type": "object"}, effect)
    async with Runtime(config, provider=FakeProvider(), registry=registry) as original:
        sid = original.session_id
    call = {"id": "effect-call", "name": "effect", "arguments": "{}"}
    with SessionStore(config.session_root, sid) as store:
        store.append(
            "input.enqueued",
            {
                "request_id": "crashed",
                "text": "do effect",
                "mode": "followup",
                "signature": fingerprint({"text": "do effect", "mode": "followup"}),
                "target_turn": None,
            },
        )
        store.append("turn.started", {}, turn_id="crashed-turn")
        store.append(
            "user.accepted",
            {
                "request_id": "crashed",
                "message": {"role": "user", "content": "do effect", "_turn_id": "crashed-turn"},
            },
            turn_id="crashed-turn",
        )
        store.append(
            "assistant.committed",
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [call],
                    "_turn_id": "crashed-turn",
                }
            },
            turn_id="crashed-turn",
        )
        if boundary != "proposed":
            store.append(
                "action.started",
                {"call": call},
                turn_id="crashed-turn",
                step_id="crashed-step",
                action_id="crashed-action",
            )
        if boundary == "completed":
            store.append(
                "action.completed",
                {"call": call, "result": result("already done")},
                turn_id="crashed-turn",
                step_id="crashed-step",
                action_id="crashed-action",
            )
    fake = FakeProvider(Completion(text="Recovered"))
    async with Runtime(config, provider=fake, registry=registry, session_id=sid) as recovered:
        assert executions == []
        assert fake.calls == []
        validate_pairing(recovered.state.messages)
        message = recovered.state.messages[-1]
        output = json.loads(message["content"])
        if boundary == "completed":
            assert output["preview_content"] == "already done"
        else:
            assert output["error"]["code"] == ("UNKNOWN" if boundary == "started" else "CANCELLED")
        assert recovered.state.requests["crashed"]["status"] == "interrupted"
        assert (await recovered.ask("Continue safely"))["text"] == "Recovered"
        assert executions == []


@pytest.mark.asyncio
async def test_automatic_compact_before_context_overflow(config):
    config.provider.context_window = 8000
    config.provider.max_output_tokens = 256
    config.keep_recent_turns = 0
    fake = FakeProvider(
        Completion(text="x" * 12000),
        Completion(text="[用户任务]\nContinue.\n[工作状态]\nPrior response was long."),
        Completion(text="continued"),
    )
    async with Runtime(config, provider=fake) as runtime:
        await runtime.ask("Start")
        answer = await runtime.ask("Continue " + "y" * 5000)
        assert answer["text"] == "continued"
        assert runtime.state.epoch == 1
        assert fake.calls[1]["tools"] == []
        assert fake.calls[2]["tools"]
        validate_pairing(runtime.state.messages)


@pytest.mark.asyncio
async def test_cancel_active_action_closes_all_results_and_stops_new_effects(config):
    entered = asyncio.Event()
    executions = []
    registry = ToolRegistry()

    async def blocked(args, ctx):
        executions.append("started")
        entered.set()
        await asyncio.Event().wait()

    registry.register("blocked", "Wait for cancellation", {"type": "object"}, blocked)
    fake = FakeProvider(
        Completion(tool_calls=[ToolCall("one", "blocked", "{}"), ToolCall("two", "blocked", "{}")])
    )
    async with Runtime(config, provider=fake, registry=registry) as runtime:
        ticket = await runtime.submit("Do two effects")
        await asyncio.wait_for(entered.wait(), 5)
        await runtime.cancel()
        answer = await asyncio.wait_for(ticket.wait(), 5)
        assert answer["status"] == "cancelled"
        assert executions == ["started"]
        validate_pairing(runtime.state.messages)
        outputs = [json.loads(m["content"]) for m in runtime.state.messages if m["role"] == "tool"]
        assert outputs[0]["effect_status"] == "unknown"
        assert outputs[1]["error"]["code"] == "CANCELLED"


@pytest.mark.asyncio
async def test_action_limit_provides_result_for_every_model_call(config):
    config.max_actions_per_step = 1
    fake = FakeProvider(
        Completion(
            tool_calls=[
                ToolCall("one", "calculator", '{"expression":"1+1"}'),
                ToolCall("two", "calculator", '{"expression":"2+2"}'),
            ]
        ),
        Completion(text="limited actions"),
    )
    async with Runtime(config, provider=fake) as runtime:
        await runtime.ask("Calculate two things")
        validate_pairing(runtime.state.messages)
        outputs = [json.loads(m["content"]) for m in runtime.state.messages if m["role"] == "tool"]
        assert outputs[0]["ok"]
        assert outputs[1]["error"]["code"] == "ACTION_LIMIT"


@pytest.mark.asyncio
async def test_failing_ui_observer_does_not_replay_tool(config):
    async def broken_ui(event):
        raise RuntimeError("UI disconnected")

    fake = FakeProvider(tool("calculator", {"expression": "2+2"}), Completion(text="4"))
    async with Runtime(config, provider=fake, on_event=broken_ui) as runtime:
        answer = await runtime.ask("2+2?")
        assert answer["text"] == "4"
        assert len(runtime.state.actions) == 1
        assert any(e["type"] == "observer.failed" for e in runtime.store.events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "summary", ["No required headings", "[用户任务]\n" + "x" * 3000 + "\n[工作状态]"]
)
async def test_invalid_compact_cannot_replace_history(config, summary):
    fake = FakeProvider(Completion(text="Important state"), Completion(text=summary))
    async with Runtime(config, provider=fake) as runtime:
        await runtime.ask("Keep this requirement")
        original = deepcopy(runtime.state.messages)
        answer = await runtime.compact()
        assert answer["status"] == "failed"
        assert runtime.state.messages == original
        assert runtime.state.epoch == 0
