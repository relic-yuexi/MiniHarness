"""Regression cases discovered during independent runtime boundary review."""

import asyncio
import json

import pytest
from test_runtime import FakeProvider

from miniharness.config import Config
from miniharness.hooks import Hooks
from miniharness.models import Completion, ProviderConfig, ToolCall
from miniharness.runtime import Runtime
from miniharness.tools import ToolRegistry, result


@pytest.mark.asyncio
async def test_compact_summarizes_readable_history_not_opaque_reasoning(config):
    fake = FakeProvider(
        Completion(
            text="Visible fact: 42",
            provider_payload={"encrypted": "OPAQUE_SECRET"},
            reasoning="PRIVATE_REASONING",
        ),
        Completion(text="[用户任务]\nRemember 42\n[工作状态]\nFact is 42"),
    )
    async with Runtime(config, provider=fake) as runtime:
        await runtime.ask("Remember the result")
        before = json.dumps(runtime.store.events)
        assert "OPAQUE_SECRET" in before
        result = await runtime.compact()
        assert result["status"] == "completed"
        summary_input = json.dumps(fake.calls[-1]["messages"])
        assert "Visible fact: 42" in summary_input
        assert "OPAQUE_SECRET" not in summary_input
        assert "PRIVATE_REASONING" not in summary_input
        assert "OPAQUE_SECRET" in json.dumps(runtime.store.events)


@pytest.fixture
def config(tmp_path):
    return Config(
        provider=ProviderConfig(model="fake"),
        session_root=tmp_path / "sessions",
        workspace=tmp_path / "workspace",
        compact_target_tokens=500,
    )


@pytest.mark.asyncio
async def test_post_turn_hook_observes_durable_terminal_state(config):
    observed = []
    hooks = Hooks()
    runtime = Runtime(config, provider=FakeProvider(Completion(text="done")), hooks=hooks)

    async def post(data):
        observed.append(
            (
                runtime.state.current_turn,
                any(e["type"] == "turn.ended" for e in runtime.store.events),
            )
        )

    hooks.add("turn", post=post)
    async with runtime:
        await runtime.ask("hello")
    assert observed == [(None, True)]


@pytest.mark.asyncio
async def test_post_step_hook_observes_durable_terminal_state(config):
    observed = []
    hooks = Hooks()
    runtime = Runtime(config, provider=FakeProvider(Completion(text="done")), hooks=hooks)

    async def post(data):
        observed.append(
            any(
                e["type"] == "step.ended" and e["step_id"] == data["step_id"]
                for e in runtime.store.events
            )
        )

    hooks.add("step", post=post)
    async with runtime:
        await runtime.ask("hello")
    assert observed == [True]


@pytest.mark.asyncio
async def test_manual_compact_respects_spent_session_budget(config):
    config.max_session_tokens = 10
    fake = FakeProvider(
        Completion(text="done", usage={"input_tokens_total": 11, "output_tokens": 1}),
        Completion(text="[用户任务]\nhello\n[工作状态]\ndone"),
    )
    async with Runtime(config, provider=fake) as runtime:
        await runtime.ask("hello")
        answer = await runtime.compact()
        assert answer["status"] in {"failed", "limited"}
        assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_length_limited_response_still_records_billed_usage(config):
    fake = FakeProvider(
        Completion(
            text="unfinished",
            finish_reason="length",
            usage={"input_tokens_total": 10, "output_tokens": 8},
        )
    )
    async with Runtime(config, provider=fake) as runtime:
        answer = await runtime.ask("hello")
        assert answer["status"] == "failed"
        assert runtime.state.total_tokens == 18
        assert not any(m["role"] == "assistant" for m in runtime.state.messages)


@pytest.mark.asyncio
async def test_steer_arriving_during_boundary_compact_stays_in_turn(config):
    first_entered, first_release = asyncio.Event(), asyncio.Event()
    compact_entered, compact_release = asyncio.Event(), asyncio.Event()

    async def first(*args, **kwargs):
        first_entered.set()
        await first_release.wait()
        return Completion(text="original answer")

    async def summary(*args, **kwargs):
        compact_entered.set()
        await compact_release.wait()
        return Completion(text="[用户任务]\nOriginal task\n[工作状态]\nOriginal answer provided")

    fake = FakeProvider(first, summary, Completion(text="steered answer"))
    async with Runtime(config, provider=fake) as runtime:
        original = await runtime.submit("Start")
        await asyncio.wait_for(first_entered.wait(), 5)
        compact = await runtime.submit("", mode="compact")
        first_release.set()
        await asyncio.wait_for(compact_entered.wait(), 5)
        steering = await runtime.steer("Add this requirement")
        compact_release.set()
        a, b, _ = await asyncio.wait_for(
            asyncio.gather(original.wait(), steering.wait(), compact.wait()), 5
        )
        assert a["turn_id"] == b["turn_id"]


@pytest.mark.asyncio
async def test_concurrent_start_enters_session_hooks_once(config):
    entered, release, contender = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = []
    hooks = Hooks()

    async def pre(data):
        calls.append("pre")
        entered.set()
        await release.wait()

    hooks.add("session", pre=pre)
    runtime = Runtime(config, provider=FakeProvider(), hooks=hooks)
    one = asyncio.create_task(runtime.start())
    await asyncio.wait_for(entered.wait(), 5)

    async def other():
        contender.set()
        return await runtime.start()

    two = asyncio.create_task(other())
    await asyncio.wait_for(contender.wait(), 5)
    release.set()
    try:
        await asyncio.wait_for(asyncio.gather(one, two), 5)
        assert calls == ["pre"]
        assert len([e for e in runtime.store.events if e["type"] == "session.resumed"]) == 1
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_close_rejects_ingress_before_awaiting_session_post_hook(config):
    entered, release = asyncio.Event(), asyncio.Event()
    hooks = Hooks()

    async def post(data):
        entered.set()
        await release.wait()

    hooks.add("session", post=post)
    runtime = await Runtime(
        config, provider=FakeProvider(Completion(text="too late")), hooks=hooks
    ).start()
    closing = asyncio.create_task(runtime.close())
    await asyncio.wait_for(entered.wait(), 5)
    try:
        with pytest.raises(RuntimeError):
            await runtime.submit("Arrives after closing began")
    finally:
        release.set()
        await asyncio.wait_for(closing, 5)


@pytest.mark.asyncio
async def test_turn_timeout_cannot_leave_acknowledged_background_job_unlaunched(
    config, monkeypatch
):
    config.turn_timeout = 1.0
    config.hook_timeout = 5
    launched, post_entered = asyncio.Event(), asyncio.Event()
    hooks, registry = Hooks(), ToolRegistry()

    async def background(args, ctx):
        response = result("accepted")
        response["_job"] = {"command": "dummy"}
        return response

    async def post(data):
        post_entered.set()
        await asyncio.Event().wait()

    async def worker(descriptor, ctx):
        launched.set()
        return result("done")

    registry.register("background", "Start background job", {"type": "object"}, background)
    hooks.add("action", post=post)
    monkeypatch.setattr("miniharness.runtime.run_bash_job", worker)
    fake = FakeProvider(Completion(tool_calls=[ToolCall("job-call", "background", "{}")]))
    async with Runtime(config, provider=fake, hooks=hooks, registry=registry) as runtime:
        ticket = await runtime.submit("Start background work")
        await asyncio.wait_for(post_entered.wait(), 5)
        await asyncio.wait_for(ticket.wait(), 5)
        await runtime.wait_idle(include_jobs=True)
        assert runtime.state.jobs
        assert launched.is_set() or all(
            job["status"] in {"unknown", "failed", "cancelled"}
            for job in runtime.state.jobs.values()
        )
