"""Named middleware ordering, transactional builders and immutable request prefixes."""

import asyncio
import json
from copy import deepcopy

import pytest
from test_runtime import FakeProvider

from miniharness.config import Config
from miniharness.hooks import Hooks
from miniharness.models import Completion, ProviderConfig, ToolCall
from miniharness.runtime import Runtime
from miniharness.tools import ToolRegistry, result


@pytest.fixture
def config(tmp_path):
    return Config(
        provider=ProviderConfig(model="fake"),
        session_root=tmp_path / "sessions",
        workspace=tmp_path / "workspace",
        compact_target_tokens=500,
    )


def append_callback(value, order):
    async def callback(data):
        order.append(value)

    return callback


async def test_named_priority_fifo_and_global_replacement():
    hooks, order = Hooks(), []
    hooks.register("first", append_callback("old", order), position="pre_action")
    hooks.register("second", append_callback("second", order), position="pre_action")
    hooks.register("first", append_callback("new", order), position="pre_action")
    hooks.register("high", append_callback("high", order), position="pre_action", priority=10)
    await hooks.run("pre_action", {})
    assert order == ["high", "new", "second"]
    order.clear()
    hooks.register("first", append_callback("moved", order), position="post_action")
    await hooks.run("pre_action", {})
    await hooks.run("post_action", {})
    assert order == ["high", "second", "moved"]


async def test_custom_registration_wins_over_builtin_in_either_order():
    for custom_first in (True, False):
        hooks, order = Hooks(), []
        for builtin in (False, True) if custom_first else (True, False):
            hooks.register(
                "tool_schema",
                append_callback("builtin" if builtin else "custom", order),
                position="post_system_prompt",
                builtin=builtin,
            )
        await hooks.run("post_system_prompt", {})
        assert order == ["custom"]


async def test_position_aliases_and_post_priority():
    hooks, order = Hooks(), []
    hooks.register("one", append_callback(1, order), position="preSystemPromptHook")
    hooks.register("two", append_callback(2, order), position="pre_system_prompt_hook")
    await hooks.run("pre_system_prompt", {})
    assert order == [1, 2]
    hooks.register("post-low", append_callback(3, order), position="post_action")
    hooks.register("post-high", append_callback(4, order), position="post_action", priority=2)
    async with hooks.scope("action", {}):
        pass
    assert order == [1, 2, 4, 3]


async def test_queue_snapshot_defers_registration_until_next_run():
    hooks, order = Hooks(), []

    async def install(data):
        order.append("install")
        hooks.register("late", append_callback("late", order), position="pre_action")

    hooks.register("install", install, position="pre_action")
    await hooks.run("pre_action", {})
    assert order == ["install"]
    await hooks.run("pre_action", {})
    assert order == ["install", "install", "late"]


async def test_legacy_add_retains_reverse_paired_unwind():
    hooks, order = Hooks(), []
    for name in ("a", "b"):
        hooks.add(
            "action",
            pre=append_callback(name + "+", order),
            post=append_callback(name + "-", order),
        )
    async with hooks.scope("action", {}):
        order.append("body")
    assert order == ["a+", "b+", "body", "b-", "a-"]


async def test_legacy_failure_does_not_enter_later_post_only_layer():
    hooks, order = Hooks(), []

    async def reject(data):
        raise ValueError("guard rejected")

    hooks.add("action", post=append_callback("entered", order))
    hooks.add("action", pre=reject)
    hooks.add("action", post=append_callback("not-entered", order))
    with pytest.raises(ValueError):
        async with hooks.scope("action", {}):
            pytest.fail("guard must reject")
    assert order == ["entered"]


async def test_mutable_pipeline_commits_only_successful_callbacks():
    hooks = Hooks()
    original = {"nested": {"items": []}, "system": "base"}

    async def first(data):
        data["nested"]["items"].append("first")
        return {"system": "changed"}

    async def failure(data):
        data["nested"]["items"].append("discard")
        raise ValueError("observer failed")

    async def last(data):
        assert data["system"] == "changed"
        assert data["nested"]["items"] == ["first"]
        data["nested"]["items"].append("last")

    hooks.register("first", first, position="pre_system_prompt")
    hooks.register("bad", failure, position="pre_system_prompt", guard=False)
    hooks.register("last", last, position="pre_system_prompt")
    built = await hooks.run("pre_system_prompt", original, mutable=True)
    assert built == {"nested": {"items": ["first", "last"]}, "system": "changed"}
    assert original == {"nested": {"items": []}, "system": "base"}


@pytest.mark.parametrize("failure", ["exception", "timeout", "cancel"])
async def test_builder_guard_failure_leaves_original_untouched(failure):
    hooks = Hooks(timeout=0.01)
    original = {"items": []}

    async def bad(data):
        data["items"].append("must not leak")
        if failure == "timeout":
            await asyncio.Event().wait()
        if failure == "cancel":
            raise asyncio.CancelledError
        raise ValueError("blocked")

    hooks.register("bad", bad, position="pre_system_prompt")
    expected = {"exception": ValueError, "timeout": TimeoutError, "cancel": asyncio.CancelledError}[
        failure
    ]
    with pytest.raises(expected):
        await hooks.run("pre_system_prompt", original, mutable=True)
    assert original == {"items": []}


async def test_observer_mode_cannot_change_input_or_later_observer():
    hooks = Hooks()
    original = {"items": []}

    async def mutate(data):
        data["items"].append("ignored")
        return {"items": ["also ignored"]}

    async def observe(data):
        assert data == original

    hooks.register("mutate", mutate, position="pre_action")
    hooks.register("observe", observe, position="pre_action")
    assert await hooks.run("pre_action", original) == original
    assert original == {"items": []}


def test_freeze_prevents_replace_move_and_add_but_copy_is_independent():
    hooks = Hooks()
    callback = append_callback(None, [])
    hooks.register("fixed", callback, position="pre_system_prompt")
    clone = hooks.copy()
    hooks.freeze_positions({"pre_system_prompt", "post_system_prompt"})
    for name, position in [
        ("fixed", "pre_action"),
        ("new", "pre_system_prompt"),
        ("fixed", "pre_system_prompt"),
    ]:
        with pytest.raises((ValueError, RuntimeError)):
            hooks.register(name, callback, position=position)
    clone.register("fixed", callback, position="pre_action")
    hooks.register("other", callback, position="pre_action")


async def test_runtime_uses_frozen_custom_prefix_and_copies_blueprint(config):
    hooks = Hooks()
    calls = []

    async def before(data):
        calls.append("pre")
        data["system"] += "\nCUSTOM"

    async def replace_schema(data):
        calls.append("post")
        data["tools"] = []

    hooks.register("before", before, position="pre_system_prompt")
    hooks.register("tool_schema", replace_schema, position="post_system_prompt")
    fake = FakeProvider(Completion(text="one"), Completion(text="two"))
    async with Runtime(config, provider=fake, hooks=hooks) as runtime:
        await runtime.ask("hello")
        history = deepcopy(runtime.state.messages)
        hooks.register("before", append_callback("unused", calls), position="pre_system_prompt")
        await runtime.ask("continue")
        assert runtime.state.messages[: len(history)] == history
        assert calls == ["pre", "post"]
        assert runtime.system_prompt == config.system + "\nCUSTOM"
        assert runtime.tool_schemas == []
        assert all(
            call["system"] == runtime.system_prompt and call["tools"] == [] for call in fake.calls
        )


async def test_effective_prefix_participates_in_restore_fingerprint(config):
    def blueprint(suffix):
        hooks = Hooks()

        async def custom(data):
            data["system"] += suffix

        hooks.register("custom", custom, position="post_system_prompt")
        return hooks

    async with Runtime(config, provider=FakeProvider(), hooks=blueprint("A")) as runtime:
        session_id = runtime.session_id
    async with Runtime(
        config, provider=FakeProvider(), hooks=blueprint("A"), session_id=session_id
    ):
        pass
    incompatible = Runtime(
        config, provider=FakeProvider(), hooks=blueprint("B"), session_id=session_id
    )
    with pytest.raises(ValueError):
        await incompatible.start()
    await incompatible.close()


async def test_exposed_schemas_are_independent_copies(config):
    async with Runtime(config, provider=FakeProvider(Completion(text="ok"))) as runtime:
        expected = runtime.tool_schemas
        modified = runtime.tool_schemas
        modified[0]["name"] = "corrupted"
        modified.clear()
        await runtime.ask("hello")
        assert runtime.tool_schemas == expected
        assert runtime.provider.calls[0]["tools"] == expected


async def test_compact_uses_no_tools_and_next_turn_restores_frozen_prefix(config):
    hooks = Hooks()
    builds = []

    async def custom(data):
        builds.append(True)
        data["system"] += "\nCustom stable system"

    hooks.register("custom", custom, position="post_system_prompt")
    fake = FakeProvider(
        Completion(text="First answer"),
        Completion(text="[用户任务]\n目标：记住名字。\n[工作状态]\n已完成及结果：用户名字为小明。"),
        Completion(text="小明"),
    )
    async with Runtime(config, hooks=hooks, provider=fake) as runtime:
        await runtime.ask("我的名字是小明")
        assert (await runtime.compact())["status"] == "completed"
        await runtime.ask("我的名字是什么？")
        assert builds == [True]
        assert fake.calls[1]["tools"] == []
        assert fake.calls[0]["tools"] == fake.calls[2]["tools"] == runtime.tool_schemas
        assert fake.calls[0]["system"] == fake.calls[2]["system"] == runtime.system_prompt


async def test_custom_schema_restriction_is_enforced_before_handler(config):
    hooks, registry, executed = Hooks(), ToolRegistry(), []

    async def handler(arguments, context):
        executed.append(arguments)
        return result("executed")

    registry.register(
        "bounded",
        "Handle integer",
        {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]},
        handler,
    )

    async def restrict(data):
        data["tools"][0]["parameters"]["properties"]["value"]["maximum"] = 10

    hooks.register("restrict", restrict, position="post_system_prompt", priority=-1)
    fake = FakeProvider(
        Completion(tool_calls=[ToolCall("over", "bounded", '{"value": 11}')]),
        Completion(tool_calls=[ToolCall("valid", "bounded", '{"value": 9}')]),
        Completion(text="done"),
    )
    async with Runtime(config, provider=fake, hooks=hooks, registry=registry) as runtime:
        assert (await runtime.ask("Do work"))["status"] == "completed"
        outputs = [
            json.loads(message["content"])
            for message in runtime.state.messages
            if message["role"] == "tool"
        ]
        assert outputs[0]["error"]["code"] == "INVALID_ARGUMENT"
        assert outputs[1]["ok"]
        assert executed == [{"value": 9}]


@pytest.mark.parametrize(
    "invalid",
    [
        "not a list",
        [{"name": "missing_handler", "description": "bad", "parameters": {"type": "object"}}],
    ],
)
async def test_invalid_schema_builder_fails_before_provider_call(config, invalid):
    hooks = Hooks()

    async def custom(data):
        data["tools"] = invalid

    hooks.register("tool_schema", custom, position="post_system_prompt")
    fake = FakeProvider()
    runtime = Runtime(config, provider=fake, hooks=hooks)
    with pytest.raises((ValueError, TypeError)):
        await runtime.start()
    assert fake.calls == []
    await runtime.close()


def test_legacy_pair_registration_is_atomic():
    hooks = Hooks()

    async def valid(data):
        pass

    with pytest.raises(TypeError):
        hooks.add("action", pre=valid, post=lambda data: None)
    assert hooks.entries("pre_action") == ()
    assert hooks.entries("post_action") == ()
