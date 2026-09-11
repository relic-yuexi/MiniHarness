"""Opt-in real API acceptance tests. These tests never substitute a mock provider.

Run with MINIHARNESS_LIVE=1 and optionally MINIHARNESS_CONFIG=/path/config.toml.
The API key must be in the environment named by provider.api_key_env. Each test
uses isolated temporary workspace/session directories, never your saved sessions.
"""

import os
from copy import deepcopy
from pathlib import Path

import pytest

from miniharness.config import load_config
from miniharness.context import validate_pairing
from miniharness.providers import HTTPProvider
from miniharness.runtime import Runtime

pytestmark = pytest.mark.live


@pytest.fixture
def live_config(tmp_path):
    if os.environ.get("MINIHARNESS_LIVE") != "1":
        pytest.skip("Real LLM API test disabled: set MINIHARNESS_LIVE=1 to opt in")
    path = Path(os.environ.get("MINIHARNESS_CONFIG", "config.toml"))
    if not path.is_file():
        pytest.skip("Real LLM API test needs an existing MINIHARNESS_CONFIG or config.toml")
    config = load_config(path)
    if not os.environ.get(config.provider.api_key_env):
        pytest.skip(f"Real LLM API credential missing: set {config.provider.api_key_env}")
    config.workspace = tmp_path / "workspace"
    config.session_root = tmp_path / "sessions"
    config.workspace.mkdir()
    config.keep_recent_turns = 0
    config.max_session_tokens = 0
    return config


def successful_actions(runtime, name):
    return [
        action
        for action in runtime.state.actions.values()
        if action.get("call", {}).get("name") == name and action.get("result", {}).get("ok")
    ]


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
async def test_live_direct_calculator_and_followup(live_config, stream):
    live_config.stream = stream
    events = []

    async def observe(event):
        events.append(event)

    async with Runtime(live_config, on_event=observe) as runtime:
        assert isinstance(runtime.provider, HTTPProvider)
        direct = await runtime.ask(
            "Reply briefly to say hello. Do not use any tools for this greeting."
        )
        assert direct["status"] == "completed", direct
        assert direct["text"].strip()
        assert not runtime.state.actions

        calculated = await runtime.ask(
            "Use the calculator tool to calculate 47 * 23. Report the result."
        )
        assert calculated["status"] == "completed", calculated
        values = [
            action["result"]["data"]["value"]
            for action in successful_actions(runtime, "calculator")
        ]
        assert "1081" in values, "Expected actual calculator execution, not mental arithmetic"
        prior_actions = len(successful_actions(runtime, "calculator"))
        followup = await runtime.ask("Now add 19 to that previous result using calculator again.")
        assert followup["status"] == "completed", followup
        calls = successful_actions(runtime, "calculator")
        assert len(calls) > prior_actions
        assert any(action["result"]["data"]["value"] == "1100" for action in calls[prior_actions:])
        validate_pairing(runtime.state.messages)
        assert runtime.state.usage, "A real API attempt must have recorded usage metadata"
        if stream:
            assert any(event.get("type") in {"provider_delta", "delta"} for event in events)


async def test_live_todo_restart_compact_and_pure_followup(live_config):
    live_config.stream = False
    async with Runtime(live_config, user_id="live-test") as runtime:
        response = await runtime.ask(
            "Use todo to replace my todo list with exactly one item: id=live-item, "
            "text=Review the agent runtime, status=pending. The initial revision is zero. "
            "Also remember my project code is COPPER-ORCHID-74 for a later question."
        )
        assert response["status"] == "completed", response
        assert successful_actions(runtime, "todo"), "A real tool call must update todo"
        assert runtime.state.todo["items"] == [
            {"id": "live-item", "text": "Review the agent runtime", "status": "pending"}
        ]
        session_id = runtime.session_id
        todo_before = deepcopy(runtime.state.todo)

    # Construct a fresh runtime: memory must come from JSONL replay, not the old object.
    async with Runtime(live_config, session_id=session_id, user_id="live-test") as resumed:
        assert resumed.state.todo == todo_before
        compacted = await resumed.compact()
        assert compacted["status"] == "completed", compacted
        assert resumed.state.epoch >= 1
        assert resumed.state.todo == todo_before
        answer = await resumed.ask(
            "Without using tools, what is my project code and what pending task did I save?"
        )
        assert answer["status"] == "completed", answer
        assert "COPPER-ORCHID-74" in answer["text"]
        assert resumed.state.todo == todo_before
        validate_pairing(resumed.state.messages)
