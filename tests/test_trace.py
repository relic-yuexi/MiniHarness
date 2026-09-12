from copy import deepcopy

from miniharness.trace import stream_trace


def test_response_snapshot_projection_preserves_evidence_without_mutation():
    source = {
        "protocol": "openai_responses",
        "event": {
            "type": "response.completed",
            "response": {
                "id": "r1",
                "status": "completed",
                "instructions": "x" * 10000,
                "tools": [{"name": "tool"}],
                "output": [{"text": "answer"}],
                "usage": {"input_tokens": 20},
            },
        },
    }
    before = deepcopy(source)
    trace = stream_trace(source)
    assert source == before
    assert trace["event"]["response"] == {
        "id": "r1",
        "status": "completed",
        "usage": {"input_tokens": 20},
    }
    assert trace["trace_projection"]["original_bytes"] > 10000
    assert len(trace["trace_projection"]["original_hash"]) == 64


def test_delta_preserved_and_done_snapshot_reduced():
    delta = {
        "protocol": "openai_responses",
        "event": {
            "type": "response.function_call_arguments.delta",
            "delta": "{",
            "item_id": "i",
            "sequence_number": 2,
        },
    }
    assert stream_trace(delta) == delta
    done = {
        "protocol": "openai_responses",
        "event": {
            "type": "response.output_item.done",
            "item": {
                "id": "i",
                "type": "function_call",
                "name": "read",
                "arguments": "large",
                "call_id": "c",
            },
        },
    }
    projected = stream_trace(done)
    assert projected["event"]["item"] == {
        "id": "i",
        "type": "function_call",
        "name": "read",
        "call_id": "c",
    }
    assert done["event"]["item"]["arguments"] == "large"


def test_other_protocols_and_unknown_envelopes_unchanged():
    for event in (
        {"event": None},
        {
            "protocol": "anthropic_messages",
            "event": {"type": "content_block_delta", "delta": {"text": "hello"}},
        },
    ):
        assert stream_trace(event) == event
