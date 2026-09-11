import json

import httpx
import pytest

from miniharness.models import ProviderConfig
from miniharness.providers import HTTPProvider, ProviderError, normalize_usage

SPEC = {
    "name": "calculator",
    "description": "Calculate",
    "parameters": {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
        "additionalProperties": False,
    },
}
CALL = {"id": "call1", "name": "calculator", "arguments": '{"expression":"2+2"}'}


def payload(protocol, tool=False):
    if protocol == "openai_chat":
        message = {"role": "assistant", "content": "Answer"}
        if tool:
            message["tool_calls"] = [
                {
                    "id": "call1",
                    "type": "function",
                    "function": {"name": CALL["name"], "arguments": CALL["arguments"]},
                }
            ]
        return {
            "choices": [{"message": message, "finish_reason": "tool_calls" if tool else "stop"}],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 10},
            },
        }
    if protocol == "anthropic_messages":
        blocks = [
            {"type": "thinking", "thinking": "Summary", "signature": "signed"},
            {"type": "text", "text": "Answer"},
        ]
        if tool:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": "call1",
                    "name": CALL["name"],
                    "input": json.loads(CALL["arguments"]),
                }
            )
        return {
            "content": blocks,
            "stop_reason": "tool_use" if tool else "end_turn",
            "usage": {
                "input_tokens": 10,
                "cache_read_input_tokens": 8,
                "cache_creation_input_tokens": 2,
                "output_tokens": 4,
            },
        }
    items = [
        {
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "opaque",
            "summary": [{"type": "summary_text", "text": "Summary"}],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Answer"}],
        },
    ]
    if tool:
        items.append(
            {
                "type": "function_call",
                "id": "fc_different",
                "call_id": "call1",
                "name": CALL["name"],
                "arguments": CALL["arguments"],
            }
        )
    return {
        "output": items,
        "status": "completed",
        "usage": {
            "input_tokens": 20,
            "output_tokens": 4,
            "input_tokens_details": {"cached_tokens": 10},
        },
    }


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "secret-never-echo")


def provider(protocol, handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return HTTPProvider(
        ProviderConfig(
            protocol=protocol, model="test", api_key_env="TEST_PROVIDER_KEY", max_retries=0
        ),
        client,
    ), client


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["openai_chat", "anthropic_messages", "openai_responses"])
@pytest.mark.parametrize("tool", [False, True])
async def test_complete_roundtrip(protocol, tool):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=payload(protocol, tool))

    adapter, client = provider(protocol, handler)
    async with client:
        completion = await adapter.complete([{"role": "user", "content": "hi"}], [SPEC], "system")
        assert completion.text == "Answer"
        assert completion.usage["input_tokens_total"] == 20
        assert completion.usage["output_tokens"] == 4
        assert len(completion.tool_calls) == int(tool)
        if tool:
            assert completion.tool_calls[0].id == "call1"
            messages = [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": completion.text,
                    "tool_calls": [CALL],
                    "provider_payload": completion.provider_payload,
                },
                {"role": "tool", "tool_call_id": "call1", "content": "4"},
                {"role": "user", "content": "continue"},
            ]
            before = json.dumps(messages)
            await adapter.complete(messages, [SPEC], "system")
            assert json.dumps(messages) == before
            encoded = json.dumps(requests[-1])
            assert "call1" in encoded
            if protocol != "openai_chat":
                assert "signed" in encoded or "opaque" in encoded


def sse(events):
    return "".join("data: " + (e if isinstance(e, str) else json.dumps(e)) + "\n\n" for e in events)


@pytest.mark.asyncio
async def test_chat_fragmented_tools_and_usage():
    events = []
    for i, char in enumerate(CALL["arguments"]):
        events.append(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call1" if i == 0 else "",
                                    "function": {
                                        "name": "calculator" if i == 0 else "",
                                        "arguments": char,
                                    },
                                }
                            ]
                        },
                    }
                ]
            }
        )
    events += [
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3}},
        "[DONE]",
    ]
    adapter, client = provider("openai_chat", lambda _: httpx.Response(200, text=sse(events)))
    deltas = []

    async def collect(event):
        deltas.append(event)

    async with client:
        result = await adapter.complete([], [SPEC], "system", stream=True, on_delta=collect)
    assert result.tool_calls[0].arguments == CALL["arguments"]
    assert result.usage["input_tokens_total"] == 10
    assert len(deltas) == len(events) - 1


@pytest.mark.asyncio
async def test_anthropic_stream_cumulative_usage_signature():
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 10, "output_tokens": 1}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "brief"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "call1", "name": "calculator", "input": {}},
        },
    ]
    events += [
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": char},
        }
        for char in CALL["arguments"]
    ]
    events += [
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 8},
        },
        {"type": "message_stop"},
    ]
    adapter, client = provider(
        "anthropic_messages", lambda _: httpx.Response(200, text=sse(events))
    )
    async with client:
        result = await adapter.complete([], [SPEC], "system", stream=True)
    assert result.usage["output_tokens"] == 8
    assert result.provider_payload[0]["signature"] == "sig"
    assert json.loads(result.tool_calls[0].arguments) == json.loads(CALL["arguments"])


@pytest.mark.asyncio
async def test_responses_completed_is_authoritative():
    raw = payload("openai_responses", True)
    events = [
        {"type": "response.output_text.delta", "delta": "Answer"},
        {"type": "response.completed", "response": raw},
    ]
    adapter, client = provider("openai_responses", lambda _: httpx.Response(200, text=sse(events)))
    async with client:
        result = await adapter.complete([], [SPEC], "system", stream=True)
    assert result.tool_calls[0].id == "call1"
    assert result.provider_payload == raw["output"]


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["openai_chat", "anthropic_messages", "openai_responses"])
async def test_truncation_is_not_completion(protocol):
    raw = payload(protocol)
    if protocol == "openai_chat":
        raw["choices"][0]["finish_reason"] = "length"
    elif protocol == "anthropic_messages":
        raw["stop_reason"] = "max_tokens"
    else:
        raw["status"] = "incomplete"
    adapter, client = provider(protocol, lambda _: httpx.Response(200, json=raw))
    async with client:
        with pytest.raises(ProviderError) as error:
            await adapter.complete([], [], "")
    assert error.value.usage["output_tokens"] == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        "data: {}\n\n",
        'data: {"error":{"message":"secret-never-echo"}}\n\n',
        "data: invalid\n\n",
        "data: [DONE]\n\n",
    ],
)
async def test_bad_chat_stream_never_commits(body):
    adapter, client = provider("openai_chat", lambda _: httpx.Response(200, text=body))
    async with client:
        with pytest.raises(ProviderError) as error:
            await adapter.complete([], [], "", stream=True)
    assert "secret-never-echo" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "tool", "tool_call_id": "x", "content": "orphan"}],
        [{"role": "assistant", "tool_calls": [CALL]}],
        [{"role": "assistant", "tool_calls": [CALL]}, {"role": "user", "content": "interrupted"}],
        [{"role": "assistant", "tool_calls": [CALL, CALL]}],
    ],
)
async def test_invalid_pairs_rejected_before_http(messages):
    def forbidden(_):
        pytest.fail("must reject before HTTP")

    adapter, client = provider("openai_chat", forbidden)
    async with client:
        with pytest.raises(ProviderError):
            await adapter.complete(messages, [], "")


@pytest.mark.asyncio
async def test_auth_errors_do_not_leak_secrets():
    adapter, client = provider(
        "openai_chat", lambda _: httpx.Response(401, text="secret-never-echo")
    )
    async with client:
        with pytest.raises(ProviderError, match="HTTP 401") as error:
            await adapter.complete([], [], "")
    assert "secret-never-echo" not in str(error.value)


def test_missing_usage_is_unknown():
    assert normalize_usage(None, "openai_chat")["input_tokens_total"] is None
    assert normalize_usage({"input_tokens": 8}, "anthropic_messages")["input_tokens_total"] == 8


@pytest.mark.parametrize(
    "protocol,prefix", [("openai_chat", "prompt"), ("openai_responses", "input")]
)
def test_openai_cache_writes_are_input_subset(protocol, prefix):
    raw = {
        f"{prefix}_tokens": 100,
        f"{prefix}_tokens_details": {"cached_tokens": 30, "cache_write_tokens": 20},
    }
    usage = normalize_usage(raw, protocol)
    assert usage["input_tokens_total"] == 100
    assert usage["input_tokens_uncached"] == 50
    assert usage["cache_read_tokens"] == 30
    assert usage["cache_write_tokens"] == 20
    assert usage["raw_usage"] == raw
    del raw[f"{prefix}_tokens_details"]["cache_write_tokens"]
    missing = normalize_usage(raw, protocol)
    assert missing["cache_write_tokens"] is None
    assert missing["input_tokens_uncached"] is None
    assert usage["raw_usage"][f"{prefix}_tokens_details"]["cache_write_tokens"] == 20


@pytest.mark.asyncio
async def test_retry_status_then_success(monkeypatch):
    attempts = []

    async def no_sleep(_):
        pass

    monkeypatch.setattr("miniharness.providers.http.asyncio.sleep", no_sleep)

    def handler(_):
        attempts.append(1)
        return (
            httpx.Response(429, headers={"retry-after": "0"})
            if len(attempts) == 1
            else httpx.Response(200, json=payload("openai_chat"))
        )

    adapter, client = provider("openai_chat", handler)
    adapter.config.max_retries = 1
    async with client:
        assert (await adapter.complete([], [], "")).text == "Answer"
    assert len(attempts) == 2


class FragmentedBody(httpx.AsyncByteStream):
    def __init__(self, data):
        self.data = data.encode("utf-8")

    async def __aiter__(self):
        for byte in self.data:
            yield bytes([byte])


@pytest.mark.asyncio
async def test_network_fragmentation_utf8_and_interleaved_calls():
    events = []
    for index in [1, 0]:
        events.append(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": "你好" if index == 0 else "",
                            "tool_calls": [
                                {
                                    "index": index,
                                    "id": f"call{index}",
                                    "function": {"name": "calculator", "arguments": "{"},
                                }
                            ],
                        },
                    }
                ]
            }
        )
    for index in [0, 1]:
        events.append(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": index, "function": {"arguments": '"expression":"2"}'}}
                            ]
                        },
                    }
                ]
            }
        )
    events += [{"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}, "[DONE]"]
    adapter, client = provider(
        "openai_chat", lambda _: httpx.Response(200, stream=FragmentedBody(sse(events)))
    )
    async with client:
        result = await adapter.complete([], [SPEC], "", stream=True)
    assert result.text == "你好"
    assert [c.id for c in result.tool_calls] == ["call0", "call1"]
    assert all(json.loads(c.arguments) == {"expression": "2"} for c in result.tool_calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events",
    [
        [
            {"type": "message_start", "message": {"usage": {}}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": "hi"},
            },
            {"type": "message_stop"},
        ],
        [{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "x"}}],
        [
            {"type": "message_start", "message": {"usage": {}}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "x", "name": "calculator", "input": {}},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"a":1,"a":2}'},
            },
            {"type": "content_block_stop", "index": 0},
        ],
    ],
)
async def test_anthropic_malformed_blocks_rejected(events):
    adapter, client = provider(
        "anthropic_messages", lambda _: httpx.Response(200, text=sse(events))
    )
    async with client:
        with pytest.raises(ProviderError):
            await adapter.complete([], [], "", stream=True)


@pytest.mark.asyncio
async def test_responses_partial_stream_not_accepted():
    adapter, client = provider(
        "openai_responses",
        lambda _: httpx.Response(
            200,
            text=sse(
                [
                    {
                        "type": "response.output_item.done",
                        "item": {
                            "type": "function_call",
                            "call_id": "x",
                            "name": "calculator",
                            "arguments": "{}",
                        },
                    }
                ]
            ),
        ),
    )
    async with client:
        with pytest.raises(ProviderError, match="terminal"):
            await adapter.complete([], [], "", stream=True)


@pytest.mark.asyncio
async def test_extras_cannot_mutate_history():
    adapter, client = provider("openai_chat", lambda _: pytest.fail("no request allowed"))
    adapter.config.extra = {"messages": []}
    async with client:
        with pytest.raises(ProviderError, match="override"):
            await adapter.complete([], [], "")


@pytest.mark.asyncio
async def test_missing_key_fails_locally(monkeypatch):
    monkeypatch.delenv("TEST_PROVIDER_KEY")
    adapter, client = provider("openai_chat", lambda _: pytest.fail("no request allowed"))
    async with client:
        with pytest.raises(ProviderError, match="TEST_PROVIDER_KEY"):
            await adapter.complete([], [], "")


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["openai_chat", "openai_responses", "anthropic_messages"])
async def test_strict_optional_nulls_compile_and_normalize_without_mutating_history(protocol):
    schema = {
        "type": "object",
        "properties": {
            "required": {"type": "string"},
            "optional": {"type": "integer"},
            "nullable": {"type": ["string", "null"]},
            "nested": {
                "type": "object",
                "properties": {"optional": {"type": "string"}},
                "additionalProperties": False,
            },
        },
        "required": ["required"],
        "additionalProperties": False,
    }
    spec = {"name": "calculator", "description": "test", "parameters": schema}
    original = json.dumps(spec)
    arguments = {"required": None, "optional": None, "nullable": None, "nested": {"optional": None}}
    raw = payload(protocol, True)
    if protocol == "openai_chat":
        raw["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps(
            arguments
        )
    elif protocol == "openai_responses":
        raw["output"][-1]["arguments"] = json.dumps(arguments)
    else:
        raw["content"][-1]["input"] = arguments
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=raw)

    adapter, client = provider(protocol, handler)
    adapter.config.strict_schema = True
    async with client:
        result = await adapter.complete([], [spec], "")
    assert json.dumps(spec) == original
    wire = requests[0]["tools"][0]
    compiled = (
        wire["function"]["parameters"]
        if protocol == "openai_chat"
        else wire["parameters"]
        if protocol == "openai_responses"
        else wire["input_schema"]
    )
    assert set(compiled["required"]) == set(schema["properties"])
    assert compiled["properties"]["optional"]["anyOf"][-1] == {"type": "null"}
    assert json.loads(result.tool_calls[0].arguments) == {
        "required": None,
        "nullable": None,
        "nested": {},
    }
    assert '"optional": null' in json.dumps(result.provider_payload).replace('\\"', '"')
    # Required null survives adapter normalization for the registry's typed error path.
    from jsonschema import Draft202012Validator

    assert not Draft202012Validator(schema).is_valid(json.loads(result.tool_calls[0].arguments))


@pytest.mark.asyncio
async def test_failed_attempt_usage_is_preserved_and_aggregated(monkeypatch):
    async def no_sleep(_):
        pass

    monkeypatch.setattr("miniharness.providers.http.asyncio.sleep", no_sleep)
    attempts = []
    events = []

    async def collect(event):
        events.append(event)

    def handler(_):
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(
                503, json={"error": "busy", "usage": {"prompt_tokens": 7, "completion_tokens": 2}}
            )
        return httpx.Response(200, json=payload("openai_chat"))

    adapter, client = provider("openai_chat", handler)
    adapter.config.max_retries = 1
    async with client:
        result = await adapter.complete([], [], "", on_delta=collect)
    assert result.usage["input_tokens_total"] == 27
    assert result.usage["output_tokens"] == 6
    assert result.usage["failed_attempts"][0]["usage"]["raw_usage"]["prompt_tokens"] == 7
    assert events[0]["type"] == "provider_attempt_failed"
    assert "cache_read_tokens" in result.usage["incomplete_fields"]


@pytest.mark.asyncio
async def test_unknown_retry_usage_not_claimed_as_zero(monkeypatch):
    async def no_sleep(_):
        pass

    monkeypatch.setattr("miniharness.providers.http.asyncio.sleep", no_sleep)
    attempts = []

    def handler(_):
        attempts.append(1)
        return (
            httpx.Response(429, json={"error": "busy"})
            if len(attempts) == 1
            else httpx.Response(200, json=payload("openai_chat"))
        )

    adapter, client = provider("openai_chat", handler)
    adapter.config.max_retries = 1
    async with client:
        result = await adapter.complete([], [], "")
    assert result.usage["input_tokens_total"] == 20
    assert result.usage["failed_attempts"][0]["usage"]["input_tokens_total"] is None
    assert "input_tokens_total" in result.usage["incomplete_fields"]
