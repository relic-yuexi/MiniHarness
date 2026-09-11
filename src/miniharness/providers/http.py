"""Three explicit wire protocols with atomic (terminal-only) completions."""

import asyncio
import copy
import json
import os
from typing import Any

import httpx
from jsonschema import Draft202012Validator

from miniharness.models import Completion, DeltaCallback, ProviderConfig, ToolCall


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, usage: dict | None = None):
        super().__init__(message)
        self.usage = usage


def _json_loads(value: str | bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = item
        return result

    def constant(value: str) -> None:
        raise ValueError("Nonfinite JSON constant")

    return json.loads(value, object_pairs_hook=pairs, parse_constant=constant)


def _strict_schema(schema: dict) -> dict:
    """Compile a copy; nullable placeholders never mutate the registry contract."""
    result = copy.deepcopy(schema)
    if result.get("type") == "object":
        required = set(result.get("required", []))
        properties = {}
        for name, child in result.get("properties", {}).items():
            compiled = _strict_schema(child)
            if name not in required and not Draft202012Validator(child).is_valid(None):
                compiled = {"anyOf": [compiled, {"type": "null"}]}
            properties[name] = compiled
        result.update(properties=properties, required=list(properties), additionalProperties=False)
    if isinstance(result.get("items"), dict):
        result["items"] = _strict_schema(result["items"])
    for keyword in ("anyOf", "oneOf", "allOf"):
        if keyword in result:
            result[keyword] = [_strict_schema(child) for child in result[keyword]]
    return result


def _remove_placeholder_nulls(value: Any, schema: dict) -> Any:
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        result = {}
        for name, item in value.items():
            child = properties.get(name)
            if child is not None:
                if (
                    item is None
                    and name not in required
                    and not Draft202012Validator(child).is_valid(None)
                ):
                    continue
                item = _remove_placeholder_nulls(item, child)
            result[name] = item
        return result
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        return [_remove_placeholder_nulls(item, schema["items"]) for item in value]
    return value


def _normalize_calls(completion: Completion, specs: list[dict], strict: bool) -> Completion:
    if not strict:
        return completion
    schemas = {tool["name"]: tool["parameters"] for tool in specs}
    for call in completion.tool_calls:
        if call.name not in schemas:
            continue
        try:
            arguments = _json_loads(call.arguments)
        except (ValueError, TypeError):
            continue  # Registry returns a paired INVALID_ARGUMENT result.
        normalized = _remove_placeholder_nulls(arguments, schemas[call.name])
        if normalized != arguments:
            call.arguments = json.dumps(normalized, ensure_ascii=False)
    # provider_payload deliberately stays byte-semantically faithful to the wire output.
    return completion


def _attempt_usage(usage: dict | None, failures: list[dict]) -> dict:
    result = copy.deepcopy(usage or normalize_usage(None, "openai_chat"))
    if not failures:
        return result
    result["failed_attempts"] = copy.deepcopy(failures)
    unknown = set()
    fields = (
        "input_tokens_total",
        "input_tokens_uncached",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
    )
    for field in fields:
        values = [result.get(field), *(failure["usage"].get(field) for failure in failures)]
        known = [value for value in values if value is not None]
        result[field] = sum(known) if known else None
        if len(known) != len(values):
            unknown.add(field)
    result["incomplete_fields"] = sorted(unknown)
    result["source"] = "provider_partial" if unknown else "provider"
    return result


def normalize_usage(raw: dict | None, protocol: str) -> dict:
    raw = raw or {}
    result = dict.fromkeys(
        (
            "input_tokens_total",
            "input_tokens_uncached",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
        )
    )
    if protocol == "anthropic_messages":
        uncached = raw.get("input_tokens")
        read = raw.get("cache_read_input_tokens")
        write = raw.get("cache_creation_input_tokens")
        result.update(
            input_tokens_uncached=uncached,
            cache_read_tokens=read,
            cache_write_tokens=write,
            output_tokens=raw.get("output_tokens"),
        )
        if uncached is not None:
            result["input_tokens_total"] = uncached + (read or 0) + (write or 0)
    else:
        chat = protocol == "openai_chat"
        total = raw.get("prompt_tokens" if chat else "input_tokens")
        details = raw.get("prompt_tokens_details" if chat else "input_tokens_details") or {}
        cached = details.get("cached_tokens")
        written = details.get("cache_write_tokens")
        output_details = (
            raw.get("completion_tokens_details" if chat else "output_tokens_details") or {}
        )
        result.update(
            input_tokens_total=total,
            cache_read_tokens=cached,
            cache_write_tokens=written,
            input_tokens_uncached=total - cached - written
            if total is not None and cached is not None and written is not None
            else None,
            output_tokens=raw.get("completion_tokens" if chat else "output_tokens"),
            reasoning_tokens=output_details.get("reasoning_tokens"),
        )
    result.update(raw_usage=copy.deepcopy(raw), source="provider" if raw else "unknown")
    return result


def _validate_pairs(messages: list[dict]) -> None:
    pending: set[str] = set()
    seen: set[str] = set()
    for message in messages:
        if message["role"] == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                raise ProviderError("Orphan or duplicate tool result")
            pending.remove(call_id)
        else:
            if pending:
                raise ProviderError("A tool-call batch must close before another message")
            for call in message.get("tool_calls", []):
                call_id = call.get("id")
                if not call_id or call_id in seen:
                    raise ProviderError("Missing or duplicate tool call ID")
                seen.add(call_id)
                pending.add(call_id)
    if pending:
        raise ProviderError("Cannot send incomplete tool-call batch")


def _checked(completion: Completion) -> Completion:
    if completion.finish_reason not in {
        "stop",
        "tool_calls",
        "end_turn",
        "tool_use",
        "stop_sequence",
        "completed",
    }:
        raise ProviderError(
            f"Incomplete or rejected model response: {completion.finish_reason}",
            usage=completion.usage,
        )
    ids: set[str] = set()
    for call in completion.tool_calls:
        if not call.id or not call.name or call.id in ids:
            raise ProviderError("Malformed or duplicate tool call", usage=completion.usage)
        ids.add(call.id)
    if not completion.text and not completion.tool_calls:
        raise ProviderError("Model returned no answer or tool calls", usage=completion.usage)
    return completion


class HTTPProvider:
    def __init__(self, config: ProviderConfig, client: httpx.AsyncClient | None = None):
        if config.protocol not in {"openai_chat", "openai_responses", "anthropic_messages"}:
            raise ValueError(f"Unsupported protocol: {config.protocol}")
        self.config = config
        self.client = client or httpx.AsyncClient(timeout=config.timeout)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    def _request(
        self, messages: list[dict], tools: list[dict], system: str, stream: bool, maximum: int
    ) -> tuple[str, dict, dict]:
        _validate_pairs(messages)
        if self.config.strict_schema:
            tools = [
                {**copy.deepcopy(tool), "parameters": _strict_schema(tool["parameters"])}
                for tool in tools
            ]
        key = os.environ.get(self.config.api_key_env)
        if not key:
            raise ProviderError(f"Missing API key environment variable: {self.config.api_key_env}")
        protocol = self.config.protocol
        headers = {"Authorization": f"Bearer {key}"}
        body: dict[str, Any] = {"model": self.config.model, "stream": stream}
        if protocol == "openai_chat":
            path = "/chat/completions"
            wire = [{"role": "system", "content": system}] if system else []
            for message in messages:
                item = {"role": message["role"], "content": message.get("content", "")}
                if message["role"] == "assistant":
                    payload = message.get("provider_payload")
                    if isinstance(payload, dict):
                        item = copy.deepcopy(payload)
                    elif message.get("tool_calls"):
                        item["tool_calls"] = [
                            {
                                "id": c["id"],
                                "type": "function",
                                "function": {"name": c["name"], "arguments": c["arguments"]},
                            }
                            for c in message["tool_calls"]
                        ]
                if message["role"] == "tool":
                    item["tool_call_id"] = message["tool_call_id"]
                wire.append(item)
            body.update(messages=wire, max_completion_tokens=maximum)
            if tools:
                body.update(
                    tools=[{"type": "function", "function": copy.deepcopy(t)} for t in tools],
                    tool_choice="auto",
                )
                if self.config.strict_schema:
                    for spec in body["tools"]:
                        spec["function"]["strict"] = True
            if stream:
                body["stream_options"] = {"include_usage": True}
        elif protocol == "openai_responses":
            path = "/responses"
            wire = []
            for message in messages:
                if message["role"] == "tool":
                    wire.append(
                        {
                            "type": "function_call_output",
                            "call_id": message["tool_call_id"],
                            "output": message.get("content", ""),
                        }
                    )
                elif message["role"] == "assistant" and isinstance(
                    message.get("provider_payload"), list
                ):
                    wire.extend(copy.deepcopy(message["provider_payload"]))
                else:
                    if message.get("content"):
                        wire.append({"role": message["role"], "content": message["content"]})
                    for call in message.get("tool_calls", []):
                        wire.append(
                            {
                                "type": "function_call",
                                "call_id": call["id"],
                                "name": call["name"],
                                "arguments": call["arguments"],
                            }
                        )
            body.update(
                input=wire,
                instructions=system,
                max_output_tokens=maximum,
                store=False,
                truncation="disabled",
                include=["reasoning.encrypted_content"],
            )
            if tools:
                body["tools"] = [
                    {"type": "function", **copy.deepcopy(t), "strict": self.config.strict_schema}
                    for t in tools
                ]
                body["tool_choice"] = "auto"
        else:
            path = "/messages"
            headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
            wire = []
            for message in messages:
                role = message["role"]
                if role == "tool":
                    blocks = [
                        {
                            "type": "tool_result",
                            "tool_use_id": message["tool_call_id"],
                            "content": message.get("content", ""),
                        }
                    ]
                    role = "user"
                elif role == "assistant" and isinstance(message.get("provider_payload"), list):
                    blocks = copy.deepcopy(message["provider_payload"])
                else:
                    blocks = (
                        [{"type": "text", "text": message["content"]}]
                        if message.get("content")
                        else []
                    )
                    for call in message.get("tool_calls", []):
                        try:
                            args = _json_loads(call["arguments"])
                        except (ValueError, TypeError) as exc:
                            raise ProviderError(
                                "Cannot replay invalid Anthropic tool input"
                            ) from exc
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": call["id"],
                                "name": call["name"],
                                "input": args,
                            }
                        )
                if wire and wire[-1]["role"] == role:
                    wire[-1]["content"].extend(blocks)
                else:
                    wire.append({"role": role, "content": blocks})
            body.update(system=system, messages=wire, max_tokens=maximum)
            if tools:
                body["tools"] = [
                    {
                        "name": t["name"],
                        "description": t["description"],
                        "input_schema": copy.deepcopy(t["parameters"]),
                    }
                    for t in tools
                ]
                body["tool_choice"] = {"type": "auto"}
        # Extras may configure temperature/reasoning, but cannot override frozen history or auth.
        protected = {
            "model",
            "stream",
            "messages",
            "input",
            "system",
            "instructions",
            "tools",
            "tool_choice",
            "store",
            "truncation",
        }
        if protected.intersection(self.config.extra):
            raise ProviderError("Provider extras cannot override protocol/history fields")
        body.update(copy.deepcopy(self.config.extra))
        return self.config.base_url.rstrip("/") + path, headers, body

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str,
        *,
        stream: bool = False,
        on_delta: DeltaCallback | None = None,
        max_output_tokens: int | None = None,
    ) -> Completion:
        url, headers, body = self._request(
            messages, tools, system, stream, max_output_tokens or self.config.max_output_tokens
        )
        failures = []
        for attempt in range(self.config.max_retries + 1):
            try:
                async with self.client.stream(
                    "POST", url, headers=headers, json=body, timeout=self.config.timeout
                ) as response:
                    if response.status_code >= 400:
                        error_usage = None
                        try:
                            error_body = _json_loads(await response.aread())
                            if isinstance(error_body, dict) and isinstance(
                                error_body.get("usage"), dict
                            ):
                                error_usage = error_body["usage"]
                        except (ValueError, TypeError):
                            pass
                        failure = {
                            "attempt_index": attempt,
                            "http_status": response.status_code,
                            "usage": normalize_usage(error_usage, self.config.protocol),
                        }
                        if on_delta:
                            await on_delta({"type": "provider_attempt_failed", **failure})
                        if (
                            response.status_code in {408, 429, 500, 502, 503, 504}
                            and attempt < self.config.max_retries
                        ):
                            failures.append(failure)
                            try:
                                delay = min(
                                    float(response.headers.get("retry-after", 2**attempt)), 10
                                )
                            except ValueError:
                                delay = 2**attempt
                            await asyncio.sleep(max(0, delay))
                            continue
                        # Never echo body/URL/headers: compatible services can reflect secrets.
                        raise ProviderError(
                            f"Provider HTTP {response.status_code}", usage=failure["usage"]
                        )
                    if stream:
                        completion = await self._stream(response, on_delta)
                    else:
                        raw = _json_loads(await response.aread())
                        completion = self._parse(raw)
                    completion.usage = _attempt_usage(completion.usage, failures)
                    return _normalize_calls(completion, tools, self.config.strict_schema)
            except asyncio.CancelledError as exc:
                if failures:
                    exc.usage = _attempt_usage(None, failures)
                raise
            except ProviderError as exc:
                exc.usage = _attempt_usage(exc.usage, failures)
                raise
            except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError) as exc:
                # No retry once a response may have started; never concatenate attempts.
                raise ProviderError(
                    f"Provider transport or protocol failure ({type(exc).__name__})",
                    usage=_attempt_usage(None, failures),
                ) from exc
        raise ProviderError("Provider retries exhausted")

    def _parse(self, raw: dict) -> Completion:
        protocol = self.config.protocol
        usage = normalize_usage(raw.get("usage"), protocol)
        if raw.get("error"):
            raise ProviderError("Provider returned error", usage=usage)
        if protocol == "openai_chat":
            choice = raw["choices"][0]
            message = choice["message"]
            if message.get("refusal"):
                raise ProviderError("Provider refused response", usage=usage)
            calls = [
                ToolCall(c["id"], c["function"]["name"], c["function"]["arguments"])
                for c in message.get("tool_calls", [])
            ]
            return _checked(
                Completion(
                    message.get("content") or "",
                    calls,
                    usage,
                    choice.get("finish_reason"),
                    copy.deepcopy(message),
                    message.get("reasoning_content"),
                )
            )
        if protocol == "anthropic_messages":
            blocks = raw["content"]
            calls = [
                ToolCall(b["id"], b["name"], json.dumps(b["input"], ensure_ascii=False))
                for b in blocks
                if b["type"] == "tool_use"
            ]
            text = "".join(b["text"] for b in blocks if b["type"] == "text")
            reasoning = (
                "".join(b.get("thinking", "") for b in blocks if b["type"] == "thinking") or None
            )
            return _checked(
                Completion(
                    text, calls, usage, raw.get("stop_reason"), copy.deepcopy(blocks), reasoning
                )
            )
        items = raw.get("output", [])
        calls = [
            ToolCall(i["call_id"], i["name"], i["arguments"])
            for i in items
            if i["type"] == "function_call"
        ]
        text = ""
        summaries = []
        for item in items:
            if item["type"] == "message":
                for block in item.get("content", []):
                    if block["type"] == "refusal":
                        raise ProviderError("Provider refused response", usage=usage)
                    if block["type"] == "output_text":
                        text += block["text"]
            if item["type"] == "reasoning":
                summaries.extend(x["text"] for x in item.get("summary", []) if "text" in x)
        return _checked(
            Completion(
                text,
                calls,
                usage,
                raw.get("status"),
                copy.deepcopy(items),
                "\n".join(summaries) or None,
            )
        )

    async def _stream(self, response: httpx.Response, callback: DeltaCallback | None) -> Completion:
        protocol = self.config.protocol
        state: dict = {
            "usage": {},
            "choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": None}],
        }
        calls: dict[int, dict] = {}
        blocks: dict[int, dict] = {}
        open_blocks: set[int] = set()
        terminal = False
        data_lines: list[str] = []

        async def handle(data: str) -> None:
            nonlocal terminal, state
            if data == "[DONE]":
                if protocol != "openai_chat":
                    raise ProviderError("Unexpected stream sentinel")
                terminal = True
                return
            event = _json_loads(data)
            kind = event.get("type", "")
            if event.get("error") or kind in {"error", "response.failed", "response.incomplete"}:
                raise ProviderError(
                    "Provider stream failed", usage=normalize_usage(state.get("usage"), protocol)
                )
            if terminal:
                raise ProviderError("Provider sent data after terminal event")
            if protocol == "openai_chat":
                if event.get("usage"):
                    state["usage"] = event["usage"]
                for choice in event.get("choices", []):
                    if choice.get("index", 0) != 0:
                        continue
                    delta = choice.get("delta", {})
                    message = state["choices"][0]["message"]
                    if delta.get("content"):
                        message["content"] += delta["content"]
                    for field in ("reasoning_content", "refusal"):
                        if delta.get(field):
                            message[field] = message.get(field, "") + delta[field]
                    for call in delta.get("tool_calls", []):
                        target = calls.setdefault(
                            call["index"],
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if call.get("id"):
                            target["id"] += call["id"]
                        for field in ("name", "arguments"):
                            target["function"][field] += call.get("function", {}).get(field, "")
                    if choice.get("finish_reason"):
                        state["choices"][0]["finish_reason"] = choice["finish_reason"]
            elif protocol == "anthropic_messages":
                if kind == "message_start":
                    state = copy.deepcopy(event["message"])
                    state.setdefault("usage", {})
                elif kind == "content_block_start":
                    index = event["index"]
                    if index in blocks:
                        raise ProviderError("Duplicate content block")
                    blocks[index] = copy.deepcopy(event["content_block"])
                    open_blocks.add(index)
                elif kind == "content_block_delta":
                    index = event["index"]
                    if index not in open_blocks:
                        raise ProviderError("Delta outside open content block")
                    block, delta = blocks[index], event["delta"]
                    mapping = {
                        "text_delta": ("text", "text"),
                        "thinking_delta": ("thinking", "thinking"),
                        "signature_delta": ("signature", "signature"),
                        "input_json_delta": ("_partial_json", "partial_json"),
                    }
                    if delta["type"] in mapping:
                        target, source = mapping[delta["type"]]
                        block[target] = block.get(target, "") + delta[source]
                elif kind == "content_block_stop":
                    index = event["index"]
                    if index not in open_blocks:
                        raise ProviderError("Unmatched content block stop")
                    open_blocks.remove(index)
                    if "_partial_json" in blocks[index]:
                        blocks[index]["input"] = _json_loads(blocks[index].pop("_partial_json"))
                elif kind == "message_delta":
                    state.update(event.get("delta", {}))
                    state["usage"].update(event.get("usage", {}))
                elif kind == "message_stop":
                    if open_blocks:
                        raise ProviderError("Unclosed content blocks")
                    terminal = True
                    state["content"] = [blocks[i] for i in sorted(blocks)]
            elif kind == "response.completed":
                state = event["response"]
                terminal = True
            # UI events are transient. Only the fully validated completion becomes history.
            if callback:
                await callback({"type": "provider_delta", "protocol": protocol, "event": event})

        try:
            async for line in response.aiter_lines():
                if not line:
                    if data_lines:
                        await handle("\n".join(data_lines))
                        data_lines.clear()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip(" "))
            if data_lines:
                await handle("\n".join(data_lines))
            if not terminal:
                raise ProviderError(
                    "Stream ended without terminal event",
                    usage=normalize_usage(state.get("usage"), protocol),
                )
            if protocol == "openai_chat" and calls:
                state["choices"][0]["message"]["tool_calls"] = [calls[i] for i in sorted(calls)]
            return self._parse(state)
        except ProviderError:
            raise
        except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError) as exc:
            raise ProviderError(
                f"Invalid or interrupted stream ({type(exc).__name__})",
                usage=normalize_usage(state.get("usage"), protocol),
            ) from exc
