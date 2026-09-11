"""Stable context projections and intentionally simple token budgeting."""

import hashlib
import json
import math


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def estimate_tokens(value) -> int:
    text = value if isinstance(value, str) else canonical(value)
    return math.ceil(len(text.encode("utf-8")) / 4)


def fingerprint(value) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def validate_pairing(messages: list[dict]) -> None:
    pending: set[str] = set()
    seen: set[str] = set()
    for message in messages:
        role = message["role"]
        if role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                raise ValueError(f"Orphan/duplicate tool result: {call_id}")
            pending.remove(call_id)
            continue
        if pending:
            raise ValueError("A message interrupts an incomplete tool batch")
        for call in message.get("tool_calls", []):
            if not call.get("id") or call["id"] in seen:
                raise ValueError("Duplicate or missing tool call ID")
            seen.add(call["id"])
            pending.add(call["id"])
    if pending:
        raise ValueError("Unresolved tool calls")
