"""Compact audit projections; never change provider input or committed messages."""

from copy import deepcopy

from .context import canonical, fingerprint


def stream_trace(envelope: dict) -> dict:
    result = deepcopy(envelope)
    event = result.get("event")
    if not isinstance(event, dict):
        return result
    if result.get("protocol") != "openai_responses":
        return result
    original = deepcopy(event)
    response = event.get("response")
    if isinstance(response, dict):
        # Lifecycle snapshots repeat the entire request prefix and final output.
        event["response"] = {
            key: response[key]
            for key in ("id", "status", "model", "error", "incomplete_details", "usage")
            if key in response
        }
    if event.get("type", "").endswith(".done"):
        # Deltas contain text/arguments; final committed messages contain the
        # authoritative complete output, including opaque continuation blocks.
        for key in ("text", "arguments", "transcript", "item", "part"):
            value = event.pop(key, None)
            if isinstance(value, dict):
                event[key] = {
                    k: value[k] for k in ("id", "type", "status", "call_id", "name") if k in value
                }
    if event != original:
        result["trace_projection"] = {
            "version": 1,
            "original_hash": fingerprint(original),
            "original_bytes": len(canonical(original).encode("utf-8")),
        }
    return result
