import pytest

from miniharness.context import estimate_tokens, validate_pairing
from miniharness.hooks import Hooks


async def test_hooks_unwind_and_observer_failure():
    order = []
    errors = []
    async def pre_a(data): order.append("A+")
    async def post_a(data): order.append("A-")
    async def fail(data): raise ValueError("no")
    async def diagnostic(data): errors.append(data)
    hooks = Hooks(on_error=diagnostic)
    hooks.add("action", pre=pre_a, post=post_a)
    hooks.add("action", pre=fail)
    with pytest.raises(ValueError):
        async with hooks.scope("action", {}):
            pytest.fail("guard should stop execution")
    assert order == ["A+", "A-"]
    assert len(errors) == 1


def test_utf8_estimate_and_batch_boundary():
    assert estimate_tokens("你好😀") == 3
    call = {"role": "assistant", "tool_calls": [{"id": "x"}, {"id": "y"}]}
    results = [{"role": "tool", "tool_call_id": x} for x in ["x", "y"]]
    validate_pairing([call, *results, {"role": "user"}])
    with pytest.raises(ValueError): validate_pairing([call, results[0], {"role": "user"}])
    with pytest.raises(ValueError): validate_pairing([*results])
