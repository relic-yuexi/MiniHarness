"""Named middleware queues, plus a compatibility adapter for paired hooks."""

import asyncio
import inspect
import re
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, replace

Callback = Callable[[dict], Awaitable[dict | None]]


def position_name(value: str) -> str:
    """Accept preActionHook, pre_action_hook and pre_action consistently."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Hook position must be a nonempty string")
    value = re.sub(r"(?<!^)(?=[A-Z])", "_", value.strip()).lower()
    if value.endswith("_hook"):
        value = value[:-5]
    if not re.fullmatch(r"[a-z][a-z0-9_]*", value):
        raise ValueError("Invalid hook position")
    return value


@dataclass(frozen=True)
class Hook:
    name: str
    position: str
    callback: Callback
    priority: int
    sequence: int
    guard: bool = True
    builtin: bool = False
    pair: str | None = None
    reverse: bool = False


class Hooks:
    """Higher priority first; ties FIFO. Names are unique per registry.

    Replacing a name keeps its sequence, even across positions. An active
    dispatch uses a snapshot, so registration only affects later dispatches.
    """

    def __init__(self, timeout: float = 5.0, on_error: Callback | None = None):
        if timeout <= 0:
            raise ValueError("Hook timeout must be positive")
        self.timeout = timeout
        self.on_error = on_error
        self._hooks: dict[str, Hook] = {}
        self._sequence = 0
        self._pairs = 0
        self._frozen: set[str] = set()

    def register(
        self,
        name: str,
        callback: Callback,
        *,
        position: str,
        priority: int = 0,
        guard: bool = True,
        builtin: bool = False,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Hook name must be a nonempty string")
        if not callable(callback) or not (
            inspect.iscoroutinefunction(callback)
            or inspect.iscoroutinefunction(type(callback).__call__)
        ):
            raise TypeError("Hooks must be cooperative async callbacks")
        if not isinstance(priority, int) or isinstance(priority, bool):
            raise TypeError("Hook priority must be an integer")
        position = position_name(position)
        previous = self._hooks.get(name)
        if builtin and previous is not None and not previous.builtin:
            return
        if position in self._frozen or (previous and previous.position in self._frozen):
            raise RuntimeError("This hook position is frozen for the active session")
        if previous:
            sequence = previous.sequence
        else:
            sequence = self._sequence
            self._sequence += 1
        self._hooks[name] = Hook(name, position, callback, priority, sequence, guard, builtin)

    def entries(self, position: str) -> tuple[Hook, ...]:
        position = position_name(position)
        return tuple(
            sorted(
                (h for h in self._hooks.values() if h.position == position),
                key=lambda h: (-h.priority, -h.sequence if h.reverse else h.sequence),
            )
        )

    def freeze_positions(self, positions) -> None:
        self._frozen.update(position_name(p) for p in positions)

    def copy(self, *, on_error: Callback | None = None) -> "Hooks":
        result = Hooks(self.timeout, self.on_error if on_error is None else on_error)
        result._hooks = self._hooks.copy()
        result._sequence = self._sequence
        result._pairs = self._pairs
        result._frozen = self._frozen.copy()
        return result

    def add(
        self,
        scope: str,
        *,
        pre: Callback | None = None,
        post: Callback | None = None,
        guard: bool = True,
    ) -> None:
        """Legacy pairs unwind in reverse; register() queues use priority/FIFO."""
        scope = position_name(scope)
        pair = f"__paired_{self._pairs}"
        if f"pre_{scope}" in self._frozen or f"post_{scope}" in self._frozen:
            raise RuntimeError("This hook position is frozen for the active session")
        if pre is None and post is not None:
            # A post-only legacy layer still has an entry point. Otherwise a
            # preceding guard failure would unwind layers never reached.
            async def pre(data):
                return None

        candidate = self.copy()
        for phase, callback in (("pre", pre), ("post", post)):
            if callback:
                name = f"{pair}_{phase}"
                candidate.register(name, callback, position=f"{phase}_{scope}", guard=guard)
                candidate._hooks[name] = replace(
                    candidate._hooks[name], pair=pair, reverse=phase == "post"
                )
        self._hooks, self._sequence = candidate._hooks, candidate._sequence
        self._pairs += 1

    async def _call(self, hook: Hook, data: dict) -> dict:
        scratch = deepcopy(data)
        async with asyncio.timeout(self.timeout):
            returned = await hook.callback(scratch)
        if returned is not None:
            if not isinstance(returned, dict):
                raise TypeError("A hook must return a dict patch or None")
            scratch.update(returned)
        return deepcopy(scratch)

    async def _diagnostic(self, hook: Hook, error: Exception) -> None:
        if self.on_error:
            async with asyncio.timeout(self.timeout):
                await self.on_error(
                    {
                        "scope": hook.position,
                        "name": hook.name,
                        "error": type(error).__name__,
                        "message": str(error),
                    }
                )

    async def run(self, position: str, data: dict, *, mutable: bool = False) -> dict:
        """Return a new context; failed mutations never leak to the caller."""
        current = deepcopy(data)
        for hook in self.entries(position):
            try:
                updated = await self._call(hook, current)
            except Exception as exc:
                await self._diagnostic(hook, exc)
                if hook.guard:
                    raise
            else:
                if mutable:
                    current = updated
        return current

    @asynccontextmanager
    async def scope(self, name: str, data: dict):
        name = position_name(name)
        pres, posts = self.entries(f"pre_{name}"), self.entries(f"post_{name}")
        pairs_with_pre = {h.pair for h in pres if h.pair}
        entered = {h.pair for h in posts if h.pair and h.pair not in pairs_with_pre}
        outcome = "succeeded"
        try:
            for hook in pres:
                try:
                    await self._call(hook, data)
                except Exception as exc:
                    await self._diagnostic(hook, exc)
                    if hook.guard:
                        raise
                else:
                    if hook.pair:
                        entered.add(hook.pair)
            yield
        except BaseException:
            outcome = "failed"
            raise
        finally:
            for hook in posts:
                if hook.pair and hook.pair not in entered:
                    continue
                try:
                    await self._call(hook, data | {"outcome": outcome})
                except Exception as exc:
                    await self._diagnostic(hook, exc)
