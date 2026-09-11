"""Bounded, ordered lifecycle middleware with explicit guard/observer semantics."""

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Awaitable, Callable

Callback = Callable[[dict], Awaitable[None]]


@dataclass
class Layer:
    pre: Callback | None = None
    post: Callback | None = None
    guard: bool = True


class Hooks:
    def __init__(self, timeout: float = 5.0, on_error: Callback | None = None):
        self.timeout = timeout
        self.on_error = on_error
        self.layers: dict[str, list[Layer]] = {}

    def add(
        self,
        scope: str,
        *,
        pre: Callback | None = None,
        post: Callback | None = None,
        guard: bool = True,
    ) -> None:
        """Scopes: session, user, turn, step, assistant, action, compact."""
        if scope not in {"session", "user", "turn", "step", "assistant", "action", "compact"}:
            raise ValueError(f"Unknown hook scope: {scope}")
        self.layers.setdefault(scope, []).append(Layer(pre, post, guard))

    async def _call(self, callback: Callback, data: dict) -> None:
        async with asyncio.timeout(self.timeout):
            await callback(deepcopy(data))

    async def _diagnostic(self, scope: str, error: Exception) -> None:
        if self.on_error:
            # Do not recursively catch a failed diagnostic write.
            await self.on_error(
                {"scope": scope, "error": type(error).__name__, "message": str(error)}
            )

    @asynccontextmanager
    async def scope(self, name: str, data: dict):
        entered: list[Layer] = []
        outcome = "succeeded"
        try:
            for layer in self.layers.get(name, []):
                if layer.pre:
                    try:
                        await self._call(layer.pre, data)
                    except Exception as exc:
                        await self._diagnostic(name, exc)
                        if layer.guard:
                            raise
                        continue
                entered.append(layer)
            yield
        except BaseException:
            outcome = "failed"
            raise
        finally:
            for layer in reversed(entered):
                if layer.post:
                    try:
                        await self._call(layer.post, data | {"outcome": outcome})
                    except Exception as exc:
                        # A post failure never undoes an already committed effect.
                        await self._diagnostic(name, exc)
