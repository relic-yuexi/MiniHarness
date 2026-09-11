"""Interactive and batch CLI; inspection never opens the session writer."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

from . import __version__
from .config import Config, load_config
from .runtime import Runtime
from .state import replay
from .storage import CorruptLog, SessionStore, StorageError

HELP = """Commands:
/session: session ID; /new: independent session instructions; /todo: committed todo list
/compact: queue compaction; /steer TEXT: steer at safe boundary; /followup TEXT: new turn
/cancel: cancel active work; /help: commands; /quit: finish accepted work and exit
Ordinary messages are queued as followup while the agent is busy.
"""


def visible_delta(data: dict) -> str:
    """Extract public text only, never reasoning or tool arguments."""
    if data.get("maintenance"):
        return ""
    event = data.get("event", {})
    if not isinstance(event, dict):
        return ""
    protocol = data.get("protocol")
    if protocol == "openai_chat":
        return "".join(
            c.get("delta", {}).get("content") or ""
            for c in event.get("choices", [])
            if c.get("index", 0) == 0
        )
    if protocol == "anthropic_messages" and event.get("type") == "content_block_delta":
        delta = event.get("delta", {})
        return delta.get("text", "") if delta.get("type") == "text_delta" else ""
    if protocol == "openai_responses" and event.get("type") == "response.output_text.delta":
        return event.get("delta", "")
    return ""


class Console:
    def __init__(self, stream=True):
        self.stream = stream
        self.partial = ""
        self.printed_turns = set()

    def complete(self, result):
        turn_id = result.get("turn_id")
        if turn_id and turn_id in self.printed_turns:
            return
        if turn_id:
            self.printed_turns.add(turn_id)
        text = result.get("text", "")
        if self.partial:
            print(flush=True)
        if text and not (self.partial and self.partial.endswith(text)):
            print(text, flush=True)
        if result.get("status") != "completed":
            print(f"[{result.get('status', 'unknown')}]", file=sys.stderr, flush=True)
        self.partial = ""

    async def event(self, data):
        kind = data.get("type")
        if kind in {"provider_delta", "delta"} and self.stream:
            text = visible_delta(data)
            if text:
                self.partial += text
                print(text, end="", flush=True)
        elif kind == "turn.ended":
            self.complete(data)
        elif kind == "action.completed":
            result = data.get("result", {})
            print(
                f"[tool {data.get('name')}: {'ok' if result.get('ok') else 'failed'}; effect={result.get('effect_status', 'none')}]",
                file=sys.stderr,
                flush=True,
            )
        elif kind == "job.completed":
            print(f"\n[job {data.get('job_id')}: {data.get('status')}]", flush=True)
            print(data.get("result", {}).get("preview_content", ""), flush=True)


def inspect_session(config: Config, session_id: str) -> dict:
    # Constructor and validation are pure: never open(), repair, lock, or run hooks.
    probe = SessionStore(config.session_root, session_id, repair=False)
    path = probe.path / "session.jsonl"
    if probe.path.is_symlink() or path.is_symlink() or probe.path.resolve().parent != probe.root:
        raise StorageError("Refusing session links outside the session root")
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise CorruptLog(
            "Empty log or unterminated tail; read-only inspection does not repair logs"
        )
    try:
        probe.user_id = json.loads(raw.splitlines()[0])["payload"]["user_id"]
    except (ValueError, TypeError, KeyError) as exc:
        raise CorruptLog("Invalid session creation record") from exc
    events = probe._validate(raw)
    state = replay(events)
    return {
        "session_id": session_id,
        "user_id": probe.user_id,
        "events": len(events),
        "last_seq": events[-1]["seq"],
        "last_hash": events[-1]["hash"],
        "event_types": dict(Counter(event["type"] for event in events)),
        "usage": state.usage,
        "total_tokens": state.total_tokens,
        "todo": state.todo,
        "queued_inputs": len(state.queue),
        "current_turn": state.current_turn,
        "jobs": state.jobs,
        "context_epoch": state.epoch,
        "hash_chain": "valid",
    }


async def run_once(config, args):
    console = Console(config.stream)
    async with Runtime(
        config, session_id=args.session, user_id=args.user, on_event=console.event
    ) as runtime:
        print(f"Session: {runtime.session_id}", file=sys.stderr)
        response = await runtime.ask(args.prompt)
        console.complete(response)
        await runtime.wait_idle(include_jobs=True)
        return 0 if response.get("status") == "completed" else 1


async def chat(config, args):
    console = Console(config.stream)
    watchers = set()
    async with Runtime(
        config, session_id=args.session, user_id=args.user, on_event=console.event
    ) as runtime:
        print(f"Session: {runtime.session_id}\nType /help for commands.")

        async def watch(ticket):
            try:
                console.complete(await ticket.wait())
            except (RuntimeError, ValueError, OSError) as exc:
                print(f"Error: {exc}", file=sys.stderr)

        while True:
            try:
                line = (await asyncio.to_thread(input, "> ")).strip()
            except EOFError:
                break
            if not line:
                continue
            if line in {"/quit", "/exit"}:
                break
            try:
                if line == "/help":
                    print(HELP)
                    continue
                if line == "/session":
                    print(runtime.session_id)
                    continue
                if line == "/new":
                    print(
                        "Open another terminal and run miniharness chat without --session for an independent session."
                    )
                    continue
                if line == "/todo":
                    print(json.dumps(runtime.state.todo, ensure_ascii=False, indent=2))
                    continue
                if line == "/cancel":
                    await runtime.cancel()
                    print("Cancellation requested.")
                    continue
                if line == "/compact":
                    ticket = await runtime.submit("", mode="compact")
                elif line.startswith("/steer "):
                    ticket = await runtime.steer(line[len("/steer ") :])
                elif line.startswith("/followup "):
                    ticket = await runtime.followup(line[len("/followup ") :])
                elif line.startswith("/"):
                    print("Unknown or incomplete command; use /help.", file=sys.stderr)
                    continue
                else:
                    ticket = await runtime.followup(line)
                task = asyncio.create_task(watch(ticket))
                watchers.add(task)
                task.add_done_callback(watchers.discard)
            except (ValueError, RuntimeError, OSError) as exc:
                print(f"Error: {exc}", file=sys.stderr)
        await runtime.wait_idle(include_jobs=True)
        if watchers:
            await asyncio.gather(*watchers)
    return 0


def parser():
    root = argparse.ArgumentParser(description="MiniHarness: a framework-free Python agent")
    root.add_argument("--version", action="version", version=__version__)
    root.add_argument("--config", default="config.toml", help="TOML configuration file")
    commands = root.add_subparsers(dest="command")
    for name in ("chat", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--session", help="Resume an existing 64-character session ID")
        sub.add_argument("--user", default="default")
        sub.add_argument("--no-stream", action="store_true")
        if name == "run":
            sub.add_argument("prompt")
    commands.add_parser("sessions", help="List sessions without changing them")
    inspect = commands.add_parser("inspect", help="Validate a session without repairing it")
    inspect.add_argument("session_id")
    commands.add_parser("doctor", help="Check local config without sending API requests")
    return root


def execute(argv=None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    if not args.command:
        cli.print_help()
        return 0
    try:
        config = load_config(args.config)
        if getattr(args, "no_stream", False):
            config.stream = False
        if args.command == "doctor":
            executable = config.bash_executable or shutil.which("bash")
            print(
                json.dumps(
                    {
                        "python": sys.version.split()[0],
                        "python_executable": sys.executable,
                        "protocol": config.provider.protocol,
                        "model": config.provider.model,
                        "api_key_env": config.provider.api_key_env,
                        "api_key_present": bool(os.environ.get(config.provider.api_key_env)),
                        "bash_executable": executable,
                        "bash_exists": bool(executable and Path(executable).is_file()),
                        "workspace": str(config.workspace),
                        "session_root": str(config.session_root),
                        "api_connection": "not tested",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "sessions":
            rows = []
            for path in sorted(config.session_root.glob("*/session.jsonl")):
                try:
                    info = inspect_session(config, path.parent.name)
                    rows.append(
                        {
                            key: info[key]
                            for key in (
                                "session_id",
                                "user_id",
                                "events",
                                "queued_inputs",
                                "total_tokens",
                            )
                        }
                    )
                except (ValueError, OSError, StorageError) as exc:
                    rows.append({"session_id": path.parent.name, "error": str(exc)})
            print(json.dumps(rows, ensure_ascii=False, indent=2))
            return 0
        if args.command == "inspect":
            print(
                json.dumps(inspect_session(config, args.session_id), ensure_ascii=False, indent=2)
            )
            return 0
        return asyncio.run(run_once(config, args) if args.command == "run" else chat(config, args))
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Resume the session to inspect durable state.", file=sys.stderr)
        return 130


def main() -> None:
    raise SystemExit(execute())
