"""Schema-driven tools. Handlers return values; only the runtime commits state."""

from __future__ import annotations

import ast
import asyncio
import codecs
import copy
import hashlib
import json
import logging
import math
import operator
import os
import re
import shutil
import signal
import stat
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from filelock import FileLock
from jsonschema import Draft202012Validator

PREVIEW_LIMIT = 32768
FILE_LIMIT = 16 * 1024 * 1024
OUTPUT_LIMIT = 8 * 1024 * 1024
CLEANUP_TIMEOUT = 2.0


@dataclass
class ToolContext:
    workspace: Path
    artifacts: Path
    todo: dict = field(default_factory=lambda: {"revision": 0, "items": []})
    bash_executable: str | None = None
    protected_root: Path | None = None


class ToolError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def result(text="", *, data=None, refs=None, effect="none", error=None):
    return {
        "ok": error is None,
        "preview_content": preview(text.encode("utf-8")),
        "ref_contents": refs or [],
        "data": data or {},
        "error": error,
        "effect_status": effect,
    }


def failure(code, message, effect="none"):
    return result(
        message, error={"code": code, "message": message, "retryable": False}, effect=effect
    )


def preview(raw: bytes, label="") -> str:
    header = ("文件位置：" + label[:512] + "\n") if label else ""
    if len(header.encode()) + len(raw) <= PREVIEW_LIMIT:
        return header + raw.decode("utf-8")
    marker = "\n[这里被省略了，如果需要请使用工具读取]\n"
    budget = PREVIEW_LIMIT - len((header + marker).encode())
    half = budget // 2
    return (
        header
        + raw[:half].decode("utf-8", errors="ignore")
        + marker
        + raw[-(budget - half) :].decode("utf-8", errors="ignore")
    )


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def path_lock(path):
    key = digest(os.path.normcase(str(path)).encode())
    locks = Path(tempfile.gettempdir()) / "miniharness-path-locks"
    locks.mkdir(exist_ok=True)
    return FileLock(str(locks / (key + ".lock")), timeout=10)


def safe_path(value: str, ctx: ToolContext, *, write=False):
    root = ctx.workspace.resolve()
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    # Resolve each component only after rejecting links/reparse points.
    for part in [candidate, *candidate.parents]:
        if part.exists() or part.is_symlink():
            st = part.lstat()
            if stat.S_ISLNK(st.st_mode) or getattr(st, "st_file_attributes", 0) & 0x400:
                raise ToolError("OUT_OF_SCOPE", "Links and reparse points are not supported")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ToolError("OUT_OF_SCOPE", "Path is outside workspace")
    if os.name == "nt" and (":" in str(candidate)[2:] or str(candidate).startswith("\\\\")):
        raise ToolError(
            "OUT_OF_SCOPE", "Device, network and alternate stream paths are not supported"
        )
    if os.name == "nt":
        reserved = {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10)),
        }
        if any(
            part.split(".")[0].upper() in reserved or part.endswith((" ", "."))
            for part in candidate.parts[1:]
        ):
            raise ToolError(
                "OUT_OF_SCOPE", "Reserved device names and ambiguous paths are not supported"
            )
    if write:
        protected = [ctx.artifacts.resolve()]
        if ctx.protected_root:
            protected.append(ctx.protected_root.resolve())
        if any(resolved.is_relative_to(p) for p in protected):
            raise ToolError("OUT_OF_SCOPE", "Runtime storage cannot be modified by file tools")
    return resolved


def atomic_write(path, raw, *, create_only=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".miniharness-", dir=path.parent)
    temp = Path(name)
    published = False
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(raw)
            out.flush()
            os.fsync(out.fileno())
        if path.exists():
            os.chmod(temp, stat.S_IMODE(path.stat().st_mode))
        if create_only:
            os.link(temp, path)  # atomic no-clobber publication
        else:
            os.replace(temp, path)
        published = True
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            if not published:
                raise
            logging.getLogger(__name__).warning(
                "Published file; temporary-file cleanup failed: %s", temp
            )


def snapshot(path):
    before = path.stat()
    if before.st_size > FILE_LIMIT:
        raise ToolError("OUTPUT_LIMIT", "File exceeds 16 MiB limit")
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ToolError("CONFLICT", "File changed while reading")
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ToolError("UNSUPPORTED_FILE", "Only valid UTF-8 text is supported") from None
    if b"\x00" in raw:
        raise ToolError("UNSUPPORTED_FILE", "Binary files are not supported")
    return raw


def artifact(raw, ctx, source="", start=0, end=None):
    key = digest(raw)
    target = ctx.artifacts / key
    ctx.artifacts.mkdir(parents=True, exist_ok=True)
    try:
        atomic_write(target, raw, create_only=True)
    except FileExistsError:
        if digest(target.read_bytes()) != key:
            raise ToolError("CONFLICT", "Artifact integrity mismatch") from None
    return {
        "artifact_id": key,
        "sha256": key,
        "source_path": source,
        "byte_start": start,
        "byte_end": len(raw) if end is None else end,
        "size_bytes": len(raw),
        "media_type": "text/plain; charset=utf-8",
    }


def read(args, ctx):
    if ("path" in args) == ("artifact_id" in args):
        raise ToolError("INVALID_ARGUMENT", "Provide exactly one of path or artifact_id")
    if ("offset" in args) != ("limit" in args):
        raise ToolError("INVALID_ARGUMENT", "offset and limit must be supplied together")
    path = safe_path(args["path"], ctx) if "path" in args else ctx.artifacts / args["artifact_id"]
    # Stream the complete source into an immutable snapshot; only ranges or
    # bounded head/tail buffers are kept in memory, including in full mode.
    ctx.artifacts.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".snapshot-", dir=ctx.artifacts)
    temp = Path(temp_name)
    hasher = hashlib.sha256()
    validator = codecs.getincrementaldecoder("utf-8")("strict")
    head, tail = b"", b""
    try:
        with os.fdopen(fd, "wb") as out, path_lock(path):
            before = path.stat()
            if before.st_size > FILE_LIMIT:
                raise ToolError("OUTPUT_LIMIT", "File exceeds 16 MiB limit")
            with path.open("rb") as source:
                while chunk := source.read(65536):
                    if out.tell() + len(chunk) > FILE_LIMIT:
                        raise ToolError("OUTPUT_LIMIT", "File exceeds 16 MiB limit")
                    if b"\x00" in chunk:
                        raise ToolError("UNSUPPORTED_FILE", "Binary files are not supported")
                    validator.decode(chunk)
                    hasher.update(chunk)
                    out.write(chunk)
                    if len(head) < PREVIEW_LIMIT:
                        head = (head + chunk)[:PREVIEW_LIMIT]
                    tail = (tail + chunk)[-PREVIEW_LIMIT:]
                validator.decode(b"", final=True)
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ToolError("CONFLICT", "File changed while reading")
            out.flush()
            os.fsync(out.fileno())
        source_hash = hasher.hexdigest()
        target = ctx.artifacts / source_hash
        try:
            os.link(temp, target)
        except FileExistsError:
            if digest(target.read_bytes()) != source_hash:
                raise ToolError("CONFLICT", "Artifact integrity mismatch") from None
    except UnicodeDecodeError:
        raise ToolError("UNSUPPORTED_FILE", "Only valid UTF-8 text is supported") from None
    finally:
        temp.unlink(missing_ok=True)
    if "artifact_id" in args and source_hash != args["artifact_id"]:
        raise ToolError("CONFLICT", "Artifact integrity mismatch")
    if args.get("expected_sha256", source_hash) != source_hash:
        raise ToolError("CONFLICT", "expected_sha256 mismatch")
    size = after.st_size
    start, end = 0, size
    if "offset" in args:
        start = args["offset"]
        if start > size:
            raise ToolError("INVALID_ARGUMENT", "offset exceeds file size")
        with target.open("rb") as source:
            source.seek(max(0, start - 3))
            window_start = max(0, start - 3)
            window = source.read(args["limit"] + 7)
        local = start - window_start
        while start < size and local > 0 and window[local] & 0xC0 == 0x80:
            local -= 1
            start -= 1
        selected = (
            window[local : local + args["limit"]].decode("utf-8", errors="ignore").encode("utf-8")
        )
        end = start + len(selected)
        ref = artifact(selected, ctx, str(path), start, end)
        content = preview(selected, str(path))
    else:
        ref = {
            "artifact_id": source_hash,
            "sha256": source_hash,
            "source_path": str(path),
            "byte_start": 0,
            "byte_end": size,
            "size_bytes": size,
            "media_type": "text/plain; charset=utf-8",
        }
        selected = (
            head
            if size <= PREVIEW_LIMIT
            else head.decode("utf-8", errors="ignore").encode()
            + tail.decode("utf-8", errors="ignore").encode()
        )
        content = preview(selected, str(path))
    data = {
        "source_sha256": source_hash,
        "byte_start": start,
        "byte_end": end,
        "requested_offset": args.get("offset", 0),
        "next_offset": end,
        "eof": end == size,
        "mode": "range" if "offset" in args else "full",
    }
    return result(content, data=data, refs=[ref])


def write(args, ctx):
    path = safe_path(args["path"], ctx, write=True)
    raw = args["content"].encode("utf-8")
    if len(raw) > FILE_LIMIT:
        raise ToolError("OUTPUT_LIMIT", "Content exceeds 16 MiB byte limit")
    with path_lock(path):
        old = snapshot(path) if path.exists() else None
        if old is not None and args.get("create_only", True):
            raise ToolError("CONFLICT", "File already exists")
        if old is not None and args.get("expected_sha256") != digest(old):
            raise ToolError("CONFLICT", "Overwrite requires matching expected_sha256")
        if old is None and "expected_sha256" in args:
            raise ToolError("CONFLICT", "Expected file does not exist")
        atomic_write(path, raw, create_only=old is None)
    return result(
        "File written",
        effect="applied",
        data={
            "before_sha256": digest(old) if old is not None else None,
            "sha256": digest(raw),
            "bytes_written": len(raw),
        },
    )


def edit(args, ctx):
    path = safe_path(args["path"], ctx, write=True)
    with path_lock(path):
        raw = snapshot(path)
        if digest(raw) != args["expected_sha256"]:
            raise ToolError("CONFLICT", "expected_sha256 mismatch")
        text = raw.decode("utf-8")
        count = text.count(args["old_string"])
        if count != args.get("expected_matches", 1):
            raise ToolError("CONFLICT", f"Expected match count differs: found {count}")
        new = text.replace(args["old_string"], args["new_string"]).encode("utf-8")
        if len(new) > FILE_LIMIT:
            raise ToolError("OUTPUT_LIMIT", "Edited file exceeds 16 MiB byte limit")
        atomic_write(path, new)
    return result(
        "File edited",
        effect="applied",
        data={"matches": count, "before_sha256": digest(raw), "sha256": digest(new)},
    )


def calculator(args, ctx):
    tree = ast.parse(args["expression"], mode="eval")
    if len(list(ast.walk(tree))) > 128:
        raise ToolError("INVALID_ARGUMENT", "Expression is too complex")
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in ops:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ToolError("INVALID_ARGUMENT", "Exponent exceeds limit")
            value = ops[type(node.op)](left, right)
        else:
            raise ToolError(
                "INVALID_ARGUMENT", "Only arithmetic literals and operators are permitted"
            )
        if (
            type(value) not in (int, float)
            or (isinstance(value, int) and value.bit_length() > 4096)
            or (isinstance(value, float) and not math.isfinite(value))
        ):
            raise ToolError("INVALID_ARGUMENT", "Result exceeds numeric limits")
        return value

    value = str(visit(tree.body))
    return result(value, data={"value": value})


def search(args, ctx):
    fixtures = [
        ("Agent runtime", "An agent loops through model decisions and tools."),
        ("Session isolation", "Sessions keep separate histories and todo state."),
        ("Context compaction", "Summaries preserve requirements and unfinished work."),
    ]
    terms = args["query"].lower().split()
    rows = [
        {"title": title, "url": f"https://example.invalid/docs/{i}", "snippet": body}
        for i, (title, body) in enumerate(fixtures)
        if any(t in (title + body).lower() for t in terms)
    ]
    data = {"mock": True, "query": args["query"], "results": rows[: args.get("limit", 5)]}
    return result("MOCK search (not live web): " + json.dumps(data), data=data)


def todo(args, ctx):
    if args["operation"] == "list":
        if "items" in args or "expected_revision" in args:
            raise ToolError("INVALID_ARGUMENT", "list does not accept update arguments")
        return result(json.dumps(ctx.todo, ensure_ascii=False), data=copy.deepcopy(ctx.todo))
    if "items" not in args or "expected_revision" not in args:
        raise ToolError("INVALID_ARGUMENT", "replace requires items and expected_revision")
    if args["expected_revision"] != ctx.todo["revision"]:
        raise ToolError("CONFLICT", "Todo revision has changed")
    ids = [item["id"] for item in args["items"]]
    if len(ids) != len(set(ids)):
        raise ToolError("INVALID_ARGUMENT", "Todo IDs must be unique")
    updated = {"revision": ctx.todo["revision"] + 1, "items": copy.deepcopy(args["items"])}
    response = result(
        json.dumps(updated, ensure_ascii=False), data=copy.deepcopy(updated), effect="applied"
    )
    response["_todo"] = updated
    return response


async def bash(args, ctx):
    executable = ctx.bash_executable or shutil.which("bash")
    if not executable or not Path(executable).is_file():
        raise ToolError("NOT_FOUND", "Configure an installed Bash executable")
    cwd = safe_path(args.get("cwd", str(ctx.workspace.resolve())), ctx)
    if not cwd.is_dir():
        raise ToolError("NOT_FOUND", "Shell cwd is not a directory")
    descriptor = {
        "job_id": uuid.uuid4().hex,
        "command": args["command"],
        "cwd": str(cwd),
        "executable": executable,
        "timeout_seconds": args.get("timeout_seconds", 30),
    }
    if args.get("background", False):
        response = result(
            "Background job accepted; completion is pending",
            data={"job_id": descriptor["job_id"], "status": "accepted"},
        )
        response["_job"] = descriptor
        return response
    return await run_bash_job(descriptor, ctx)


async def run_bash_job(descriptor: dict, context: ToolContext) -> dict:
    kwargs = (
        {"start_new_session": True}
        if os.name != "nt"
        else {"creationflags": 0x08000000 | 0x00000200}
    )
    try:
        process = await asyncio.create_subprocess_exec(
            descriptor["executable"],
            "--noprofile",
            "--norc",
            "-c",
            descriptor["command"],
            cwd=descriptor["cwd"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **kwargs,
        )
    except OSError as exc:
        return failure("IO_ERROR", str(exc))

    buffers = [bytearray(), bytearray()]
    dropped_counts = [0, 0]

    async def drain(stream, index):
        while chunk := await stream.read(65536):
            remaining = OUTPUT_LIMIT - len(buffers[index])
            buffers[index].extend(chunk[:remaining])
            dropped_counts[index] += max(0, len(chunk) - remaining)

    streams = [
        asyncio.create_task(drain(process.stdout, 0)),
        asyncio.create_task(drain(process.stderr, 1)),
    ]
    timed_out = False
    cleanup_incomplete = False

    async def terminate():
        if os.name == "nt":
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(process.pid),
                "/T",
                "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                await killer.wait()
            finally:
                if killer.returncode is None:
                    try:
                        killer.kill()
                    except ProcessLookupError:
                        pass
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await process.wait()

    async def cleanup():
        nonlocal cleanup_incomplete
        try:
            async with asyncio.timeout(CLEANUP_TIMEOUT):
                await terminate()
                await asyncio.gather(*streams)
        except (TimeoutError, OSError):
            cleanup_incomplete = True
        finally:
            for task in streams:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*streams, return_exceptions=True)

    try:
        async with asyncio.timeout(descriptor["timeout_seconds"]):
            await process.wait()
            # wait() does not cancel drain tasks when the caller deadline expires.
            await asyncio.wait(streams)
            for task in streams:
                task.result()
    except TimeoutError:
        timed_out = True
        await _settle_task(asyncio.create_task(cleanup()))
    except asyncio.CancelledError:
        await _settle_task(asyncio.create_task(cleanup()))
        raise
    outputs = []
    refs = []
    dropped_total = 0
    for index in range(len(streams)):
        raw, dropped = bytes(buffers[index]), dropped_counts[index]
        dropped_total += dropped
        # Shell output may use arbitrary encodings; normalize artifact to UTF-8.
        normalized = raw.decode("utf-8", errors="replace").encode("utf-8")
        refs.append(artifact(normalized, context, "stdout" if index == 0 else "stderr"))
        outputs.append(preview(normalized))
    data = {
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "cleanup_incomplete": cleanup_incomplete,
        "output_partial": timed_out or cleanup_incomplete,
        "stdout": outputs[0],
        "stderr": outputs[1],
        "dropped_bytes": dropped_total,
        "truncated": dropped_total > 0,
    }
    error = None
    if timed_out:
        error = {
            "code": "TIMEOUT",
            "message": "Command timed out; side effects may have occurred",
            "retryable": False,
        }
    elif process.returncode:
        error = {
            "code": "PROCESS_ERROR",
            "message": f"Command exited with code {process.returncode}",
            "retryable": False,
        }
    return result(
        "stdout:\n" + outputs[0] + "\nstderr:\n" + outputs[1],
        data=data,
        refs=refs,
        effect="unknown" if timed_out or process.returncode else "applied",
        error=error,
    )


async def _settle_task(task):
    """Cancellation cannot release ownership while a synchronous worker still runs."""
    cancelled = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancelled = exc
        except BaseException:
            break
    if cancelled is not None:
        if not task.cancelled():
            task.exception()  # Retrieve worker failure; cancellation remains authoritative.
        raise cancelled
    return task.result()


def strict_json(value):
    def pairs(items):
        answer = {}
        for key, item in items:
            if key in answer:
                raise ValueError("Duplicate JSON key")
            answer[key] = item
        return answer

    def invalid(value):
        raise ValueError("Nonfinite JSON number")

    if isinstance(value, str):
        if len(value.encode()) > 2 * FILE_LIMIT:
            raise ValueError("Arguments exceed size limit")
        value = json.loads(value, object_pairs_hook=pairs, parse_constant=invalid)
    else:
        # Reject NaN/Infinity even in caller-supplied dicts.
        json.dumps(value, allow_nan=False)
    if not isinstance(value, dict):
        raise TypeError("Tool arguments must be a JSON object")
    return value


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, name: str, description: str, parameters: dict, handler: Callable):
        if name in self._tools:
            raise ValueError(f"Duplicate tool: {name}")
        if not callable(handler) or not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]{0,63}", name):
            raise ValueError("Invalid tool registration")
        Draft202012Validator.check_schema(parameters)
        self._tools[name] = (
            {"name": name, "description": description, "parameters": copy.deepcopy(parameters)},
            handler,
        )

    def specs(self) -> list[dict]:
        return [copy.deepcopy(self._tools[name][0]) for name in sorted(self._tools)]

    async def execute(self, name: str, arguments: str | dict, context: ToolContext) -> dict:
        if name not in self._tools:
            return failure("UNKNOWN_TOOL", f"Unknown tool: {name}")
        spec, handler = self._tools[name]
        try:
            args = strict_json(arguments)
            errors = list(Draft202012Validator(spec["parameters"]).iter_errors(args))
            if errors:
                raise ValueError(errors[0].message)
        except (ValueError, TypeError, RecursionError) as exc:
            return failure("INVALID_ARGUMENT", str(exc))
        try:
            if asyncio.iscoroutinefunction(handler):
                return await handler(args, context)
            return await _settle_task(
                asyncio.create_task(asyncio.to_thread(handler, args, context))
            )
        except ToolError as exc:
            return failure(exc.code, str(exc))
        except FileNotFoundError as exc:
            return failure("NOT_FOUND", str(exc))
        except FileExistsError as exc:
            return failure("CONFLICT", str(exc))
        except PermissionError as exc:
            return failure("PERMISSION_DENIED", str(exc))
        except (ValueError, ArithmeticError, SyntaxError) as exc:
            return failure("INVALID_ARGUMENT", str(exc))
        except OSError as exc:
            return failure("IO_ERROR", str(exc))
        except Exception as exc:  # noqa: BLE001 -- plugin errors must close the tool call
            return failure("INTERNAL_ERROR", f"{type(exc).__name__}: {exc}")


def builtin_registry() -> ToolRegistry:
    registry = ToolRegistry()
    string = {"type": "string", "minLength": 1, "maxLength": 4096}
    sha = {"type": "string", "pattern": "^[a-f0-9]{64}$"}

    def add(name, description, properties, required, handler):
        registry.register(
            name,
            description,
            {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            handler,
        )

    add(
        "read",
        "Read UTF-8 files or immutable artifacts; byte ranges require offset and limit.",
        {
            "path": string,
            "artifact_id": sha,
            "expected_sha256": sha,
            "offset": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 4, "maximum": 1048576},
        },
        [],
        read,
    )
    add(
        "write",
        "Create or overwrite a UTF-8 file. Overwrite requires create_only=false and expected_sha256.",
        {
            "path": string,
            "content": {"type": "string", "maxLength": FILE_LIMIT},
            "expected_sha256": sha,
            "create_only": {"type": "boolean"},
        },
        ["path", "content"],
        write,
    )
    add(
        "edit",
        "Precisely replace text after checking hash and exact match count.",
        {
            "path": string,
            "old_string": {"type": "string", "minLength": 1, "maxLength": FILE_LIMIT},
            "new_string": {"type": "string", "maxLength": FILE_LIMIT},
            "expected_sha256": sha,
            "expected_matches": {"type": "integer", "minimum": 1},
        },
        ["path", "old_string", "new_string", "expected_sha256"],
        edit,
    )
    add(
        "bash",
        "Execute a Bash command. Background jobs return an accepted handle, not completion.",
        {
            "command": {"type": "string", "minLength": 1, "maxLength": 65536},
            "cwd": string,
            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 120},
            "background": {"type": "boolean"},
        },
        ["command"],
        bash,
    )
    add(
        "calculator",
        "Evaluate bounded arithmetic expressions without code execution.",
        {"expression": {"type": "string", "minLength": 1, "maxLength": 2048}},
        ["expression"],
        calculator,
    )
    add(
        "search",
        "Search deterministic MOCK documentation, not the live web.",
        {
            "query": {"type": "string", "minLength": 1, "maxLength": 2048},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        ["query"],
        search,
    )
    item = {
        "type": "object",
        "properties": {
            "id": string,
            "text": string,
            "status": {"enum": ["pending", "in_progress", "completed"]},
        },
        "required": ["id", "text", "status"],
        "additionalProperties": False,
    }
    add(
        "todo",
        "List or replace the current session todo list using optimistic revision checks.",
        {
            "operation": {"enum": ["list", "replace"]},
            "expected_revision": {"type": "integer", "minimum": 0},
            "items": {"type": "array", "items": item, "maxItems": 100},
        },
        ["operation"],
        todo,
    )
    return registry
