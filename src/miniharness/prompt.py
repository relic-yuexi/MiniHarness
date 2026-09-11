"""Request-prefix construction through replaceable named middleware."""

from copy import deepcopy

from jsonschema import Draft202012Validator

from .hooks import Hooks

POSITIONS = {"pre_system_prompt", "post_system_prompt"}


async def build_prefix(hooks: Hooks, system: str, registry) -> dict:
    async def tool_schema(context):
        context["tools"] = registry.specs()

    hooks.register("tool_schema", tool_schema, position="post_system_prompt", builtin=True)
    hooks.freeze_positions(POSITIONS)
    context = await hooks.run("pre_system_prompt", {"system": system, "tools": []}, mutable=True)
    context = await hooks.run("post_system_prompt", context, mutable=True)
    if not isinstance(context.get("system"), str) or not isinstance(context.get("tools"), list):
        raise ValueError("Prompt middleware must produce system:str and tools:list")
    registered = {spec["name"] for spec in registry.specs()}
    names = set()
    for spec in context["tools"]:
        if not isinstance(spec, dict) or set(spec) != {"name", "description", "parameters"}:
            raise ValueError("Each injected tool needs name, description and parameters")
        name = spec["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Injected tool names must be nonempty and unique")
        if name not in registered:
            raise ValueError(f"Injected tool has no registered handler: {name}")
        if not isinstance(spec["description"], str) or not spec["description"].strip():
            raise ValueError("Injected tools require a description")
        if not isinstance(spec["parameters"], dict):
            raise ValueError("Injected tool parameters must be a JSON Schema object")
        Draft202012Validator.check_schema(spec["parameters"])
        names.add(name)
    return deepcopy({"system": context["system"], "tools": context["tools"]})
