"""The provider contract and JSON handling shared by every provider."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class LLMRequest:
    role: str  # digest | dream | critique | dive | ping
    system: str
    prompt: str
    schema: dict[str, Any] | None = None
    max_tokens: int = 4000
    effort: str | None = None  # low | medium | high
    # May the crew look things up (web, papers) for this errand? Providers without tools ignore it.
    research: bool = False
    # The folder the errand works in (e.g. with downloaded papers); otherwise an empty temporary one.
    workdir: Path | None = None
    # Structured copy of the input. Real providers ignore it; the offline
    # provider uses it to answer deterministically in tests and demos.
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResult:
    ok: bool
    text: str = ""
    data: dict[str, Any] | None = None
    provider: str = ""
    model: str = ""
    ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    error: str = ""

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}" if self.model else self.provider


class Provider(Protocol):
    name: str

    def available(self) -> bool: ...

    def complete(self, request: LLMRequest, model: str | None) -> LLMResult: ...


def full_prompt(request: LLMRequest) -> str:
    """For providers without a separate system channel (CLI runners)."""
    parts = [request.system.strip(), "---", request.prompt.strip()]
    if request.schema:
        parts.append(
            "Reply with a single JSON object that matches this JSON Schema. No prose, no code fences.\n"
            + json.dumps(request.schema, ensure_ascii=False)
        )
    return "\n\n".join(parts)


def schema_hint(schema: dict[str, Any]) -> str:
    return (
        "Reply with a single JSON object that matches this JSON Schema. No prose, no code fences.\n"
        + json.dumps(schema, ensure_ascii=False)
    )


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> dict[str, Any] | None:
    """Find the JSON object in a model reply (bare, fenced, or embedded in prose)."""
    if not text:
        return None
    candidates = [text.strip()]
    candidates += [m.strip() for m in _FENCE.findall(text)]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return None


def check(data: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """A deliberately small JSON-Schema subset: type, required, properties,
    items, enum, minItems. Enough to catch the ways models actually go wrong."""
    errors: list[str] = []
    expected = schema.get("type")
    types = {
        "object": dict, "array": list, "string": str, "boolean": bool,
        "integer": int, "number": (int, float),
    }
    if expected in types:
        python_type = types[expected]
        if not isinstance(data, python_type) or (expected in {"integer", "number"} and isinstance(data, bool)):
            return [f"{path}: expected {expected}"]
    if "enum" in schema and data not in schema["enum"]:
        errors.append(f"{path}: {data!r} not in {schema['enum']}")
    if expected == "object":
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"{path}.{key}: missing")
        for key, sub in schema.get("properties", {}).items():
            if key in data:
                errors.extend(check(data[key], sub, f"{path}.{key}"))
    if expected == "array":
        if len(data) < schema.get("minItems", 0):
            errors.append(f"{path}: needs at least {schema['minItems']} items")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(data[:50]):
                errors.extend(check(item, item_schema, f"{path}[{index}]"))
    return errors[:12]


def obj(properties: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Strict object schema: every property required, nothing extra. Both the
    Codex CLI and the Anthropic structured-output API require this shape."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
        **extra,
    }


def arr(items: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"type": "array", "items": items, **extra}


STR = {"type": "string"}
INT = {"type": "integer"}
NUM = {"type": "number"}
BOOL = {"type": "boolean"}
STRS = {"type": "array", "items": {"type": "string"}}
