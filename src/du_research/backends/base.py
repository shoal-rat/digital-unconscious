"""Provider-neutral response types and prompt helpers."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any, Protocol


class AIBackend(Protocol):
    def call(
        self,
        prompt: str,
        *,
        mode: str = "balanced",
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 2048,
        json_schema: dict | None = None,
        session_id: str | None = None,
        allowed_tools: list[str] | None = None,
        use_chrome: bool = False,
        max_turns: int = 25,
        agent: str | None = None,
        think: int | str | None = None,
        images: list[bytes] | None = None,
        web_search: bool = False,
    ) -> AIResponse: ...


@dataclass
class AIResponse:
    text: str
    model: str = ""
    session_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)
    structured: dict | None = None

    @property
    def ok(self) -> bool:
        return bool(self.text)


MODE_TEMPERATURE = {
    "creative": 0.95,
    "balanced": 0.7,
    "strict": 0.1,
    "deterministic": 0.0,
}
EFFORT_BUDGETS = {"low": 2048, "medium": 8192, "high": 16384, "xhigh": 32768, "max": 65536}


def parse_structured_text(text: str, json_schema: dict | None) -> dict | None:
    if not json_schema:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}") + 1
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(text[start:end])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else {"result": parsed}


def usage_value(usage: Any, key: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(key, 0) or 0)
    return int(getattr(usage, key, 0) or 0)


def thinking_budget(think: int | str | bool | None) -> int:
    if think is None or think is False:
        return 0
    if think is True:
        return EFFORT_BUDGETS["medium"]
    if isinstance(think, str):
        return EFFORT_BUDGETS.get(think.strip().lower(), 0)
    try:
        return max(0, int(think))
    except (TypeError, ValueError):
        return 0


def reasoning_effort(think: int | str | bool | None) -> str | None:
    if isinstance(think, str):
        value = think.strip().lower()
        return value if value in EFFORT_BUDGETS else "medium"
    budget = thinking_budget(think)
    if budget <= 0:
        return None
    if budget <= EFFORT_BUDGETS["low"]:
        return "low"
    if budget < EFFORT_BUDGETS["high"]:
        return "medium"
    if budget < EFFORT_BUDGETS["xhigh"]:
        return "high"
    if budget < EFFORT_BUDGETS["max"]:
        return "xhigh"
    return "max"


def image_media_type(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def anthropic_user_content(prompt: str, images: list[bytes] | None):
    if not images:
        return prompt
    blocks = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image_media_type(image),
                "data": base64.standard_b64encode(image).decode("ascii"),
            },
        }
        for image in images
    ]
    blocks.append({"type": "text", "text": prompt})
    return blocks


def openai_user_content(prompt: str, images: list[bytes] | None):
    if not images:
        return prompt
    parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image in images:
        encoded = base64.standard_b64encode(image).decode("ascii")
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_media_type(image)};base64,{encoded}"},
            }
        )
    return parts


def compose_prompt(prompt: str, system: str | None, mode: str, max_tokens: int) -> str:
    mode_text = {
        "creative": "Explore several genuinely different possibilities before choosing.",
        "strict": "Be conservative: distinguish evidence, inference, and uncertainty.",
        "deterministic": "Return the single most defensible answer without ornamental prose.",
    }.get(mode, "")
    parts = [system, mode_text, f"Keep the final answer within roughly {max_tokens} tokens.", prompt]
    return "\n\n".join(part for part in parts if part)
