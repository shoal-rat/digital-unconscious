from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from du_research.ai_backend import AIBackend


WRITER_SYSTEM_PROMPT = """You are the Paper Writer agent.
Write a research manuscript in Markdown based on the provided literature, feasibility memo,
datasets, and analysis artifacts. Be concrete, cautious, and citation-aware.
Return ONLY JSON: {"manuscript_markdown": "..."}"""


@dataclass
class WriterAgent:
    backend: AIBackend
    model: str = "opus"
    system_prompt: str | None = None

    def write(self, payload: dict[str, Any]) -> str | None:
        response = self.backend.call(
            json.dumps(payload, indent=2, ensure_ascii=False),
            mode="balanced",
            system=self.system_prompt or WRITER_SYSTEM_PROMPT,
            model=self.model,
            max_tokens=4000,
        )
        if not response.ok:
            return None
        try:
            parsed = json.loads(response.text)
            return parsed.get("manuscript_markdown")
        except json.JSONDecodeError:
            return None
