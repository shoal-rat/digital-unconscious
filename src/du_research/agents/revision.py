from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from du_research.ai_backend import AIBackend


REVISION_SYSTEM_PROMPT = """You are the Revision Agent.
Revise the manuscript to address the critique list while staying conservative.
Return ONLY JSON: {"revised_markdown": "..."}"""


@dataclass
class RevisionAgent:
    backend: AIBackend
    model: str = "sonnet"
    system_prompt: str | None = None

    def revise(self, manuscript_text: str, review_payload: dict[str, Any]) -> str | None:
        response = self.backend.call(
            json.dumps(
                {"manuscript": manuscript_text, "review": review_payload},
                indent=2,
                ensure_ascii=False,
            ),
            mode="balanced",
            system=self.system_prompt or REVISION_SYSTEM_PROMPT,
            model=self.model,
            max_tokens=4000,
        )
        if not response.ok:
            return None
        try:
            return json.loads(response.text).get("revised_markdown")
        except json.JSONDecodeError:
            return None
