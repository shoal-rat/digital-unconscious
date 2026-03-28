from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from du_research.ai_backend import AIBackend


REVIEWER_SYSTEM_PROMPT = """You are an adversarial peer reviewer.
Score the manuscript on novelty, statistical_rigor, clarity, reproducibility, figure_quality,
abstract_accuracy, overreach_detection, and reference_quality, all 0-100.
Return ONLY JSON:
{"dimension_scores": {...}, "critique_types": [...], "suggestions": [...]}"""


@dataclass
class ReviewerAgent:
    backend: AIBackend
    model: str = "sonnet"
    system_prompt: str | None = None

    def review(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        response = self.backend.call(
            json.dumps(payload, indent=2, ensure_ascii=False),
            mode="strict",
            system=self.system_prompt or REVIEWER_SYSTEM_PROMPT,
            model=self.model,
            max_tokens=2500,
        )
        if not response.ok:
            return None
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            return None
