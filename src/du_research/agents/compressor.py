"""Compression agent — distills raw behaviour frames into structured summaries.

Uses Claude Haiku (fast, cheap) to compress a 30-minute window of screen
observations into a 200-400 word structured JSON summary, preserving the
signals that matter for idea generation.

LLM-only: no heuristic fallback. If the AI is unreachable, returns None
so the engine can queue the task for later.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from du_research.ai_backend import AIBackend
from du_research.observation import BehaviorFrame

logger = logging.getLogger(__name__)

COMPRESSOR_SYSTEM_PROMPT = """\
You are a behaviour-compression specialist for the Digital Unconscious system.

Your job: take raw screen observation frames and produce a structured JSON
summary that captures the **intent signals** behind the behaviour — what the
human was thinking about, researching, or exploring.

Rules:
- Output ONLY valid JSON (no markdown, no explanation).
- Keep the summary between 200-400 words across all fields combined.
- Preserve: dominant topics, high-weight content (by dwell time), search
  queries, app distribution, cross-domain hints.
- Discard: duplicate content, system UI noise, mechanical navigation.
- Weight content by dwell_seconds — longer dwell = more important.
- Identify intent_signals: what was the human trying to learn or solve?
- Identify cross_domain_hints: unexpected topic combinations that may
  spark creative ideas.

Output schema:
{
  "time_range": "HH:MM-HH:MM",
  "dominant_topics": ["topic1", "topic2", ...],
  "high_weight_content": [
    {"content": "...", "dwell_time": "Xmin", "weight": 0.0-1.0}
  ],
  "app_distribution": {"app_name": fraction, ...},
  "intent_signals": ["signal1", "signal2", ...],
  "cross_domain_hints": ["hint1", "hint2", ...],
  "search_queries": ["query1", ...]
}
"""


@dataclass
class CompressionAgent:
    """Compresses a window of behaviour frames into a structured summary."""

    backend: AIBackend
    model: str = "haiku"
    system_prompt: str | None = None

    def compress(self, frames: list[BehaviorFrame]) -> dict[str, Any] | None:
        """Compress *frames* into a structured behaviour summary.

        Returns None if the LLM is unavailable (caller should queue).
        """
        if not frames:
            return {"time_range": "", "dominant_topics": [], "high_weight_content": [],
                    "app_distribution": {}, "intent_signals": [], "cross_domain_hints": [],
                    "search_queries": []}

        frame_descriptions = []
        for f in frames[:80]:
            desc = (
                f"[{f.timestamp}] app={f.app_name} "
                f"title=\"{f.window_title[:100]}\" "
                f"dwell={f.dwell_seconds:.0f}s "
                f"type={f.frame_type}"
            )
            if f.text_content:
                desc += f" text=\"{f.text_content[:200]}\""
            if f.url:
                desc += f" url={f.url}"
            frame_descriptions.append(desc)

        prompt = (
            f"Compress the following {len(frames)} behaviour observation frames "
            f"into a structured JSON summary:\n\n" + "\n".join(frame_descriptions)
        )

        schema = {
            "type": "object",
            "properties": {
                "time_range": {"type": "string"},
                "dominant_topics": {"type": "array", "items": {"type": "string"}},
                "high_weight_content": {"type": "array"},
                "app_distribution": {"type": "object"},
                "intent_signals": {"type": "array", "items": {"type": "string"}},
                "cross_domain_hints": {"type": "array", "items": {"type": "string"}},
                "search_queries": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["time_range", "dominant_topics", "intent_signals"],
        }

        response = self.backend.call(
            prompt,
            mode="strict",
            system=self.system_prompt or COMPRESSOR_SYSTEM_PROMPT,
            model=self.model,
            max_tokens=800,
            json_schema=schema,
        )

        if not response.ok:
            logger.warning("Compression failed — LLM unavailable")
            return None

        # Prefer structured_output from --json-schema, fallback to text parsing
        if response.structured:
            return response.structured
        return _parse_json(response.text)


def _parse_json(text: str) -> dict[str, Any] | None:
    """Try to extract JSON from potentially messy AI output."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    logger.warning("Could not parse compression output")
    return None
