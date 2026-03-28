"""Compression agent — distills raw behaviour frames into structured summaries.

Uses Claude Haiku (fast, cheap) to compress a 30-minute window of screen
observations into a 200-400 word structured JSON summary, preserving the
signals that matter for idea generation.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from du_research.ai_backend import AIBackend, AIResponse
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

    def compress(self, frames: list[BehaviorFrame]) -> dict[str, Any]:
        """Compress *frames* into a structured behaviour summary."""
        if not frames:
            return _empty_summary()

        # Build the input for the AI
        frame_descriptions = []
        for f in frames[:80]:  # cap to avoid token overflow
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

        frames_text = "\n".join(frame_descriptions)
        prompt = (
            f"Compress the following {len(frames)} behaviour observation frames "
            f"into a structured JSON summary:\n\n{frames_text}"
        )

        response = self.backend.call(
            prompt,
            mode="strict",
            system=self.system_prompt or COMPRESSOR_SYSTEM_PROMPT,
            model=self.model,
            max_tokens=800,
        )

        if not response.ok:
            logger.warning("Compression failed, using heuristic fallback")
            return _heuristic_compress(frames)

        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            text = response.text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            logger.warning("Could not parse compression output, using heuristic")
            return _heuristic_compress(frames)


def _empty_summary() -> dict[str, Any]:
    return {
        "time_range": "",
        "dominant_topics": [],
        "high_weight_content": [],
        "app_distribution": {},
        "intent_signals": [],
        "cross_domain_hints": [],
        "search_queries": [],
    }


def _heuristic_compress(frames: list[BehaviorFrame]) -> dict[str, Any]:
    """Fallback compression without AI — uses simple heuristics."""
    from collections import Counter
    from du_research.utils import tokenize

    app_counts: Counter[str] = Counter()
    all_text_tokens: list[str] = []
    high_weight: list[dict[str, Any]] = []

    for f in frames:
        app_counts[f.app_name] += 1
        tokens = tokenize(f"{f.window_title} {f.text_content}")
        all_text_tokens.extend(tokens)
        if f.dwell_seconds > 30:
            high_weight.append({
                "content": f.text_content[:200],
                "dwell_time": f"{f.dwell_seconds / 60:.1f}min",
                "weight": min(f.dwell_seconds / 300, 1.0),
            })

    total_apps = sum(app_counts.values()) or 1
    topic_counter = Counter(all_text_tokens)
    dominant = [word for word, _ in topic_counter.most_common(8)]

    time_range = ""
    if frames:
        time_range = f"{frames[0].timestamp[:16]}-{frames[-1].timestamp[:16]}"

    return {
        "time_range": time_range,
        "dominant_topics": dominant[:6],
        "high_weight_content": sorted(
            high_weight, key=lambda x: x["weight"], reverse=True
        )[:5],
        "app_distribution": {
            app: round(count / total_apps, 2)
            for app, count in app_counts.most_common(6)
        },
        "intent_signals": dominant[:3],
        "cross_domain_hints": [],
        "search_queries": [],
    }
