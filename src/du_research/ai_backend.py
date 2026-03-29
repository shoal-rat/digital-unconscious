"""AIBackend abstraction layer — dual-mode support for Claude Code and Anthropic API.

Both backends expose the same ``call()`` interface so that every upstream agent
can switch between Claude Code (headless) and the Anthropic Python SDK without
changing a single line of agent logic.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------


class AIBackend(Protocol):
    """Common interface every backend must satisfy."""

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
    ) -> AIResponse:
        ...


@dataclass
class AIResponse:
    """Normalised response returned by every backend."""

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


# ---------------------------------------------------------------------------
# Mode → parameter mapping
# ---------------------------------------------------------------------------

_MODE_PARAMS: dict[str, dict[str, Any]] = {
    "creative": {"temperature": 0.95},
    "balanced": {"temperature": 0.7},
    "strict": {"temperature": 0.1},
    "deterministic": {"temperature": 0.0},
}

# Model tier shorthand → canonical model id
MODEL_ALIASES: dict[str, str] = {
    "opus": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5",
}


def _resolve_model(model: str | None, default: str = "claude-sonnet-4-6") -> str:
    if model is None:
        return default
    return MODEL_ALIASES.get(model, model)


# ---------------------------------------------------------------------------
# Claude Code backend (headless ``claude -p``)
# ---------------------------------------------------------------------------


@dataclass
class ClaudeCodeBackend:
    """Calls ``claude -p --bare`` as a subprocess.

    Temperature is **not** directly controllable via the CLI — creative
    divergence is simulated through prompt-engineering (instructions in the
    system prompt).  The ``mode`` parameter selects an ``--append-system-prompt``
    that steers behaviour.
    """

    timeout_seconds: int = 300
    model_override: str | None = None  # e.g. "opus"

    # Mode → extra system-prompt snippet injected via --append-system-prompt
    _MODE_PROMPTS: dict[str, str] = field(default_factory=lambda: {
        "creative": (
            "You MUST think divergently. Break conventional patterns. "
            "Generate multiple diverse candidates. Favour novelty over safety."
        ),
        "balanced": "",
        "strict": (
            "Be extremely precise and conservative. Only state facts you are "
            "highly confident about. Minimise speculation."
        ),
        "deterministic": (
            "Produce the single most likely correct answer. No creativity, "
            "no hedging — deterministic precision only."
        ),
    })

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
    ) -> AIResponse:
        cmd: list[str] = ["claude", "-p", prompt, "--output-format", "json"]

        if model or self.model_override:
            resolved = model or self.model_override
            if resolved in MODEL_ALIASES:
                resolved = MODEL_ALIASES[resolved]
            cmd.extend(["--model", resolved])

        mode_snippet = self._MODE_PROMPTS.get(mode, "")
        combined_system = "\n\n".join(p for p in [system, mode_snippet] if p)
        if combined_system:
            cmd.extend(["--append-system-prompt", combined_system])

        if session_id:
            cmd.extend(["--resume", session_id])

        if allowed_tools:
            cmd.extend(["--allowedTools", ",".join(allowed_tools)])

        if json_schema:
            cmd.extend(["--json-schema", json.dumps(json_schema)])

        cmd.extend(["--max-turns", "25"])

        logger.debug("ClaudeCodeBackend cmd: %s", " ".join(cmd[:6]))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return AIResponse(text="", raw={"error": "timeout"})
        except FileNotFoundError:
            return AIResponse(text="", raw={"error": "claude CLI not found"})

        # Claude Code returns JSON on stdout even on non-zero exit codes
        output = result.stdout.strip() or result.stderr.strip()
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            if result.returncode != 0:
                logger.warning("claude -p returned %d: %s", result.returncode, output[:500])
                return AIResponse(text="", raw={"error": output[:2000]})
            return AIResponse(text=output, raw={"raw_stdout": output[:2000]})

        # Check for is_error flag in Claude Code JSON response
        if parsed.get("is_error"):
            error_msg = parsed.get("result", "unknown error")
            logger.warning("claude -p error: %s", error_msg)
            return AIResponse(text="", raw={"error": error_msg, **parsed})

        return AIResponse(
            text=parsed.get("result", ""),
            model=model or self.model_override or "",
            session_id=parsed.get("session_id"),
            input_tokens=parsed.get("usage", {}).get("input_tokens", 0),
            output_tokens=parsed.get("usage", {}).get("output_tokens", 0),
            cost_usd=parsed.get("total_cost_usd", 0.0),
            raw=parsed,
            structured=parsed.get("structured_output"),
        )


# ---------------------------------------------------------------------------
# Anthropic API backend (direct SDK)
# ---------------------------------------------------------------------------


@dataclass
class AnthropicAPIBackend:
    """Calls the Anthropic Python SDK directly.

    Requires ``pip install anthropic`` and ``ANTHROPIC_API_KEY`` in the
    environment.
    """

    api_key: str | None = None
    default_model: str = "claude-sonnet-4-6"
    _client: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._client is not None:
            return
        try:
            import anthropic  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required for API mode: pip install anthropic"
            ) from exc
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY must be set for API mode")
        self._client = anthropic.Anthropic(api_key=key)

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
    ) -> AIResponse:
        resolved_model = _resolve_model(model, self.default_model)
        params = _MODE_PARAMS.get(mode, _MODE_PARAMS["balanced"])
        temperature = params["temperature"]

        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        try:
            response = self._client.messages.create(**kwargs)
        except Exception as exc:
            logger.error("Anthropic API error: %s", exc)
            return AIResponse(text="", raw={"error": str(exc)})

        text_parts = [
            block.text for block in response.content if hasattr(block, "text")
        ]
        full_text = "\n".join(text_parts)

        structured = None
        if json_schema:
            try:
                structured = json.loads(full_text)
            except json.JSONDecodeError:
                pass

        return AIResponse(
            text=full_text,
            model=resolved_model,
            session_id=session_id,
            input_tokens=getattr(response.usage, "input_tokens", 0),
            output_tokens=getattr(response.usage, "output_tokens", 0),
            cost_usd=0.0,  # calculated downstream
            raw={"stop_reason": response.stop_reason},
            structured=structured,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_backend(mode: str = "auto", **kwargs: Any) -> AIBackend:
    """Create the appropriate backend based on *mode*.

    ``auto`` — pick API if ``ANTHROPIC_API_KEY`` is set, otherwise Claude Code.
    ``claude_code`` — always use headless Claude Code CLI.
    ``api`` — always use the Anthropic Python SDK.
    """
    if mode == "auto":
        if os.environ.get("ANTHROPIC_API_KEY"):
            mode = "api"
        else:
            mode = "claude_code"

    if mode == "api":
        return AnthropicAPIBackend(**kwargs)
    return ClaudeCodeBackend(**kwargs)
