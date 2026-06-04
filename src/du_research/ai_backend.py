"""AI backend abstraction layer.

The project can run against Claude Code, Anthropic API, OpenAI API, or
Kimi/Moonshot API while exposing one small call interface to the rest of the
pipeline.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


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
        use_chrome: bool = False,
        max_turns: int = 25,
        agent: str | None = None,
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
# Shared model and mode mappings
# ---------------------------------------------------------------------------

_MODE_PARAMS: dict[str, dict[str, Any]] = {
    "creative": {"temperature": 0.95},
    "balanced": {"temperature": 0.7},
    "strict": {"temperature": 0.1},
    "deterministic": {"temperature": 0.0},
}

ANTHROPIC_MODEL_ALIASES: dict[str, str] = {
    "opus": "claude-opus-4-8",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5",
}

OPENAI_MODEL_ALIASES: dict[str, str] = {
    "codex": "gpt-5.5",
    "gpt": "gpt-5.5",
    "opus": "gpt-5.5",
    "sonnet": "gpt-5.5",
    "haiku": "gpt-5.4-mini",
}

KIMI_MODEL_ALIASES: dict[str, str] = {
    "kimi": "kimi-k2.6",
    "k2": "kimi-k2.6",
    "k2.6": "kimi-k2.6",
    "opus": "kimi-k2.6",
    "sonnet": "kimi-k2.6",
    "haiku": "kimi-k2.6",
}

# Backward-compatible export used by older callers/tests.
MODEL_ALIASES = ANTHROPIC_MODEL_ALIASES

_PROVIDER_PREFIXES = {
    "anthropic",
    "claude",
    "claude_code",
    "openai",
    "codex",
    "kimi",
    "moonshot",
}


def _split_provider_model(model: str | None) -> tuple[str | None, str | None]:
    if not model or ":" not in model:
        return None, model
    provider, bare_model = model.split(":", 1)
    provider = provider.strip().lower()
    if provider in _PROVIDER_PREFIXES:
        return provider, bare_model.strip()
    return None, model


def _resolve_model(model: str | None, default: str = "claude-sonnet-4-6") -> str:
    _, bare_model = _split_provider_model(model)
    if bare_model is None:
        return default
    return ANTHROPIC_MODEL_ALIASES.get(bare_model, bare_model)


def _resolve_provider_model(
    model: str | None,
    *,
    default: str,
    aliases: dict[str, str],
) -> str:
    _, bare_model = _split_provider_model(model)
    if bare_model is None:
        return default
    return aliases.get(bare_model, bare_model)


def _parse_structured_text(text: str, json_schema: dict | None) -> dict | None:
    if not json_schema:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else {"result": parsed}


def _usage_value(usage: Any, key: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(key, 0) or 0)
    return int(getattr(usage, key, 0) or 0)


# ---------------------------------------------------------------------------
# Claude Code backend (headless ``claude -p``)
# ---------------------------------------------------------------------------


@dataclass
class ClaudeCodeBackend:
    """Calls Claude Code CLI as a subprocess.

    The flags are best-effort: older Claude Code versions may reject a newer
    flag, in which case the raw CLI error is returned as a failed AIResponse.
    """

    timeout_seconds: int = 300
    model_override: str | None = None

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
            "no hedging: deterministic precision only."
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
        use_chrome: bool = False,
        max_turns: int = 25,
        agent: str | None = None,
    ) -> AIResponse:
        cmd: list[str] = [
            "claude",
            "-p", prompt,
            "--output-format", "json",
            "--permission-mode", "auto",
        ]

        if agent:
            cmd.extend(["--agent", agent])

        requested_model = model or self.model_override
        if requested_model:
            cmd.extend(["--model", _resolve_model(requested_model)])

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

        if use_chrome:
            cmd.append("--chrome")

        cmd.extend(["--max-turns", str(max_turns)])

        logger.debug(
            "ClaudeCodeBackend: %s (model=%s, mode=%s, tools=%s, chrome=%s)",
            prompt[:80],
            requested_model,
            mode,
            allowed_tools,
            use_chrome,
        )
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

        output = result.stdout.strip() or result.stderr.strip()
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            if result.returncode != 0:
                logger.warning("claude -p returned %d: %s", result.returncode, output[:500])
                return AIResponse(text="", raw={"error": output[:2000]})
            return AIResponse(text=output, raw={"raw_stdout": output[:2000]})

        if parsed.get("is_error"):
            error_msg = parsed.get("result", "unknown error")
            logger.warning("claude -p error: %s", error_msg)
            return AIResponse(text="", raw={"error": error_msg, **parsed})

        text = parsed.get("result", "")
        structured = parsed.get("structured_output") or _parse_structured_text(text, json_schema)
        usage = parsed.get("usage", {})
        return AIResponse(
            text=text,
            model=_resolve_model(requested_model) if requested_model else "",
            session_id=parsed.get("session_id"),
            input_tokens=_usage_value(usage, "input_tokens"),
            output_tokens=_usage_value(usage, "output_tokens"),
            cost_usd=parsed.get("total_cost_usd", 0.0),
            raw=parsed,
            structured=structured,
        )


# ---------------------------------------------------------------------------
# Anthropic API backend (direct SDK)
# ---------------------------------------------------------------------------


@dataclass
class AnthropicAPIBackend:
    """Calls the Anthropic Python SDK directly."""

    api_key: str | None = None
    default_model: str = "claude-sonnet-4-6"
    _client: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._client is not None:
            return
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required for Anthropic API mode: pip install anthropic"
            ) from exc
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY must be set for Anthropic API mode")
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
        use_chrome: bool = False,
        max_turns: int = 25,
        agent: str | None = None,
    ) -> AIResponse:
        resolved_model = _resolve_model(model, self.default_model)
        params = _MODE_PARAMS.get(mode, _MODE_PARAMS["balanced"])
        temperature = params["temperature"]

        system_parts = [system]
        if json_schema:
            system_parts.append(
                "Return only a JSON object that conforms to this schema: "
                + json.dumps(json_schema, ensure_ascii=False)
            )
        combined_system = "\n\n".join(p for p in system_parts if p)

        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if combined_system:
            kwargs["system"] = combined_system

        try:
            response = self._client.messages.create(**kwargs)
        except Exception as exc:
            logger.error("Anthropic API error: %s", exc)
            return AIResponse(text="", raw={"error": str(exc), "provider": "anthropic"})

        text_parts = [
            block.text for block in response.content if hasattr(block, "text")
        ]
        full_text = "\n".join(text_parts)
        usage = getattr(response, "usage", None)

        return AIResponse(
            text=full_text,
            model=resolved_model,
            session_id=session_id,
            input_tokens=_usage_value(usage, "input_tokens"),
            output_tokens=_usage_value(usage, "output_tokens"),
            cost_usd=0.0,
            raw={"stop_reason": getattr(response, "stop_reason", None), "provider": "anthropic"},
            structured=_parse_structured_text(full_text, json_schema),
        )


# ---------------------------------------------------------------------------
# OpenAI-compatible chat backends
# ---------------------------------------------------------------------------


@dataclass
class OpenAICompatibleChatBackend:
    """Backend for OpenAI SDK compatible chat completions providers."""

    api_key: str | None = None
    default_model: str = "gpt-5.5"
    base_url: str | None = None
    env_key: str = "OPENAI_API_KEY"
    alternate_env_keys: tuple[str, ...] = ()
    provider_name: str = "openai"
    model_aliases: dict[str, str] = field(default_factory=lambda: OPENAI_MODEL_ALIASES.copy())
    omit_temperature: bool = False
    kimi_thinking: bool | None = None
    _client: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAI-compatible API modes: pip install openai"
            ) from exc
        key = self.api_key or os.environ.get(self.env_key)
        if not key:
            for env_key in self.alternate_env_keys:
                key = os.environ.get(env_key)
                if key:
                    break
        if not key:
            all_keys = ", ".join((self.env_key, *self.alternate_env_keys))
            raise ValueError(f"{all_keys} must be set for {self.provider_name} API mode")
        kwargs: dict[str, Any] = {"api_key": key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)

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
    ) -> AIResponse:
        resolved_model = _resolve_provider_model(
            model,
            default=self.default_model,
            aliases=self.model_aliases,
        )
        params = _MODE_PARAMS.get(mode, _MODE_PARAMS["balanced"])
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        if json_schema:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Return only a JSON object that conforms to this schema: "
                        + json.dumps(json_schema, ensure_ascii=False)
                    ),
                }
            )
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if not self.omit_temperature:
            kwargs["temperature"] = params["temperature"]

        if json_schema and self.provider_name == "openai":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "digital_unconscious_response",
                    "schema": json_schema,
                    "strict": False,
                },
            }

        if self.provider_name in {"kimi", "moonshot"}:
            thinking_enabled = self.kimi_thinking
            if thinking_enabled is None:
                thinking_enabled = not (json_schema or mode == "deterministic")
            kwargs["extra_body"] = {
                "thinking": {"type": "enabled" if thinking_enabled else "disabled"}
            }

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            logger.error("%s API error: %s", self.provider_name, exc)
            return AIResponse(text="", raw={"error": str(exc), "provider": self.provider_name})

        choice = response.choices[0] if getattr(response, "choices", None) else None
        message = getattr(choice, "message", None)
        full_text = getattr(message, "content", "") or ""
        usage = getattr(response, "usage", None)

        return AIResponse(
            text=full_text,
            model=resolved_model,
            session_id=session_id,
            input_tokens=_usage_value(usage, "prompt_tokens"),
            output_tokens=_usage_value(usage, "completion_tokens"),
            cost_usd=0.0,
            raw={
                "provider": self.provider_name,
                "finish_reason": getattr(choice, "finish_reason", None),
                "ignored_tools": bool(allowed_tools or use_chrome or agent),
            },
            structured=_parse_structured_text(full_text, json_schema),
        )


@dataclass
class OpenAIAPIBackend(OpenAICompatibleChatBackend):
    provider_name: str = "openai"
    env_key: str = "OPENAI_API_KEY"
    default_model: str = "gpt-5.5"
    model_aliases: dict[str, str] = field(default_factory=lambda: OPENAI_MODEL_ALIASES.copy())


@dataclass
class KimiAPIBackend(OpenAICompatibleChatBackend):
    provider_name: str = "kimi"
    env_key: str = "MOONSHOT_API_KEY"
    alternate_env_keys: tuple[str, ...] = ("KIMI_API_KEY",)
    base_url: str | None = "https://api.moonshot.ai/v1"
    default_model: str = "kimi-k2.6"
    model_aliases: dict[str, str] = field(default_factory=lambda: KIMI_MODEL_ALIASES.copy())
    omit_temperature: bool = True


# ---------------------------------------------------------------------------
# Multi-provider router
# ---------------------------------------------------------------------------


@dataclass
class MultiProviderBackend:
    """Routes calls by provider prefix or available credentials.

    Examples:
    - ``model="openai:gpt-5.5"`` uses OpenAI.
    - ``model="kimi:kimi-k2.6"`` uses Kimi/Moonshot.
    - ``model="claude_code:opus"`` uses the local Claude Code CLI.
    - no prefix falls back to Anthropic API, OpenAI, Kimi, then Claude Code.
    """

    api_key: str | None = None
    openai_api_key: str | None = None
    kimi_api_key: str | None = None
    default_model: str = "claude-sonnet-4-6"
    openai_default_model: str = "gpt-5.5"
    kimi_default_model: str = "kimi-k2.6"
    timeout_seconds: int = 300
    _providers: dict[str, AIBackend] = field(default_factory=dict, repr=False)

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
    ) -> AIResponse:
        provider, bare_model = self._select_provider(model)
        try:
            backend = self._get_provider(provider)
        except Exception as exc:
            logger.error("Could not initialize %s backend: %s", provider, exc)
            return AIResponse(text="", raw={"error": str(exc), "provider": provider})
        return backend.call(
            prompt,
            mode=mode,
            system=system,
            model=bare_model,
            max_tokens=max_tokens,
            json_schema=json_schema,
            session_id=session_id,
            allowed_tools=allowed_tools,
            use_chrome=use_chrome,
            max_turns=max_turns,
            agent=agent,
        )

    def _select_provider(self, model: str | None) -> tuple[str, str | None]:
        prefix, bare_model = _split_provider_model(model)
        if prefix in {"openai", "codex"}:
            return "openai", bare_model
        if prefix in {"kimi", "moonshot"}:
            return "kimi", bare_model
        if prefix == "claude_code":
            return "claude_code", bare_model
        if prefix in {"anthropic", "claude"}:
            return "anthropic", bare_model
        if self.api_key or os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic", bare_model
        if self.openai_api_key or os.environ.get("OPENAI_API_KEY"):
            return "openai", bare_model
        if self.kimi_api_key or os.environ.get("MOONSHOT_API_KEY") or os.environ.get("KIMI_API_KEY"):
            return "kimi", bare_model
        return "claude_code", bare_model

    def _get_provider(self, provider: str) -> AIBackend:
        if provider in self._providers:
            return self._providers[provider]
        if provider == "anthropic":
            backend: AIBackend = AnthropicAPIBackend(
                api_key=self.api_key,
                default_model=self.default_model,
            )
        elif provider == "openai":
            backend = OpenAIAPIBackend(
                api_key=self.openai_api_key,
                default_model=self.openai_default_model,
            )
        elif provider == "kimi":
            backend = KimiAPIBackend(
                api_key=self.kimi_api_key,
                default_model=self.kimi_default_model,
            )
        else:
            backend = ClaudeCodeBackend(timeout_seconds=self.timeout_seconds)
        self._providers[provider] = backend
        return backend


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_backend(mode: str = "auto", **kwargs: Any) -> AIBackend:
    """Create the appropriate backend based on ``mode``.

    Modes:
    - ``auto``: Anthropic API, OpenAI API, Kimi API, then Claude Code.
    - ``multi``: route each call by provider prefix or available credentials.
    - ``claude_code``: local Claude Code CLI.
    - ``api``/``anthropic``: Anthropic API.
    - ``openai``/``codex``: OpenAI API.
    - ``kimi``/``moonshot``: Kimi/Moonshot API.
    """
    mode = (mode or "auto").strip().lower()

    api_key = kwargs.get("api_key")
    openai_api_key = kwargs.get("openai_api_key")
    kimi_api_key = kwargs.get("kimi_api_key") or kwargs.get("moonshot_api_key")
    default_model = kwargs.get("default_model")

    if mode == "auto":
        if api_key or os.environ.get("ANTHROPIC_API_KEY"):
            mode = "api"
        elif openai_api_key or os.environ.get("OPENAI_API_KEY"):
            mode = "openai"
        elif kimi_api_key or os.environ.get("MOONSHOT_API_KEY") or os.environ.get("KIMI_API_KEY"):
            mode = "kimi"
        else:
            mode = "claude_code"

    if mode in {"multi", "router"}:
        return MultiProviderBackend(
            api_key=api_key,
            openai_api_key=openai_api_key,
            kimi_api_key=kimi_api_key,
            default_model=default_model or "claude-sonnet-4-6",
            openai_default_model=kwargs.get("openai_default_model", "gpt-5.5"),
            kimi_default_model=kwargs.get("kimi_default_model", "kimi-k2.6"),
            timeout_seconds=kwargs.get("timeout_seconds", 300),
        )

    if mode in {"api", "anthropic", "claude"}:
        return AnthropicAPIBackend(
            api_key=api_key,
            default_model=default_model or "claude-sonnet-4-6",
        )

    if mode in {"openai", "codex"}:
        return OpenAIAPIBackend(
            api_key=openai_api_key or api_key,
            default_model=kwargs.get("openai_default_model") or default_model or "gpt-5.5",
        )

    if mode in {"kimi", "moonshot"}:
        return KimiAPIBackend(
            api_key=kimi_api_key or api_key,
            default_model=kwargs.get("kimi_default_model") or default_model or "kimi-k2.6",
        )

    return ClaudeCodeBackend(
        timeout_seconds=kwargs.get("timeout_seconds", 300),
        model_override=kwargs.get("model_override"),
    )
