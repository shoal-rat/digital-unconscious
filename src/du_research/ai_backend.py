"""AI backend abstraction layer.

The project can run against Claude Code, Anthropic API, OpenAI API, or
Kimi/Moonshot API while exposing one small call interface to the rest of the
pipeline.
"""
from __future__ import annotations

import base64
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
        think: int | str | None = None,
        images: list[bytes] | None = None,
        web_search: bool = False,
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
        # The model may wrap JSON in prose (e.g. when web search adds citations).
        start, end = text.find("{"), text.rfind("}") + 1
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(text[start:end])
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
# Extended-thinking helpers
# ---------------------------------------------------------------------------
#
# ``think`` is a single knob exposed on every backend so the rest of the
# pipeline can ask a model to reason harder without caring which provider
# serves the call. It accepts an integer token budget, a "low"/"medium"/"high"
# string, or a bool. Each backend translates it into its native control:
# Anthropic extended thinking (budget tokens), OpenAI reasoning effort, the
# Kimi thinking toggle, or a Claude Code thinking keyword.

_EFFORT_BUDGETS = {"low": 2048, "medium": 8192, "high": 16384}


def _thinking_budget(think: int | str | bool | None) -> int:
    """Normalise ``think`` to an Anthropic-style token budget (0 == off)."""
    if think is None or think is False:
        return 0
    if think is True:
        return _EFFORT_BUDGETS["medium"]
    if isinstance(think, str):
        return _EFFORT_BUDGETS.get(think.strip().lower(), 0)
    try:
        return max(0, int(think))
    except (TypeError, ValueError):
        return 0


def _reasoning_effort(think: int | str | bool | None) -> str | None:
    """Normalise ``think`` to an OpenAI reasoning-effort label (None == off)."""
    if isinstance(think, str):
        label = think.strip().lower()
        return label if label in _EFFORT_BUDGETS else "medium"
    budget = _thinking_budget(think)
    if budget <= 0:
        return None
    if budget <= _EFFORT_BUDGETS["low"]:
        return "low"
    if budget <= _EFFORT_BUDGETS["high"] - 1:
        return "medium"
    return "high"


def _claude_code_think_keyword(think: int | str | bool | None) -> str | None:
    """Map ``think`` onto a Claude Code thinking trigger keyword."""
    budget = _thinking_budget(think)
    if budget <= 0:
        return None
    if budget >= _EFFORT_BUDGETS["high"]:
        return "ultrathink"
    if budget >= _EFFORT_BUDGETS["medium"]:
        return "think harder"
    return "think hard"


# ---------------------------------------------------------------------------
# Multimodal image helpers (vision input)
# ---------------------------------------------------------------------------


def _b64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode("ascii")


def _image_media_type(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _anthropic_user_content(prompt: str, images: list[bytes] | None):
    if not images:
        return prompt
    blocks: list[dict[str, Any]] = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": _image_media_type(img), "data": _b64(img)},
        }
        for img in images
    ]
    blocks.append({"type": "text", "text": prompt})
    return blocks


def _openai_user_content(prompt: str, images: list[bytes] | None):
    if not images:
        return prompt
    parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in images:
        url = f"data:{_image_media_type(img)};base64,{_b64(img)}"
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


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
        think: int | str | None = None,
        images: list[bytes] | None = None,
        web_search: bool = False,
    ) -> AIResponse:
        # The headless Claude Code CLI does not take inline images; vision input
        # requires an API backend. Ignore images here rather than erroring.
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
        think_keyword = _claude_code_think_keyword(think)
        think_snippet = (
            f"Before answering, {think_keyword} about the problem step by step."
            if think_keyword
            else ""
        )
        combined_system = "\n\n".join(p for p in [system, mode_snippet, think_snippet] if p)
        if combined_system:
            cmd.extend(["--append-system-prompt", combined_system])

        if session_id:
            cmd.extend(["--resume", session_id])

        tools = list(allowed_tools or [])
        if web_search and "WebSearch" not in tools:
            tools.append("WebSearch")
        if tools:
            cmd.extend(["--allowedTools", ",".join(tools)])

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
        think: int | str | None = None,
        images: list[bytes] | None = None,
        web_search: bool = False,
    ) -> AIResponse:
        resolved_model = _resolve_model(model, self.default_model)
        params = _MODE_PARAMS.get(mode, _MODE_PARAMS["balanced"])
        temperature = params["temperature"]
        thinking_budget = _thinking_budget(think)

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
            "messages": [{"role": "user", "content": _anthropic_user_content(prompt, images)}],
        }
        if thinking_budget > 0:
            # Extended thinking needs token headroom beyond the budget and must
            # keep the default temperature (a custom temperature is rejected).
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
            if max_tokens <= thinking_budget:
                kwargs["max_tokens"] = thinking_budget + max_tokens
        else:
            kwargs["temperature"] = temperature
        if combined_system:
            kwargs["system"] = combined_system
        if web_search:
            # Server-side web search: Claude decides when to search, capped by max_uses.
            kwargs["tools"] = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]

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
        think: int | str | None = None,
        images: list[bytes] | None = None,
        web_search: bool = False,
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
        messages.append({"role": "user", "content": _openai_user_content(prompt, images)})

        effort = _reasoning_effort(think)
        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        # Reasoning models reject a custom sampling temperature, so only send a
        # temperature when no reasoning effort was requested.
        if not self.omit_temperature and not (effort and self.provider_name == "openai"):
            kwargs["temperature"] = params["temperature"]

        if effort and self.provider_name == "openai":
            kwargs["reasoning_effort"] = effort

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
            if think:
                thinking_enabled = _thinking_budget(think) > 0
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
                "ignored_tools": bool(allowed_tools or use_chrome or agent or web_search),
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
    enable_fallback: bool = True
    fallback_order: list[str] = field(
        default_factory=lambda: ["anthropic", "openai", "kimi", "claude_code"]
    )
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
        think: int | str | None = None,
        images: list[bytes] | None = None,
        web_search: bool = False,
    ) -> AIResponse:
        provider, bare_model = self._select_provider(model)
        chain = self._provider_chain(provider)
        last = AIResponse(text="", raw={"error": "no provider available", "provider": provider})
        for index, prov in enumerate(chain):
            try:
                backend = self._get_provider(prov)
            except Exception as exc:
                logger.warning("Skipping %s backend (initialisation failed): %s", prov, exc)
                last = AIResponse(text="", raw={"error": str(exc), "provider": prov})
                continue
            # Only the primary provider receives the caller's requested model; a
            # fallback provider uses its own default model instead.
            call_model = bare_model if prov == provider else None
            response = backend.call(
                prompt,
                mode=mode,
                system=system,
                model=call_model,
                max_tokens=max_tokens,
                json_schema=json_schema,
                session_id=session_id,
                allowed_tools=allowed_tools,
                use_chrome=use_chrome,
                max_turns=max_turns,
                agent=agent,
                think=think,
                images=images,
                web_search=web_search,
            )
            if isinstance(response.raw, dict):
                response.raw.setdefault("router_provider", prov)
                if index > 0:
                    response.raw["router_fallback_from"] = provider
            if response.ok:
                if index > 0:
                    logger.info(
                        "Router failed over to %s after primary provider %s failed",
                        prov,
                        provider,
                    )
                return response
            error = response.raw.get("error", "empty response") if isinstance(response.raw, dict) else "empty response"
            logger.warning(
                "Provider %s returned no usable response (%s); %s",
                prov,
                error,
                "falling back to next provider" if index + 1 < len(chain) else "no fallback left",
            )
            last = response
        return last

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

    def _provider_available(self, provider: str) -> bool:
        """Whether a provider can be tried: already built, has a key, or is local."""
        if provider in self._providers:
            return True
        if provider == "anthropic":
            return bool(self.api_key or os.environ.get("ANTHROPIC_API_KEY"))
        if provider == "openai":
            return bool(self.openai_api_key or os.environ.get("OPENAI_API_KEY"))
        if provider == "kimi":
            return bool(
                self.kimi_api_key
                or os.environ.get("MOONSHOT_API_KEY")
                or os.environ.get("KIMI_API_KEY")
            )
        if provider == "claude_code":
            return True  # local CLI; if absent it returns a clean error and the chain ends
        return False

    def _provider_chain(self, primary: str) -> list[str]:
        """Ordered providers to attempt: the primary first, then available fallbacks."""
        if not self.enable_fallback:
            return [primary]
        chain = [primary]
        for prov in self.fallback_order:
            if prov != primary and prov not in chain and self._provider_available(prov):
                chain.append(prov)
        return chain


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
    enable_fallback = kwargs.get("enable_fallback", True)
    fallback_order = kwargs.get("fallback_order") or ["anthropic", "openai", "kimi", "claude_code"]

    has_anthropic = bool(api_key or os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(openai_api_key or os.environ.get("OPENAI_API_KEY"))
    has_kimi = bool(
        kimi_api_key or os.environ.get("MOONSHOT_API_KEY") or os.environ.get("KIMI_API_KEY")
    )

    # When at least one hosted provider is configured and fallback is enabled,
    # auto mode uses the multi-provider router so a failing primary provider
    # transparently fails over (ultimately to the local Claude Code CLI). With no
    # hosted keys there is nothing to fail over to, so auto stays single-backend.
    if mode == "auto" and enable_fallback and (has_anthropic or has_openai or has_kimi):
        mode = "multi"

    if mode == "auto":
        if has_anthropic:
            mode = "api"
        elif has_openai:
            mode = "openai"
        elif has_kimi:
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
            enable_fallback=enable_fallback,
            fallback_order=fallback_order,
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


# ---------------------------------------------------------------------------
# Routing inspection (shared by the CLI and dashboard)
# ---------------------------------------------------------------------------


def resolve_routing(config: Any) -> dict[str, Any]:
    """Resolve how each agent's model alias maps to a provider, given config.

    Drives ``du models`` and the dashboard routing panel. Provider selection for a
    prefix-less alias follows the same first-available order the router uses
    (``fallback_order``), so the displayed provider matches runtime behaviour.
    """
    ai = config.ai
    available = {
        "anthropic": bool(getattr(ai, "api_key", "") or os.environ.get("ANTHROPIC_API_KEY")),
        "openai": bool(getattr(ai, "openai_api_key", "") or os.environ.get("OPENAI_API_KEY")),
        "kimi": bool(
            getattr(ai, "kimi_api_key", "")
            or os.environ.get("MOONSHOT_API_KEY")
            or os.environ.get("KIMI_API_KEY")
        ),
        "claude_code": True,
    }
    order = list(getattr(ai, "fallback_order", ["anthropic", "openai", "kimi", "claude_code"]))
    alias_tables = {
        "anthropic": ANTHROPIC_MODEL_ALIASES,
        "openai": OPENAI_MODEL_ALIASES,
        "kimi": KIMI_MODEL_ALIASES,
        "claude_code": ANTHROPIC_MODEL_ALIASES,
    }

    def first_available() -> str:
        for provider in order:
            if provider != "claude_code" and available.get(provider):
                return provider
        return "claude_code"

    def resolve(alias: str) -> tuple[str, str]:
        prefix, bare = _split_provider_model(alias)
        if prefix in {"openai", "codex"}:
            provider = "openai"
        elif prefix in {"kimi", "moonshot"}:
            provider = "kimi"
        elif prefix == "claude_code":
            provider = "claude_code"
        elif prefix in {"anthropic", "claude"}:
            provider = "anthropic"
        else:
            provider = first_available()
        return provider, alias_tables[provider].get(bare or "", bare or "")

    agents = [
        ("compressor", ai.compressor_model),
        ("idea_generator", ai.creative_model),
        ("judge", ai.judge_model),
        ("briefing", ai.briefing_model),
        ("writer", ai.writer_model),
        ("reviewer", ai.reviewer_model),
        ("revision", ai.revision_model),
        ("analysis", ai.analysis_model),
    ]
    routing = []
    for name, alias in agents:
        provider, model = resolve(alias)
        routing.append({
            "agent": name,
            "configured": alias,
            "provider": provider,
            "model": model,
            "available": available.get(provider, False),
        })
    return {
        "mode": ai.mode,
        "fallback": ai.fallback,
        "fallback_order": order,
        "available_providers": [p for p, ok in available.items() if ok],
        "routing": routing,
    }
