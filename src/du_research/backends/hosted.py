"""Optional API backends. Local CLI runners remain the zero-key default."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from du_research.backends.base import (
    MODE_TEMPERATURE,
    AIResponse,
    anthropic_user_content,
    openai_user_content,
    parse_structured_text,
    reasoning_effort,
    thinking_budget,
    usage_value,
)

logger = logging.getLogger(__name__)

ANTHROPIC_MODEL_ALIASES = {
    "opus": "claude-opus-4-8",
    "sonnet": "claude-sonnet-5",
    "haiku": "claude-haiku-4-5-20251001",
}
OPENAI_MODEL_ALIASES = {
    "gpt": "gpt-5.6-sol",
    "codex": "gpt-5.6-sol",
    "opus": "gpt-5.6-sol",
    "sonnet": "gpt-5.6-terra",
    "haiku": "gpt-5.6-luna",
}
DEEPSEEK_MODEL_ALIASES = {
    "deepseek": "deepseek-v4-flash",
    "flash": "deepseek-v4-flash",
    "pro": "deepseek-v4-pro",
    "haiku": "deepseek-v4-flash",
    "sonnet": "deepseek-v4-flash",
    "opus": "deepseek-v4-pro",
}
GLM_MODEL_ALIASES = {
    "glm": "glm-5.1",
    "haiku": "glm-4.7-flash",
    "sonnet": "glm-5",
    "opus": "glm-5.1",
}
KIMI_MODEL_ALIASES = {
    "kimi": "kimi-k2.6",
    "k2": "kimi-k2.6",
    "haiku": "kimi-k2.6",
    "sonnet": "kimi-k2.6",
    "opus": "kimi-k2.6",
}


def _resolve(model: str | None, default: str, aliases: dict[str, str]) -> str:
    return aliases.get(model or "", model or default)


@dataclass
class AnthropicAPIBackend:
    api_key: str | None = None
    default_model: str = "claude-sonnet-5"
    _client: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._client is not None:
            return
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("Install the 'api' extra for Anthropic API mode") from exc
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is not configured")
        self._client = anthropic.Anthropic(api_key=key)

    def call(self, prompt: str, **kwargs: Any) -> AIResponse:
        model = _resolve(kwargs.get("model"), self.default_model, ANTHROPIC_MODEL_ALIASES)
        max_tokens = int(kwargs.get("max_tokens", 2048))
        schema = kwargs.get("json_schema")
        system_parts = [kwargs.get("system")]
        if schema:
            system_parts.append("Return only JSON matching this schema: " + json.dumps(schema))
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": anthropic_user_content(prompt, kwargs.get("images"))}],
        }
        system = "\n\n".join(part for part in system_parts if part)
        if system:
            request["system"] = system
        budget = thinking_budget(kwargs.get("think"))
        if budget:
            request["thinking"] = {"type": "enabled", "budget_tokens": budget}
            request["max_tokens"] = max(max_tokens, budget + max_tokens)
        else:
            request["temperature"] = MODE_TEMPERATURE.get(kwargs.get("mode", "balanced"), 0.7)
        if kwargs.get("web_search"):
            request["tools"] = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
        try:
            response = self._client.messages.create(**request)
        except Exception as exc:
            logger.error("Anthropic API error: %s", exc)
            return AIResponse(text="", raw={"error": str(exc), "provider": "anthropic"})
        text = "\n".join(block.text for block in response.content if hasattr(block, "text"))
        usage = getattr(response, "usage", None)
        return AIResponse(
            text=text,
            model=model,
            input_tokens=usage_value(usage, "input_tokens"),
            output_tokens=usage_value(usage, "output_tokens"),
            raw={"provider": "anthropic", "stop_reason": getattr(response, "stop_reason", None)},
            structured=parse_structured_text(text, schema),
        )


@dataclass
class OpenAICompatibleChatBackend:
    api_key: str | None = None
    default_model: str = ""
    base_url: str | None = None
    env_key: str = "OPENAI_API_KEY"
    alternate_env_keys: tuple[str, ...] = ()
    provider_name: str = "openai-compatible"
    model_aliases: dict[str, str] = field(default_factory=dict)
    omit_temperature: bool = False
    supports_thinking: bool = False
    _client: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install the 'providers' extra for hosted provider mode") from exc
        key = self.api_key or os.environ.get(self.env_key)
        for name in self.alternate_env_keys:
            key = key or os.environ.get(name)
        if not key:
            raise ValueError(f"{self.env_key} is not configured for {self.provider_name}")
        options: dict[str, Any] = {"api_key": key}
        if self.base_url:
            options["base_url"] = self.base_url
        self._client = OpenAI(**options)

    def call(self, prompt: str, **kwargs: Any) -> AIResponse:
        model = _resolve(kwargs.get("model"), self.default_model, self.model_aliases)
        schema = kwargs.get("json_schema")
        messages: list[dict[str, Any]] = []
        if kwargs.get("system"):
            messages.append({"role": "system", "content": kwargs["system"]})
        if schema:
            messages.append(
                {"role": "system", "content": "Return only valid JSON matching: " + json.dumps(schema)}
            )
        messages.append({"role": "user", "content": openai_user_content(prompt, kwargs.get("images"))})
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": int(kwargs.get("max_tokens", 2048)),
        }
        effort = reasoning_effort(kwargs.get("think"))
        if not self.omit_temperature and not effort:
            request["temperature"] = MODE_TEMPERATURE.get(kwargs.get("mode", "balanced"), 0.7)
        if effort and self.provider_name == "openai":
            request["reasoning_effort"] = effort
        if self.supports_thinking:
            enabled = bool(effort) or (not schema and kwargs.get("mode") != "deterministic")
            request["extra_body"] = {"thinking": {"type": "enabled" if enabled else "disabled"}}
            if effort:
                request["reasoning_effort"] = "max" if effort in {"xhigh", "max"} else "high"
        if schema:
            request["response_format"] = {"type": "json_object"}
        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:
            logger.error("%s API error: %s", self.provider_name, exc)
            return AIResponse(text="", raw={"error": str(exc), "provider": self.provider_name})
        choice = response.choices[0] if getattr(response, "choices", None) else None
        message = getattr(choice, "message", None)
        text = getattr(message, "content", "") or ""
        usage = getattr(response, "usage", None)
        return AIResponse(
            text=text,
            model=model,
            input_tokens=usage_value(usage, "prompt_tokens"),
            output_tokens=usage_value(usage, "completion_tokens"),
            raw={"provider": self.provider_name, "finish_reason": getattr(choice, "finish_reason", None)},
            structured=parse_structured_text(text, schema),
        )


@dataclass
class OpenAIAPIBackend(OpenAICompatibleChatBackend):
    """OpenAI Responses API with a chat fallback for older SDK doubles."""

    provider_name: str = "openai"
    env_key: str = "OPENAI_API_KEY"
    default_model: str = "gpt-5.6-sol"
    model_aliases: dict[str, str] = field(default_factory=lambda: OPENAI_MODEL_ALIASES.copy())
    omit_temperature: bool = True

    def call(self, prompt: str, **kwargs: Any) -> AIResponse:
        if not hasattr(self._client, "responses"):
            return super().call(prompt, **kwargs)
        model = _resolve(kwargs.get("model"), self.default_model, self.model_aliases)
        schema = kwargs.get("json_schema")
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for part in openai_user_content("", kwargs.get("images")) if kwargs.get("images") else []:
            if part.get("type") == "image_url":
                content.append({"type": "input_image", "image_url": part["image_url"]["url"]})
        request: dict[str, Any] = {
            "model": model,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": int(kwargs.get("max_tokens", 2048)),
        }
        if kwargs.get("system"):
            request["instructions"] = kwargs["system"]
        effort = reasoning_effort(kwargs.get("think"))
        if effort:
            request["reasoning"] = {"effort": effort}
        if schema:
            request["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "digital_unconscious",
                    "schema": schema,
                    "strict": False,
                }
            }
        if kwargs.get("web_search"):
            request["tools"] = [{"type": "web_search"}]
        try:
            response = self._client.responses.create(**request)
        except Exception as exc:
            logger.error("OpenAI Responses API error: %s", exc)
            return AIResponse(text="", raw={"error": str(exc), "provider": "openai"})
        text = getattr(response, "output_text", "") or ""
        usage = getattr(response, "usage", None)
        return AIResponse(
            text=text,
            model=model,
            input_tokens=usage_value(usage, "input_tokens"),
            output_tokens=usage_value(usage, "output_tokens"),
            raw={"provider": "openai", "response_id": getattr(response, "id", None)},
            structured=parse_structured_text(text, schema),
        )


@dataclass
class DeepSeekAPIBackend(OpenAICompatibleChatBackend):
    provider_name: str = "deepseek"
    env_key: str = "DEEPSEEK_API_KEY"
    base_url: str | None = "https://api.deepseek.com"
    default_model: str = "deepseek-v4-flash"
    model_aliases: dict[str, str] = field(default_factory=lambda: DEEPSEEK_MODEL_ALIASES.copy())
    omit_temperature: bool = True
    supports_thinking: bool = True


@dataclass
class GLMAPIBackend(OpenAICompatibleChatBackend):
    provider_name: str = "glm"
    env_key: str = "ZAI_API_KEY"
    alternate_env_keys: tuple[str, ...] = ("GLM_API_KEY",)
    base_url: str | None = "https://api.z.ai/api/paas/v4"
    default_model: str = "glm-5.1"
    model_aliases: dict[str, str] = field(default_factory=lambda: GLM_MODEL_ALIASES.copy())
    omit_temperature: bool = True


@dataclass
class KimiAPIBackend(OpenAICompatibleChatBackend):
    provider_name: str = "kimi"
    env_key: str = "MOONSHOT_API_KEY"
    alternate_env_keys: tuple[str, ...] = ("KIMI_API_KEY",)
    base_url: str | None = "https://api.moonshot.ai/v1"
    default_model: str = "kimi-k2.6"
    model_aliases: dict[str, str] = field(default_factory=lambda: KIMI_MODEL_ALIASES.copy())
    omit_temperature: bool = True
    supports_thinking: bool = True
