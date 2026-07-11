"""Capability routing and transparent fallback across local and hosted models."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any

from du_research.backends.base import AIBackend, AIResponse
from du_research.backends.hosted import (
    ANTHROPIC_MODEL_ALIASES,
    DEEPSEEK_MODEL_ALIASES,
    GLM_MODEL_ALIASES,
    KIMI_MODEL_ALIASES,
    OPENAI_MODEL_ALIASES,
    AnthropicAPIBackend,
    DeepSeekAPIBackend,
    GLMAPIBackend,
    KimiAPIBackend,
    OpenAIAPIBackend,
)
from du_research.backends.local import ClaudeCodeBackend, CodexCLIBackend

logger = logging.getLogger(__name__)

PROVIDER_PREFIXES = {
    "anthropic": "anthropic",
    "claude": "anthropic",
    "claude_code": "claude_code",
    "openai": "openai",
    "codex": "codex",
    "deepseek": "deepseek",
    "ds": "deepseek",
    "glm": "glm",
    "zai": "glm",
    "kimi": "kimi",
    "moonshot": "kimi",
}
MODEL_ALIASES = {
    "anthropic": ANTHROPIC_MODEL_ALIASES,
    "claude_code": {"opus": "opus", "sonnet": "sonnet", "haiku": "haiku", "default": "default"},
    "openai": OPENAI_MODEL_ALIASES,
    "codex": {"default": "default", "gpt": "default", "codex": "default"},
    "deepseek": DEEPSEEK_MODEL_ALIASES,
    "glm": GLM_MODEL_ALIASES,
    "kimi": KIMI_MODEL_ALIASES,
}
DEFAULT_FALLBACK_ORDER = ["deepseek", "glm", "codex", "claude_code", "openai", "anthropic", "kimi"]


def split_provider_model(model: str | None) -> tuple[str | None, str | None]:
    if not model or ":" not in model:
        return None, model
    prefix, bare = model.split(":", 1)
    provider = PROVIDER_PREFIXES.get(prefix.strip().lower())
    return (provider, bare.strip()) if provider else (None, model)


def _has_env(*names: str) -> bool:
    return any(os.environ.get(name) for name in names)


def provider_availability(overrides: dict[str, Any] | None = None) -> dict[str, bool]:
    values = overrides or {}
    return {
        "deepseek": bool(values.get("deepseek_api_key") or _has_env("DEEPSEEK_API_KEY")),
        "glm": bool(values.get("glm_api_key") or _has_env("ZAI_API_KEY", "GLM_API_KEY")),
        "codex": bool(shutil.which("codex")),
        "claude_code": bool(shutil.which("claude")),
        "openai": bool(values.get("openai_api_key") or _has_env("OPENAI_API_KEY")),
        "anthropic": bool(values.get("api_key") or _has_env("ANTHROPIC_API_KEY")),
        "kimi": bool(values.get("kimi_api_key") or _has_env("MOONSHOT_API_KEY", "KIMI_API_KEY")),
    }


@dataclass
class MultiProviderBackend:
    api_key: str | None = None
    openai_api_key: str | None = None
    kimi_api_key: str | None = None
    deepseek_api_key: str | None = None
    glm_api_key: str | None = None
    default_model: str = "claude-sonnet-4-6"
    openai_default_model: str = "gpt-5.6-sol"
    kimi_default_model: str = "kimi-k2.6"
    deepseek_default_model: str = "deepseek-v4-flash"
    glm_default_model: str = "glm-5.1"
    timeout_seconds: int = 300
    enable_fallback: bool = True
    fallback_order: list[str] = field(default_factory=lambda: DEFAULT_FALLBACK_ORDER.copy())
    _providers: dict[str, AIBackend] = field(default_factory=dict, repr=False)

    def _overrides(self) -> dict[str, Any]:
        return {
            "api_key": self.api_key,
            "openai_api_key": self.openai_api_key,
            "kimi_api_key": self.kimi_api_key,
            "deepseek_api_key": self.deepseek_api_key,
            "glm_api_key": self.glm_api_key,
        }

    def _primary(self, model: str | None) -> tuple[str, str | None]:
        provider, bare = split_provider_model(model)
        if provider:
            return provider, bare
        # Explicitly injected providers form a closed test/extension graph.
        # This keeps dependency injection deterministic and never leaks into an
        # installed local CLI during unit tests.
        if self._providers:
            return next(iter(self._providers)), model
        available = provider_availability(self._overrides())
        for candidate in self.fallback_order:
            if available.get(candidate):
                return candidate, model
        return "codex", model

    def _chain(self, primary: str) -> list[str]:
        if not self.enable_fallback:
            return [primary]
        if self._providers:
            return [primary] + [name for name in self._providers if name != primary]
        available = provider_availability(self._overrides())
        return [primary] + [
            provider for provider in self.fallback_order if provider != primary and available.get(provider)
        ]

    def _get_provider(self, provider: str) -> AIBackend:
        if provider in self._providers:
            return self._providers[provider]
        if provider == "codex":
            backend: AIBackend = CodexCLIBackend(timeout_seconds=self.timeout_seconds)
        elif provider == "claude_code":
            backend = ClaudeCodeBackend(timeout_seconds=self.timeout_seconds)
        elif provider == "deepseek":
            backend = DeepSeekAPIBackend(
                api_key=self.deepseek_api_key, default_model=self.deepseek_default_model
            )
        elif provider == "glm":
            backend = GLMAPIBackend(api_key=self.glm_api_key, default_model=self.glm_default_model)
        elif provider == "openai":
            backend = OpenAIAPIBackend(api_key=self.openai_api_key, default_model=self.openai_default_model)
        elif provider == "anthropic":
            backend = AnthropicAPIBackend(api_key=self.api_key, default_model=self.default_model)
        else:
            backend = KimiAPIBackend(api_key=self.kimi_api_key, default_model=self.kimi_default_model)
        self._providers[provider] = backend
        return backend

    def call(self, prompt: str, **kwargs: Any) -> AIResponse:
        requested_model = kwargs.get("model")
        primary, bare_model = self._primary(requested_model)
        chain = self._chain(primary)
        last = AIResponse(text="", raw={"error": "no provider available", "provider": primary})
        for index, provider in enumerate(chain):
            try:
                backend = self._get_provider(provider)
            except Exception as exc:
                last = AIResponse(text="", raw={"error": str(exc), "provider": provider})
                continue
            call_kwargs = dict(kwargs)
            call_kwargs["model"] = bare_model if index == 0 else None
            response = backend.call(prompt, **call_kwargs)
            response.raw.setdefault("router_provider", provider)
            response.raw["router_chain"] = chain
            if index:
                response.raw["router_fallback_from"] = primary
            if response.ok:
                return response
            last = response
            logger.warning("Provider %s failed; trying the next available backend", provider)
        return last


def create_backend(mode: str = "auto", **kwargs: Any) -> AIBackend:
    mode = (mode or "auto").strip().lower()
    aliases = {"api": "anthropic", "claude": "anthropic", "moonshot": "kimi", "zai": "glm", "ds": "deepseek"}
    mode = aliases.get(mode, mode)
    if mode in {"auto", "multi", "router"}:
        return MultiProviderBackend(
            api_key=kwargs.get("api_key"),
            openai_api_key=kwargs.get("openai_api_key"),
            kimi_api_key=kwargs.get("kimi_api_key") or kwargs.get("moonshot_api_key"),
            deepseek_api_key=kwargs.get("deepseek_api_key"),
            glm_api_key=kwargs.get("glm_api_key"),
            default_model=kwargs.get("default_model") or "claude-sonnet-4-6",
            openai_default_model=kwargs.get("openai_default_model") or "gpt-5.6-sol",
            kimi_default_model=kwargs.get("kimi_default_model") or "kimi-k2.6",
            deepseek_default_model=kwargs.get("deepseek_default_model") or "deepseek-v4-flash",
            glm_default_model=kwargs.get("glm_default_model") or "glm-5.1",
            timeout_seconds=kwargs.get("timeout_seconds", 300),
            enable_fallback=kwargs.get("enable_fallback", True),
            fallback_order=list(kwargs.get("fallback_order") or DEFAULT_FALLBACK_ORDER),
        )
    if mode == "codex":
        return CodexCLIBackend(
            timeout_seconds=kwargs.get("timeout_seconds", 300), model_override=kwargs.get("model_override")
        )
    if mode == "claude_code":
        return ClaudeCodeBackend(
            timeout_seconds=kwargs.get("timeout_seconds", 300), model_override=kwargs.get("model_override")
        )
    classes = {
        "deepseek": DeepSeekAPIBackend,
        "glm": GLMAPIBackend,
        "openai": OpenAIAPIBackend,
        "anthropic": AnthropicAPIBackend,
        "kimi": KimiAPIBackend,
    }
    cls = classes.get(mode)
    if cls is None:
        raise ValueError(f"Unknown AI backend mode: {mode}")
    key_names = {
        "anthropic": "api_key",
        "openai": "openai_api_key",
        "kimi": "kimi_api_key",
        "deepseek": "deepseek_api_key",
        "glm": "glm_api_key",
    }
    default_names = {
        "openai": "openai_default_model",
        "kimi": "kimi_default_model",
        "deepseek": "deepseek_default_model",
        "glm": "glm_default_model",
    }
    default = kwargs.get(default_names.get(mode, "default_model")) or kwargs.get("default_model")
    values: dict[str, Any] = {"api_key": kwargs.get(key_names[mode]) or kwargs.get("api_key")}
    if default:
        values["default_model"] = default
    return cls(**values)


def resolve_routing(config: Any) -> dict[str, Any]:
    ai = config.ai
    overrides = {
        "api_key": getattr(ai, "api_key", ""),
        "openai_api_key": getattr(ai, "openai_api_key", ""),
        "kimi_api_key": getattr(ai, "kimi_api_key", ""),
        "deepseek_api_key": getattr(ai, "deepseek_api_key", ""),
        "glm_api_key": getattr(ai, "glm_api_key", ""),
    }
    available = provider_availability(overrides)
    order = list(getattr(ai, "fallback_order", DEFAULT_FALLBACK_ORDER))

    def resolve(value: str) -> tuple[str, str]:
        provider, bare = split_provider_model(value)
        if provider is None:
            provider = next((name for name in order if available.get(name)), "codex")
        model = MODEL_ALIASES.get(provider, {}).get(bare or "", bare or "default")
        return provider, model

    fields = [
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
    for agent, configured in fields:
        provider, model = resolve(configured)
        ready = available.get(provider, False)
        fallback_provider = (
            None
            if ready
            else next((name for name in order if name != provider and available.get(name)), None)
        )
        fallback_model = None
        if fallback_provider:
            fallback_model = {
                "codex": "default",
                "claude_code": "default",
                "deepseek": getattr(ai, "deepseek_default_model", "deepseek-v4-flash"),
                "glm": getattr(ai, "glm_default_model", "glm-5.1"),
                "openai": getattr(ai, "openai_default_model", "gpt-5.6-sol"),
                "anthropic": getattr(ai, "default_model", "claude-sonnet-4-6"),
                "kimi": getattr(ai, "kimi_default_model", "kimi-k2.6"),
            }[fallback_provider]
        routing.append(
            {
                "agent": agent,
                "configured": configured,
                "provider": provider,
                "model": model,
                "available": ready,
                "fallback_provider": fallback_provider,
                "fallback_model": fallback_model,
            }
        )
    return {
        "mode": ai.mode,
        "fallback": ai.fallback,
        "fallback_order": order,
        "available_providers": [name for name, ready in available.items() if ready],
        "routing": routing,
    }


def diagnose_providers() -> list[dict[str, Any]]:
    """Return a fast, secret-free readiness report for `du doctor`."""
    availability = provider_availability()
    rows: list[dict[str, Any]] = []
    for provider in DEFAULT_FALLBACK_ORDER:
        kind = "subscription CLI" if provider in {"codex", "claude_code"} else "optional API"
        ready = availability[provider]
        detail = "configured" if ready else "not configured"
        executable = "codex" if provider == "codex" else "claude" if provider == "claude_code" else None
        if executable and shutil.which(executable):
            try:
                completed = subprocess.run(
                    [executable, "--version"], capture_output=True, text=True, timeout=5
                )
                version = (completed.stdout or completed.stderr).strip()
                auth_command = (
                    ["codex", "login", "status"] if provider == "codex" else ["claude", "auth", "status"]
                )
                auth = subprocess.run(auth_command, capture_output=True, text=True, timeout=8)
                if provider == "claude_code" and auth.returncode == 0:
                    try:
                        signed_in = bool(json.loads(auth.stdout).get("loggedIn"))
                    except (json.JSONDecodeError, AttributeError):
                        signed_in = False
                else:
                    signed_in = auth.returncode == 0
                ready = signed_in
                detail = f"{version} · {'signed in' if signed_in else 'sign-in required'}"
            except Exception:
                detail = "installed"
        rows.append({"provider": provider, "kind": kind, "ready": ready, "detail": detail})
    return rows
