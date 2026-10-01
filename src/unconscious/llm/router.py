"""Work-first routing: each role names the kind of thinking it needs, and the
router finds the best *available* model for it, falling back in order.

Roles
  digest    cheap, structured: group the day's subjects into topics and threads
  dream     the creative step: reflection, undercurrent, spark candidates
  critique  an independent skeptic; prefers a different provider than dream
  dive      literature synthesis for one spark
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from unconscious.llm.base import LLMRequest, LLMResult, check, extract_json
from unconscious.llm.providers import AnthropicAPI, ClaudeCodeCLI, CodexCLI, OpenAICompatible

if TYPE_CHECKING:
    from unconscious.config import Settings
    from unconscious.store import Store

log = logging.getLogger(__name__)

ROLES = ("digest", "dream", "critique", "dive")

ROLE_PREFERENCES: dict[str, list[str]] = {
    "digest": ["deepseek", "claude:haiku", "codex", "anthropic:claude-haiku-4-5", "openai", "glm", "kimi", "ollama"],
    "dream": ["claude:opus", "codex", "anthropic:claude-opus-5-5", "openai", "deepseek", "glm", "kimi", "ollama"],
    "critique": ["codex", "claude:sonnet", "anthropic:claude-sonnet-5-5", "openai", "deepseek", "glm", "kimi", "ollama"],
    "dive": ["claude:sonnet", "codex", "anthropic:claude-opus-5-5", "openai", "deepseek", "glm", "kimi", "ollama"],
}
ROLE_EFFORT = {"digest": "low", "dream": "high", "critique": "medium", "dive": "medium"}
ALIASES = {"claude_code": "claude", "zai": "glm", "moonshot": "kimi"}


def parse_spec(spec: str) -> tuple[str, str | None]:
    name, _, model = (spec or "").strip().partition(":")
    name = ALIASES.get(name.strip().lower(), name.strip().lower())
    return name, (model.strip() or None)


def build_providers(settings: Settings) -> dict[str, Any]:
    timeout = settings.models.timeout_seconds
    providers: dict[str, Any] = {
        "claude": ClaudeCodeCLI(timeout=timeout),
        "codex": CodexCLI(timeout=timeout),
        "anthropic": AnthropicAPI(timeout=timeout),
        "openai": OpenAICompatible("openai", "https://api.openai.com/v1", ("OPENAI_API_KEY",), "gpt-5.6-sol", timeout),
        "deepseek": OpenAICompatible("deepseek", "https://api.deepseek.com", ("DEEPSEEK_API_KEY",), "deepseek-v4-flash", timeout),
        "glm": OpenAICompatible("glm", "https://api.z.ai/api/paas/v4", ("ZAI_API_KEY", "GLM_API_KEY"), "glm-5.1", timeout),
        "kimi": OpenAICompatible("kimi", "https://api.moonshot.ai/v1", ("MOONSHOT_API_KEY", "KIMI_API_KEY"), "kimi-k2.6", timeout),
    }
    for name, cfg in settings.providers.items():
        key = str(name).lower()
        if key == "ollama" or cfg.get("kind", "openai") == "openai":
            env = cfg.get("key_env") or ()
            providers[key] = OpenAICompatible(
                name=key,
                base_url=str(cfg.get("base_url") or "http://localhost:11434/v1"),
                key_env=(env,) if isinstance(env, str) else tuple(env),
                default_model=str(cfg.get("model") or ""),
                timeout=timeout,
                send_temperature=bool(cfg.get("temperature", True)),
                local=not env,
            )
    if "ollama" not in providers:
        # Only usable once a model is configured: [providers.ollama] model = "qwen3:14b"
        providers["ollama"] = OpenAICompatible("ollama", "http://localhost:11434/v1", (), "", timeout, True, local=True)
    return providers


@dataclass
class Router:
    settings: Settings
    store: Store | None = None
    providers: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.providers:
            self.providers = build_providers(self.settings)

    def _available(self, name: str) -> bool:
        provider = self.providers.get(name)
        if provider is None:
            return False
        if isinstance(provider, OpenAICompatible) and provider.local and not provider.default_model:
            return False
        try:
            return bool(provider.available())
        except Exception:  # availability probes must never break routing
            return False

    def chain(self, role: str) -> list[tuple[str, str | None]]:
        configured = getattr(self.settings.models, role, "auto") if role in ROLES else "auto"
        specs: list[str] = []
        if configured and configured != "auto":
            specs.append(configured)
        if not specs or self.settings.models.fallback:
            specs += ROLE_PREFERENCES.get(role, ROLE_PREFERENCES["dream"])
        seen: set[str] = set()
        out: list[tuple[str, str | None]] = []
        for spec in specs:
            name, model = parse_spec(spec)
            if name in seen or not self._available(name):
                continue
            seen.add(name)
            out.append((name, model))
        if role == "critique" and len(out) > 1:
            # An independent challenge is worth more than the strongest model:
            # prefer a provider other than the one that dreamt.
            dreamer = next(iter(self.chain("dream")), (None, None))[0]
            out.sort(key=lambda item: item[0] == dreamer)
        return out

    def describe(self) -> dict[str, Any]:
        available = {name: self._available(name) for name in self.providers}
        routes = {}
        for role in ROLES:
            chain = self.chain(role)
            routes[role] = {
                "configured": getattr(self.settings.models, role),
                "chain": [f"{n}:{m}" if m else n for n, m in chain],
            }
        return {"available": available, "routes": routes}

    def call(self, request: LLMRequest) -> LLMResult:
        chain = self.chain(request.role)
        if not chain:
            return LLMResult(
                False,
                error="No model is available. Sign in to Claude Code (`claude`) or Codex (`codex`), "
                "or set an API key such as DEEPSEEK_API_KEY. `dun doctor` shows what is missing.",
            )
        request.effort = request.effort or ROLE_EFFORT.get(request.role)
        errors: list[str] = []
        for name, model in chain:
            provider = self.providers[name]
            result = self._attempt(provider, request, model)
            if result.ok and request.schema:
                problems = check(result.data, request.schema) if result.data is not None else ["no JSON object"]
                if problems:
                    log.info("%s returned invalid JSON (%s); asking once more", name, problems[:3])
                    repair = LLMRequest(
                        role=request.role,
                        system=request.system,
                        prompt=request.prompt
                        + "\n\nYour previous reply did not match the required JSON schema: "
                        + "; ".join(problems[:6])
                        + ". Reply again with only the corrected JSON object.",
                        schema=request.schema,
                        max_tokens=request.max_tokens,
                        effort=request.effort,
                        payload=request.payload,
                    )
                    result = self._attempt(provider, repair, model)
                    if result.ok:
                        data = result.data if result.data is not None else extract_json(result.text)
                        problems = check(data, request.schema) if data is not None else ["no JSON object"]
                        result.data = data
                    if not result.ok or problems:
                        result.ok = False
                        result.error = result.error or "invalid JSON: " + "; ".join(problems[:4])
            if result.ok:
                return result
            errors.append(f"{result.label}: {result.error}")
            if not self.settings.models.fallback:
                break
        return LLMResult(False, error=" | ".join(errors)[-1200:])

    def _attempt(self, provider: Any, request: LLMRequest, model: str | None) -> LLMResult:
        try:
            result = provider.complete(request, model)
        except Exception as exc:  # a provider bug must not take down the dream
            log.exception("provider %s crashed", getattr(provider, "name", "?"))
            result = LLMResult(False, provider=getattr(provider, "name", "?"), model=model or "", error=f"{type(exc).__name__}: {exc}")
        if self.store is not None:
            try:
                self.store.log_llm(
                    role=request.role,
                    provider=result.provider or getattr(provider, "name", "?"),
                    model=result.model or (model or ""),
                    ok=int(result.ok),
                    ms=result.ms,
                    tokens_in=result.tokens_in,
                    tokens_out=result.tokens_out,
                    cost=result.cost,
                    error=result.error[:500],
                )
            except Exception:
                log.debug("could not log llm call", exc_info=True)
        return result
