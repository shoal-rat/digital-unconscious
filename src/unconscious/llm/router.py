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
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from unconscious.llm.base import LLMRequest, LLMResult, check, extract_json
from unconscious.llm.crew import REGION, Crew, mark
from unconscious.llm.providers import AnthropicAPI, ClaudeCodeCLI, CodexCLI, OpenAICompatible
from unconscious.llm.region import GUARDED, UNKNOWN, RegionCheck, region_check

if TYPE_CHECKING:
    from unconscious.config import Settings
    from unconscious.store import Store

log = logging.getLogger(__name__)

ROLES = ("digest", "dream", "critique", "dive")

# Claude work goes to Sonnet 5.5 and Opus 5.5 by their full IDs, so an alias
# moving to a different generation never changes behaviour silently.
SONNET = "claude-sonnet-5-5"
OPUS = "claude-opus-5-5"
ROLE_PREFERENCES: dict[str, list[str]] = {
    "digest": [f"claude:{SONNET}", "codex", f"anthropic:{SONNET}", "deepseek", "openai", "glm", "kimi", "ollama"],
    "dream": [f"claude:{OPUS}", "codex", f"anthropic:{OPUS}", "openai", "deepseek", "glm", "kimi", "ollama"],
    "critique": ["codex", f"claude:{SONNET}", f"anthropic:{SONNET}", "openai", "deepseek", "glm", "kimi", "ollama"],
    "dive": [f"claude:{SONNET}", "codex", f"anthropic:{SONNET}", "openai", "deepseek", "glm", "kimi", "ollama"],
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
    region: RegionCheck = field(default_factory=lambda: region_check)
    crew: Crew | None = None
    _seen: dict[str, tuple[float, bool]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.providers:
            self.providers = build_providers(self.settings)
        if self.crew is None:
            self.crew = Crew(self.store)

    def _available(self, name: str) -> bool:
        # Looking for crew means searching PATH and probing servers; the answer
        # changes rarely, so it is remembered for a couple of minutes.
        cached = self._seen.get(name)
        if cached and time.monotonic() - cached[0] < 120:
            return cached[1]
        provider = self.providers.get(name)
        if provider is None:
            ok = False
        elif isinstance(provider, OpenAICompatible) and provider.local and not provider.default_model:
            ok = False
        else:
            try:
                ok = bool(provider.available())
            except Exception:  # availability probes must never break routing
                ok = False
        self._seen[name] = (time.monotonic(), ok)
        return ok

    def ashore(self, name: str, max_age: float | None = None) -> str | None:
        """The country holding this crew member ashore, or None if it may sail.
        May look the connection up: call it from background threads only."""
        guard = self.settings.models
        if not guard.region_guard or name not in GUARDED:
            return None
        self.region.held = frozenset(guard.hold_regions)
        country = self.region.current(max_age)
        return country if country == UNKNOWN or country in guard.hold_regions else None

    def hold(self, name: str, max_age: float | None = None) -> tuple[str, str] | None:
        """Why this crew member cannot sail right now, as (kind, detail), or None.
        Background threads only: it may look up the connection or ask a CLI if it is signed in."""
        trouble = self.crew.blocked(name, self.providers.get(name))
        if trouble is not None:
            return trouble.kind, trouble.detail
        country = self.ashore(name, max_age)
        return (REGION, country) if country is not None else None

    def ready(self, role: str) -> bool:
        """Is anyone aboard who may sail for this role right now? (background threads)"""
        return any(self.hold(name) is None for name, _ in self.chain(role))

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
        """For the shore and the harbour: never touches the network, so the region is the last one seen."""
        available = {name: self._available(name) for name in self.providers}
        routes = {}
        for role in ROLES:
            chain = self.chain(role)
            routes[role] = {
                "configured": getattr(self.settings.models, role),
                "chain": [f"{n}:{m}" if m else n for n, m in chain],
            }
        fix = self.region.last()
        guard = self.settings.models
        held = bool(fix) and guard.region_guard and (fix.country == UNKNOWN or fix.country in guard.hold_regions)
        region = {
            "guard": guard.region_guard,
            "country": fix.country if fix else None,
            "checked": fix.at if fix else None,
            "ashore": sorted(n for n in GUARDED if available.get(n)) if held else [],
        }
        return {"available": available, "routes": routes, "region": region, "troubles": self.crew.snapshot()}

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
        held: str | None = None  # the first crew mark, if no one could sail
        for name, model in chain:
            reason = self.hold(name, max_age=0)  # the connection is looked up right before each errand
            if reason is not None:
                held = held or mark(reason[0], name, reason[1])
                log.info("%s cannot sail (%s)", name, reason[0])
                continue
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
                        workdir=request.workdir,  # only reshaping the answer: no new research
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
                self.crew.sailed(name)
                return result
            kind = self.crew.failed(name, result.error)
            if kind:  # signed out, limited, offline or refused: not this errand's fault
                held = held or mark(kind, name, result.error[-200:].replace(":", " "))
            else:
                errors.append(f"{result.label}: {result.error}")
            if not self.settings.models.fallback:
                break
        if errors:
            return LLMResult(False, error=" | ".join(errors)[-1200:])
        return LLMResult(False, error=held or "no crew member could sail")

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
