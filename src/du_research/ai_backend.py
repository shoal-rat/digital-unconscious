"""Compatibility facade for the modular backend kernel.

New code may import from :mod:`du_research.backends`; this module keeps the
stable public surface used by existing extensions and older configurations.
"""

# ruff: noqa: F401
from du_research.backends import (
    AIBackend,
    AIResponse,
    AnthropicAPIBackend,
    ClaudeCodeBackend,
    CodexCLIBackend,
    DeepSeekAPIBackend,
    GLMAPIBackend,
    KimiAPIBackend,
    MultiProviderBackend,
    OpenAIAPIBackend,
    create_backend,
    resolve_routing,
)
from du_research.backends.base import (
    parse_structured_text as _parse_structured_text,
)
from du_research.backends.base import (
    reasoning_effort as _reasoning_effort,
)
from du_research.backends.base import (
    thinking_budget as _thinking_budget,
)
from du_research.backends.hosted import (
    ANTHROPIC_MODEL_ALIASES,
    DEEPSEEK_MODEL_ALIASES,
    GLM_MODEL_ALIASES,
    KIMI_MODEL_ALIASES,
    OPENAI_MODEL_ALIASES,
)

MODEL_ALIASES = ANTHROPIC_MODEL_ALIASES


def _claude_code_think_keyword(think):
    effort = _reasoning_effort(think)
    return {
        "low": "think hard",
        "medium": "think harder",
        "high": "ultrathink",
        "xhigh": "ultrathink",
        "max": "ultrathink",
    }.get(effort)


__all__ = [
    "AIBackend",
    "AIResponse",
    "AnthropicAPIBackend",
    "ClaudeCodeBackend",
    "CodexCLIBackend",
    "DeepSeekAPIBackend",
    "GLMAPIBackend",
    "KimiAPIBackend",
    "MultiProviderBackend",
    "OpenAIAPIBackend",
    "create_backend",
    "resolve_routing",
]
