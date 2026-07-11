"""Small, composable model backends used by Digital Unconscious."""

from du_research.backends.base import AIBackend, AIResponse
from du_research.backends.hosted import (
    AnthropicAPIBackend,
    DeepSeekAPIBackend,
    GLMAPIBackend,
    KimiAPIBackend,
    OpenAIAPIBackend,
)
from du_research.backends.local import ClaudeCodeBackend, CodexCLIBackend
from du_research.backends.router import MultiProviderBackend, create_backend, resolve_routing

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
