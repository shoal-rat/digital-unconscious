"""Circuit breaker and retry logic for resilient AI backend calls.

Implements a three-state circuit breaker (CLOSED → OPEN → HALF_OPEN)
with exponential back-off retries and adaptive rate-limit handling.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

from du_research.ai_backend import AIBackend, AIResponse

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "closed"          # normal operation
    OPEN = "open"              # tripped — all calls short-circuited
    HALF_OPEN = "half_open"    # testing recovery with one probe call


@dataclass
class CircuitBreaker:
    """Wraps an :class:`AIBackend` with retry + circuit-breaker logic.

    Parameters
    ----------
    backend : AIBackend
        The underlying backend to call.
    max_retries : int
        Number of retries (with exponential back-off) before a call is
        declared failed.
    initial_wait : float
        Seconds to wait before the first retry.
    failure_threshold : int
        Consecutive failures that trip the breaker open.
    recovery_timeout : float
        Seconds the breaker stays OPEN before transitioning to HALF_OPEN.
    max_wait : float
        Upper bound on exponential back-off wait.
    """

    backend: AIBackend
    max_retries: int = 3
    initial_wait: float = 2.0
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    max_wait: float = 30.0

    # Internal state ----------------------------------------------------------
    _state: CircuitState = field(default=CircuitState.CLOSED, repr=False)
    _consecutive_failures: int = field(default=0, repr=False)
    _last_failure_time: float = field(default=0.0, repr=False)
    _total_calls: int = field(default=0, repr=False)
    _total_failures: int = field(default=0, repr=False)
    _total_retries: int = field(default=0, repr=False)

    # Public API --------------------------------------------------------------

    def call(self, prompt: str, **kwargs: Any) -> AIResponse:
        """Call the backend with retry and circuit-breaker protection."""
        self._total_calls += 1

        # Fast-fail if breaker is OPEN
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                logger.info("Circuit breaker transitioning OPEN → HALF_OPEN")
                self._state = CircuitState.HALF_OPEN
            else:
                remaining = self.recovery_timeout - (time.monotonic() - self._last_failure_time)
                logger.warning(
                    "Circuit breaker OPEN — rejecting call (%.0fs until recovery)",
                    remaining,
                )
                return AIResponse(
                    text="",
                    raw={"error": "circuit_breaker_open", "retry_after_seconds": remaining},
                )

        wait = self.initial_wait
        last_error: str = ""

        for attempt in range(1, self.max_retries + 1):
            response = self.backend.call(prompt, **kwargs)

            # Check for success
            if response.ok:
                self._on_success()
                return response

            # Check for explicit rate-limit signal
            error_text = response.raw.get("error", "")
            if "rate_limit" in str(error_text).lower() or "429" in str(error_text):
                self._total_retries += 1
                logger.warning(
                    "Rate limited (attempt %d/%d), backing off %.1fs",
                    attempt, self.max_retries, wait,
                )
                time.sleep(wait)
                wait = min(wait * 2, self.max_wait)
                last_error = str(error_text)
                continue

            # Non-retryable error (auth, invalid request, etc.)
            if any(
                code in str(error_text)
                for code in ("401", "403", "invalid", "authentication")
            ):
                self._on_failure()
                return response

            # Generic failure — retry
            self._total_retries += 1
            last_error = str(error_text)
            if attempt < self.max_retries:
                logger.info(
                    "Retrying (attempt %d/%d) after %.1fs: %s",
                    attempt, self.max_retries, wait, last_error[:200],
                )
                time.sleep(wait)
                wait = min(wait * 2, self.max_wait)

        # All retries exhausted
        self._on_failure()
        return AIResponse(text="", raw={"error": f"max_retries_exhausted: {last_error}"})

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "total_retries": self._total_retries,
            "consecutive_failures": self._consecutive_failures,
        }

    # Internal ----------------------------------------------------------------

    def _on_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker recovered → CLOSED")
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0

    def _on_failure(self) -> None:
        self._consecutive_failures += 1
        self._total_failures += 1
        self._last_failure_time = time.monotonic()

        if self._consecutive_failures >= self.failure_threshold:
            logger.error(
                "Circuit breaker tripped OPEN after %d consecutive failures",
                self._consecutive_failures,
            )
            self._state = CircuitState.OPEN


# ---------------------------------------------------------------------------
# Convenience: resilient backend factory
# ---------------------------------------------------------------------------


def resilient(backend: AIBackend, **kwargs: Any) -> CircuitBreaker:
    """Wrap *backend* in a :class:`CircuitBreaker`."""
    return CircuitBreaker(backend=backend, **kwargs)
