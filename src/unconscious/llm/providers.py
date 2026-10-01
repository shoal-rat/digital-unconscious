"""Concrete providers.

Local subscription CLIs (Claude Code, Codex) are first-class: they need no API
key. Every CLI call runs in a fresh temporary directory with tools disabled, so
a summarisation request never inherits access to the person's files.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from unconscious.llm.base import LLMRequest, LLMResult, extract_json, full_prompt, schema_hint


def _elapsed(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def _tail(text: str, limit: int = 600) -> str:
    text = (text or "").strip()
    return text[-limit:]


@dataclass
class ClaudeCodeCLI:
    """`claude -p` using the person's Claude subscription login."""

    name: str = "claude"
    timeout: int = 300
    default_model: str = "claude-sonnet-5-5"

    def available(self) -> bool:
        return shutil.which("claude") is not None

    def complete(self, request: LLMRequest, model: str | None) -> LLMResult:
        model = model or self.default_model
        cmd = [
            "claude", "-p",
            "--output-format", "json",
            "--model", model,
            "--system-prompt", request.system,
            "--tools", "",
            "--safe-mode",
            "--strict-mcp-config",
            "--no-session-persistence",
        ]
        if request.schema:
            cmd += ["--json-schema", json.dumps(request.schema, ensure_ascii=False)]
        if request.effort:
            cmd += ["--effort", request.effort]
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="dun-claude-") as tmp:
            try:
                proc = subprocess.run(
                    cmd, input=request.prompt, cwd=tmp, capture_output=True, text=True, timeout=self.timeout
                )
            except subprocess.TimeoutExpired:
                return LLMResult(False, provider=self.name, model=model, ms=_elapsed(start), error="timed out")
            except OSError as exc:
                return LLMResult(False, provider=self.name, model=model, ms=_elapsed(start), error=str(exc))
        ms = _elapsed(start)
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            if proc.returncode != 0:
                return LLMResult(False, provider=self.name, model=model, ms=ms,
                                 error=_tail(proc.stderr or proc.stdout) or f"exit {proc.returncode}")
            text = proc.stdout.strip()
            return LLMResult(bool(text), text, extract_json(text), self.name, model, ms)
        if payload.get("is_error") or proc.returncode != 0:
            return LLMResult(False, provider=self.name, model=model, ms=ms,
                             error=_tail(str(payload.get("result") or proc.stderr)) or "claude reported an error")
        text = str(payload.get("result") or "")
        data = payload.get("structured_output")
        if not isinstance(data, dict):
            data = extract_json(text)
        usage = payload.get("usage") or {}
        return LLMResult(
            ok=bool(text or data),
            text=text,
            data=data,
            provider=self.name,
            model=model,
            ms=ms,
            tokens_in=int(usage.get("input_tokens") or 0) + int(usage.get("cache_read_input_tokens") or 0),
            tokens_out=int(usage.get("output_tokens") or 0),
            cost=float(payload.get("total_cost_usd") or 0.0),
        )


@dataclass
class CodexCLI:
    """`codex exec` using the person's ChatGPT/Codex login, read-only sandbox."""

    name: str = "codex"
    timeout: int = 300
    default_model: str = "default"

    def available(self) -> bool:
        return shutil.which("codex") is not None

    def complete(self, request: LLMRequest, model: str | None) -> LLMResult:
        model = model or self.default_model
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="dun-codex-") as tmp:
            root = Path(tmp)
            answer = root / "answer.txt"
            cmd = [
                "codex", "exec",
                "--ephemeral", "--skip-git-repo-check",
                "--sandbox", "read-only",
                "--color", "never",
                "--output-last-message", str(answer),
            ]
            if model not in {"default", "auto", ""}:
                cmd += ["--model", model]
            if request.effort:
                cmd += ["-c", f'model_reasoning_effort="{request.effort}"']
            if request.schema:
                schema_path = root / "schema.json"
                schema_path.write_text(json.dumps(request.schema), encoding="utf-8")
                cmd += ["--output-schema", str(schema_path)]
            cmd.append("-")
            try:
                proc = subprocess.run(
                    cmd, input=full_prompt(request), cwd=root, capture_output=True, text=True, timeout=self.timeout
                )
            except subprocess.TimeoutExpired:
                return LLMResult(False, provider=self.name, model=model, ms=_elapsed(start), error="timed out")
            except OSError as exc:
                return LLMResult(False, provider=self.name, model=model, ms=_elapsed(start), error=str(exc))
            text = answer.read_text(encoding="utf-8").strip() if answer.exists() else ""
        ms = _elapsed(start)
        if proc.returncode != 0 or not text:
            return LLMResult(False, provider=self.name, model=model, ms=ms,
                             error=_tail(proc.stderr or proc.stdout) or f"exit {proc.returncode}")
        return LLMResult(True, text, extract_json(text), self.name, model, ms)


@dataclass
class OpenAICompatible:
    """Any `/chat/completions` endpoint: OpenAI, DeepSeek, GLM, Kimi, Ollama, OpenRouter…"""

    name: str
    base_url: str
    key_env: tuple[str, ...] = ()
    default_model: str = ""
    timeout: int = 300
    send_temperature: bool = False
    local: bool = False  # e.g. Ollama: no key, availability means "reachable"
    _reachable: tuple[float, bool] = field(default=(0.0, False), repr=False)

    def _key(self) -> str:
        for env in self.key_env:
            if os.environ.get(env):
                return os.environ[env]
        return ""

    def available(self) -> bool:
        if self.local:
            checked, ok = self._reachable
            if time.monotonic() - checked < 60:
                return ok
            try:
                with urllib.request.urlopen(self.base_url.rstrip("/") + "/models", timeout=1.5):
                    ok = True
            except (urllib.error.URLError, OSError, ValueError):
                ok = False
            self._reachable = (time.monotonic(), ok)
            return ok
        return bool(self._key()) and bool(self.default_model)

    def complete(self, request: LLMRequest, model: str | None) -> LLMResult:
        model = model or self.default_model
        system = request.system
        if request.schema:
            system += "\n\n" + schema_hint(request.schema)
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": request.prompt}],
        }
        if self.name == "openai":
            body["max_completion_tokens"] = request.max_tokens
            if request.effort:
                body["reasoning_effort"] = request.effort
        else:
            body["max_tokens"] = request.max_tokens
        if self.send_temperature:
            body["temperature"] = 0.9 if request.role == "dream" else 0.3
        if request.schema:
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        key = self._key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        url = self.base_url.rstrip("/") + "/chat/completions"
        start = time.monotonic()
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:600]
            return LLMResult(False, provider=self.name, model=model, ms=_elapsed(start), error=f"HTTP {exc.code}: {detail}")
        except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError) as exc:
            return LLMResult(False, provider=self.name, model=model, ms=_elapsed(start), error=str(exc))
        choices = payload.get("choices") or []
        text = ((choices[0].get("message") or {}).get("content") or "") if choices else ""
        usage = payload.get("usage") or {}
        return LLMResult(
            ok=bool(text),
            text=text,
            data=extract_json(text) if request.schema else None,
            provider=self.name,
            model=model,
            ms=_elapsed(start),
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
            error="" if text else "empty response",
        )


@dataclass
class AnthropicAPI:
    """Claude through the official SDK (optional extra: `pip install digital-unconscious[anthropic]`)."""

    name: str = "anthropic"
    timeout: int = 300
    default_model: str = "claude-opus-5-5"

    def available(self) -> bool:
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True

    def complete(self, request: LLMRequest, model: str | None) -> LLMResult:
        import anthropic

        model = model or self.default_model
        output_config: dict[str, Any] = {}
        if request.schema:
            output_config["format"] = {"type": "json_schema", "schema": request.schema}
        if request.effort:
            output_config["effort"] = request.effort
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max(request.max_tokens, 8000),
            "system": request.system,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if output_config:
            kwargs["output_config"] = output_config
        start = time.monotonic()
        try:
            client = anthropic.Anthropic(timeout=self.timeout)
            with client.messages.stream(**kwargs) as stream:
                message = stream.get_final_message()
        except anthropic.APIStatusError as exc:
            return LLMResult(False, provider=self.name, model=model, ms=_elapsed(start),
                             error=f"HTTP {exc.status_code}: {_tail(str(exc.message))}")
        except anthropic.APIConnectionError as exc:
            return LLMResult(False, provider=self.name, model=model, ms=_elapsed(start), error=str(exc))
        if message.stop_reason == "refusal":
            return LLMResult(False, provider=self.name, model=model, ms=_elapsed(start), error="model declined the request")
        text = "".join(block.text for block in message.content if block.type == "text")
        usage = message.usage
        return LLMResult(
            ok=bool(text),
            text=text,
            data=extract_json(text) if request.schema else None,
            provider=self.name,
            model=model,
            ms=_elapsed(start),
            tokens_in=int(getattr(usage, "input_tokens", 0) or 0),
            tokens_out=int(getattr(usage, "output_tokens", 0) or 0),
        )
