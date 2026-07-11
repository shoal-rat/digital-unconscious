"""Subscription-backed local CLI runners: no provider API key required."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from du_research.backends.base import (
    AIResponse,
    compose_prompt,
    image_media_type,
    parse_structured_text,
    reasoning_effort,
    usage_value,
)

logger = logging.getLogger(__name__)


def _suffix(data: bytes) -> str:
    return {"image/jpeg": ".jpg", "image/webp": ".webp"}.get(image_media_type(data), ".png")


def _write_images(directory: Path, images: list[bytes] | None) -> list[Path]:
    paths: list[Path] = []
    for index, data in enumerate(images or []):
        path = directory / f"input-{index + 1}{_suffix(data)}"
        path.write_bytes(data)
        paths.append(path)
    return paths


def _command_error(name: str, result: subprocess.CompletedProcess[str]) -> AIResponse:
    message = (result.stderr or result.stdout or f"{name} exited {result.returncode}").strip()
    logger.warning("%s failed: %s", name, message[:500])
    return AIResponse(
        text="", raw={"error": message[:2000], "provider": name, "exit_code": result.returncode}
    )


@dataclass
class CodexCLIBackend:
    """Run `codex exec` with the user's ChatGPT/Codex login.

    Each call uses a fresh temporary workspace and a read-only sandbox. The
    model can reason, search when requested, and inspect attached images, but it
    cannot read the user's project or mutate files.
    """

    timeout_seconds: int = 300
    model_override: str | None = None

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
        if not shutil.which("codex"):
            return AIResponse(text="", raw={"error": "codex CLI not found", "provider": "codex"})
        with tempfile.TemporaryDirectory(prefix="du-codex-") as temp:
            root = Path(temp)
            output_path = root / "answer.txt"
            schema_path = root / "schema.json"
            image_paths = _write_images(root, images)
            cmd = [
                "codex",
                "exec",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--color",
                "never",
                "--output-last-message",
                str(output_path),
                "--json",
            ]
            requested_model = model or self.model_override
            if requested_model and requested_model not in {"default", "auto"}:
                cmd.extend(["--model", requested_model])
            if json_schema:
                schema_path.write_text(json.dumps(json_schema), encoding="utf-8")
                cmd.extend(["--output-schema", str(schema_path)])
            if web_search:
                cmd.extend(["--config", 'web_search="live"'])
            for path in image_paths:
                cmd.extend(["--image", str(path)])
            cmd.append("-")
            full_prompt = compose_prompt(prompt, system, mode, max_tokens)
            try:
                result = subprocess.run(
                    cmd,
                    input=full_prompt,
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                return AIResponse(text="", raw={"error": "timeout", "provider": "codex"})
            if result.returncode != 0:
                return _command_error("codex", result)
            text = output_path.read_text(encoding="utf-8").strip() if output_path.exists() else ""
            thread_id = None
            events: list[dict[str, Any]] = []
            for line in result.stdout.splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                events.append(event)
                if event.get("type") == "thread.started":
                    thread_id = event.get("thread_id")
            return AIResponse(
                text=text,
                model=requested_model or "codex-default",
                session_id=thread_id,
                raw={
                    "provider": "codex",
                    "subscription_auth": True,
                    "events": events[-8:],
                    "ignored": {
                        "session_id": bool(session_id),
                        "allowed_tools": bool(allowed_tools),
                        "chrome": use_chrome,
                        "agent": agent,
                        "max_turns": max_turns,
                    },
                },
                structured=parse_structured_text(text, json_schema),
            )


@dataclass
class ClaudeCodeBackend:
    """Run `claude -p` with the user's Claude subscription login."""

    timeout_seconds: int = 300
    model_override: str | None = None

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
        if not shutil.which("claude"):
            return AIResponse(text="", raw={"error": "claude CLI not found", "provider": "claude_code"})
        with tempfile.TemporaryDirectory(prefix="du-claude-") as temp:
            root = Path(temp)
            image_paths = _write_images(root, images)
            full_prompt = compose_prompt(prompt, system, mode, max_tokens)
            if image_paths:
                attachments = ", ".join(str(path) for path in image_paths)
                full_prompt += (
                    f"\n\nInspect the attached local image file(s) with the Read tool: {attachments}"
                )
            cmd = [
                "claude",
                "-p",
                "--output-format",
                "json",
                "--safe-mode",
                "--permission-mode",
                "dontAsk",
                "--no-session-persistence",
                "--max-turns",
                str(max_turns),
            ]
            requested_model = model or self.model_override
            if requested_model and requested_model not in {"default", "auto"}:
                cmd.extend(["--model", requested_model])
            if json_schema:
                cmd.extend(["--json-schema", json.dumps(json_schema)])
            effort = reasoning_effort(think)
            if effort:
                cmd.extend(["--effort", effort])
            tools = list(allowed_tools or [])
            if image_paths and "Read" not in tools:
                tools.append("Read")
            if web_search and "WebSearch" not in tools:
                tools.append("WebSearch")
            cmd.extend(["--tools", ",".join(tools) if tools else ""])
            if agent:
                cmd.extend(["--agent", agent])
            if use_chrome:
                cmd.append("--chrome")
            if session_id:
                cmd.extend(["--resume", session_id])
            try:
                result = subprocess.run(
                    cmd,
                    input=full_prompt,
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                return AIResponse(text="", raw={"error": "timeout", "provider": "claude_code"})
            if result.returncode != 0:
                return _command_error("claude_code", result)
            try:
                parsed = json.loads(result.stdout)
            except json.JSONDecodeError:
                return AIResponse(
                    text=result.stdout.strip(),
                    model=requested_model or "claude-default",
                    raw={"provider": "claude_code"},
                )
            if parsed.get("is_error"):
                return AIResponse(
                    text="", raw={"error": parsed.get("result", "unknown error"), "provider": "claude_code"}
                )
            text = parsed.get("result", "")
            usage = parsed.get("usage", {})
            return AIResponse(
                text=text,
                model=requested_model or "claude-default",
                session_id=parsed.get("session_id"),
                input_tokens=usage_value(usage, "input_tokens"),
                output_tokens=usage_value(usage, "output_tokens"),
                cost_usd=float(parsed.get("total_cost_usd", 0.0) or 0.0),
                raw={**parsed, "provider": "claude_code", "subscription_auth": True},
                structured=parsed.get("structured_output") or parse_structured_text(text, json_schema),
            )
