from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from du_research.ai_backend import AIBackend


ANALYSIS_CODER_SYSTEM_PROMPT = """You are the Analysis Engine.
Write a complete Python script that reads the provided CSV path, performs descriptive analysis,
generates chart files in the requested output directory, and writes results.json.
Return ONLY JSON: {"script": "...python code..."}"""


@dataclass
class AnalysisCoderAgent:
    backend: AIBackend
    model: str = "sonnet"
    system_prompt: str | None = None

    def generate_script(self, payload: dict[str, Any]) -> str | None:
        response = self.backend.call(
            json.dumps(payload, indent=2, ensure_ascii=False),
            mode="balanced",
            system=self.system_prompt or ANALYSIS_CODER_SYSTEM_PROMPT,
            model=self.model,
            max_tokens=4000,
        )
        if not response.ok:
            return None
        try:
            return json.loads(response.text).get("script")
        except json.JSONDecodeError:
            return None


def execute_script(script_path: Path, timeout_seconds: int) -> dict[str, Any]:
    result = subprocess.run(
        ["python", str(script_path)],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
