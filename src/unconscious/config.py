"""Settings live in one human-editable TOML file inside the app home.

The dashboard edits the same file, so there is exactly one source of truth.
Python can read TOML but not write it, hence the small writer at the bottom;
it only needs to handle the flat shapes this module produces.
"""

from __future__ import annotations

import locale
import os
import platform
import subprocess
import tomllib
from dataclasses import asdict, dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from typing import Any


def app_home() -> Path:
    raw = os.environ.get("DUN_HOME")
    return Path(raw).expanduser() if raw else Path.home() / ".digital-unconscious"


@dataclass
class You:
    name: str = ""
    # Free text: who you are and what kind of ideas you want. The single most
    # useful steering input for the dream.
    persona: str = ""
    focus: list[str] = field(default_factory=list)
    language: str = "auto"  # auto | en | zh


@dataclass
class Sense:
    enabled: bool = True
    interval_seconds: int = 15
    idle_seconds: int = 120
    capture_titles: bool = True
    capture_urls: bool = True
    # Never recorded at all: the sample is dropped before it touches disk.
    quiet_apps: list[str] = field(
        default_factory=lambda: ["1Password", "Bitwarden", "Keychain Access", "LastPass", "KeePassXC"]
    )
    quiet_domains: list[str] = field(
        default_factory=lambda: ["bank", "paypal.com", "accounts.google.com", "login.", "signin.", "auth."]
    )
    # Recorded as "time in <app>" only; window titles are discarded.
    private_apps: list[str] = field(
        default_factory=lambda: [
            "Messages", "WeChat", "微信", "Slack", "Mail", "Telegram", "WhatsApp",
            "Discord", "Signal", "Microsoft Outlook", "Microsoft Teams", "QQ", "Lark", "飞书",
        ]
    )
    retention_days: int = 90
    # Glance less often while running on battery (and back off further while
    # attention stays on one thing). Dwell time is credited from real elapsed
    # time either way, so totals stay accurate.
    battery_saver: bool = True


@dataclass
class Dream:
    time: str = "21:30"
    auto: bool = True
    sparks: int = 3
    candidates: int = 6
    critique: bool = True


@dataclass
class Models:
    digest: str = "auto"
    dream: str = "auto"
    critique: str = "auto"
    dive: str = "auto"
    fallback: bool = True
    timeout_seconds: int = 300
    # Anthropic and OpenAI do not serve mainland China: while the connection is there,
    # Claude, Codex, the Anthropic API and OpenAI stay ashore (see llm/region.py).
    region_guard: bool = True
    hold_regions: list[str] = field(default_factory=lambda: ["CN"])
    # Let Claude search the web and read papers while it dreams and dives (llm/research.py).
    research: bool = True


@dataclass
class Ui:
    theme: str = "system"  # system | light | dark
    start_hidden: bool = False
    motion: str = "auto"  # auto (full on mains, calm on battery) | full | calm | off


SECTIONS = {"you": You, "sense": Sense, "dream": Dream, "models": Models, "ui": Ui}


@dataclass
class Settings:
    you: You = field(default_factory=You)
    sense: Sense = field(default_factory=Sense)
    dream: Dream = field(default_factory=Dream)
    models: Models = field(default_factory=Models)
    ui: Ui = field(default_factory=Ui)
    # Extra OpenAI-compatible endpoints, e.g. [providers.ollama]
    # kind = "openai", base_url = "http://localhost:11434/v1", model = "qwen3:14b"
    providers: dict[str, dict[str, Any]] = field(default_factory=dict)
    path: Path | None = None

    @property
    def language(self) -> str:
        return resolve_language(self.you.language)

    def to_dict(self) -> dict[str, Any]:
        data = {name: asdict(getattr(self, name)) for name in SECTIONS}
        data["providers"] = {k: dict(v) for k, v in self.providers.items()}
        return data

    def update(self, patch: dict[str, Any]) -> list[str]:
        """Apply a partial update (as sent by the dashboard); returns changed keys."""
        changed: list[str] = []
        for section_name, values in (patch or {}).items():
            if section_name == "providers" and isinstance(values, dict):
                self.providers = {str(k): dict(v) for k, v in values.items() if isinstance(v, dict)}
                changed.append("providers")
                continue
            if section_name not in SECTIONS or not isinstance(values, dict):
                continue
            section = getattr(self, section_name)
            for f in fields(section):
                if f.name not in values:
                    continue
                coerced = _coerce(values[f.name], getattr(section, f.name))
                if coerced != getattr(section, f.name):
                    setattr(section, f.name, coerced)
                    changed.append(f"{section_name}.{f.name}")
        _clamp(self)
        return changed

    def save(self) -> Path:
        path = self.path or app_home() / "config.toml"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".toml.tmp")
        tmp.write_text(_HEADER + dump_toml(self.to_dict()), encoding="utf-8")
        tmp.replace(path)
        self.path = path
        return path


_HEADER = (
    "# Digital Unconscious settings. Edited by `dun` and the dashboard; safe to edit by hand.\n"
    "# API keys never belong here: optional hosted providers read them from environment variables.\n\n"
)


def load_settings(path: Path | None = None) -> Settings:
    path = path or app_home() / "config.toml"
    settings = Settings(path=path)
    if path.exists():
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
        settings.update({k: v for k, v in raw.items() if k != "providers"})
        providers = raw.get("providers")
        if isinstance(providers, dict):
            settings.providers = {str(k): dict(v) for k, v in providers.items() if isinstance(v, dict)}
    _clamp(settings)
    return settings


def _coerce(value: Any, current: Any) -> Any:
    if isinstance(current, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(current, int):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return current
    if isinstance(current, list):
        if isinstance(value, str):
            value = [part for chunk in value.splitlines() for part in chunk.split(",")]
        if not isinstance(value, list):
            return current
        cleaned = []
        for item in value:
            text = str(item).strip()
            if text and text not in cleaned:
                cleaned.append(text)
        return cleaned
    return "" if value is None else str(value).strip()


def _clamp(settings: Settings) -> None:
    s = settings.sense
    s.interval_seconds = max(5, min(s.interval_seconds, 300))
    s.idle_seconds = max(30, min(s.idle_seconds, 3600))
    s.retention_days = max(7, min(s.retention_days, 3650))
    d = settings.dream
    d.sparks = max(1, min(d.sparks, 6))
    d.candidates = max(d.sparks, min(d.candidates, 10))
    if not _valid_clock(d.time):
        d.time = "21:30"
    if settings.you.language not in {"auto", "en", "zh"}:
        settings.you.language = "auto"
    settings.models.timeout_seconds = max(30, min(settings.models.timeout_seconds, 1800))
    settings.models.hold_regions = [code.strip().upper() for code in settings.models.hold_regions if code.strip()]
    if settings.ui.theme not in {"system", "light", "dark"}:
        settings.ui.theme = "system"
    if settings.ui.motion not in {"auto", "full", "calm", "off"}:
        settings.ui.motion = "auto"


def _valid_clock(text: str) -> bool:
    try:
        hour, minute = text.split(":")
        return 0 <= int(hour) < 24 and 0 <= int(minute) < 60
    except ValueError:
        return False


@lru_cache(maxsize=4)
def resolve_language(preference: str) -> str:
    if preference in {"en", "zh"}:
        return preference
    candidates = [os.environ.get(name, "") for name in ("DUN_LANG", "LC_ALL", "LC_MESSAGES", "LANG")]
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(
                ["defaults", "read", "-g", "AppleLanguages"], capture_output=True, text=True, timeout=2
            ).stdout
            first = next((line.strip(' ",()') for line in out.splitlines() if line.strip(' ",()')), "")
            candidates.insert(1, first)
        except (OSError, subprocess.SubprocessError):
            pass
    candidates.append(locale.getlocale()[0] or "")
    for value in candidates:
        if value:
            return "zh" if value.lower().startswith("zh") else "en"
    return "en"


# --------------------------------------------------------------------------- TOML


def dump_toml(data: dict[str, Any]) -> str:
    lines: list[str] = []
    for table, values in data.items():
        if not isinstance(values, dict):
            continue
        nested = {k: v for k, v in values.items() if isinstance(v, dict)}
        plain = {k: v for k, v in values.items() if not isinstance(v, dict)}
        if plain or not nested:
            lines.append(f"[{table}]")
            lines.extend(f"{_key(k)} = {_value(v)}" for k, v in plain.items())
            lines.append("")
        for sub, sub_values in nested.items():
            lines.append(f"[{table}.{_key(sub)}]")
            lines.extend(
                f"{_key(k)} = {_value(v)}" for k, v in sub_values.items() if not isinstance(v, dict)
            )
            lines.append("")
    return "\n".join(lines)


def _key(key: str) -> str:
    bare = key.replace("_", "").replace("-", "").isalnum() and key.isascii()
    return key if bare else _string(key)


def _value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_value(v) for v in value) + "]"
    return _string(str(value))


def _string(text: str) -> str:
    escaped = (
        text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
    )
    return f'"{escaped}"'
