from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from du_research.utils import iso_now


def _ensure_bytes_key(value: bytes) -> bytes:
    if len(value) == 32:
        return value
    if len(value) > 32:
        return value[:32]
    return value.ljust(32, b"\0")


@dataclass
class CredentialBroker:
    vault_path: Path
    key_path: Path

    def __post_init__(self) -> None:
        self.vault_path.parent.mkdir(parents=True, exist_ok=True)
        self.key_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_key(self) -> bytes:
        env_key = os.environ.get("DU_VAULT_KEY")
        if env_key:
            try:
                return _ensure_bytes_key(base64.urlsafe_b64decode(env_key.encode("utf-8")))
            except Exception:
                return _ensure_bytes_key(env_key.encode("utf-8"))
        if self.key_path.exists():
            return _ensure_bytes_key(base64.urlsafe_b64decode(self.key_path.read_text(encoding="utf-8").encode("utf-8")))
        key = AESGCM.generate_key(bit_length=256)
        self.key_path.write_text(base64.urlsafe_b64encode(key).decode("utf-8"), encoding="utf-8")
        return key

    def _load_vault(self) -> dict[str, Any]:
        if not self.vault_path.exists():
            return {"credentials": {}}
        key = self._load_key()
        payload = json.loads(self.vault_path.read_text(encoding="utf-8"))
        nonce = base64.urlsafe_b64decode(payload["nonce"].encode("utf-8"))
        ciphertext = base64.urlsafe_b64decode(payload["ciphertext"].encode("utf-8"))
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)
        return json.loads(plaintext.decode("utf-8"))

    def _save_vault(self, vault: dict[str, Any]) -> None:
        key = self._load_key()
        nonce = os.urandom(12)
        ciphertext = AESGCM(key).encrypt(nonce, json.dumps(vault, ensure_ascii=False).encode("utf-8"), None)
        payload = {
            "nonce": base64.urlsafe_b64encode(nonce).decode("utf-8"),
            "ciphertext": base64.urlsafe_b64encode(ciphertext).decode("utf-8"),
            "updated_at": iso_now(),
        }
        self.vault_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def set_credential(
        self,
        resource: str,
        username: str,
        password: str,
        *,
        login_url: str | None = None,
        notes: str = "",
        extra_fields: dict[str, str] | None = None,
    ) -> None:
        vault = self._load_vault()
        vault.setdefault("credentials", {})[resource] = {
            "username": username,
            "password": password,
            "login_url": login_url,
            "notes": notes,
            "extra_fields": extra_fields or {},
            "updated_at": iso_now(),
        }
        self._save_vault(vault)

    def get_credential(self, resource: str) -> dict[str, Any] | None:
        return self._load_vault().get("credentials", {}).get(resource)

    def list_resources(self) -> list[str]:
        return sorted(self._load_vault().get("credentials", {}).keys())

