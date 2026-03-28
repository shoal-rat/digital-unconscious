from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_HEADERS = {
    "User-Agent": "du-research/0.1 (+local-first research pipeline)",
    "Accept": "application/json, application/atom+xml, text/xml;q=0.9, */*;q=0.8",
}


def fetch_bytes(url: str, timeout: int = 15, headers: dict[str, str] | None = None) -> bytes:
    request_headers = dict(DEFAULT_HEADERS)
    if headers:
        request_headers.update(headers)
    request = Request(url, headers=request_headers)
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_json(url: str, timeout: int = 15, headers: dict[str, str] | None = None) -> dict:
    return json.loads(fetch_bytes(url, timeout=timeout, headers=headers).decode("utf-8"))


def build_url(base: str, params: dict[str, str | int]) -> str:
    return f"{base}?{urlencode(params)}"
