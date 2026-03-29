from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlparse


def _extract_domain(url: str) -> str | None:
    if not url or "://" not in url:
        return None
    netloc = urlparse(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc or None


def _build_login_flow(resource: str, credential: dict[str, Any]) -> list[dict[str, Any]]:
    extra = credential.get("extra_fields", {}) or {}
    username_selector = extra.get("username_selector")
    password_selector = extra.get("password_selector")
    submit_selector = extra.get("submit_selector")
    post_login_wait_selector = extra.get("post_login_wait_selector")
    login_url = credential.get("login_url")
    if not login_url or not username_selector or not password_selector:
        return []

    flow = [
        {"action": "open", "url": login_url},
        {"action": "type_css", "selector": username_selector, "value": f"credential:{resource}:username"},
        {"action": "type_css", "selector": password_selector, "value": f"credential:{resource}:password"},
    ]
    if submit_selector:
        flow.append({"action": "click_css", "selector": submit_selector})
    else:
        flow.append({"action": "press_enter_css", "selector": password_selector})
    if post_login_wait_selector:
        flow.append({"action": "wait_css", "selector": post_login_wait_selector})
    flow.append({"action": "screenshot"})
    return flow


def export_computer_use_task(
    run_dir: Path,
    *,
    credential_lookup: Callable[[str], dict[str, Any] | None] | None = None,
    institutional_proxy_url: str | None = None,
    checkpoint_policy: str = "best_effort",
) -> Path:
    checkpoint_policy = checkpoint_policy.strip().lower()
    strict_checkpoints = checkpoint_policy == "strict"
    literature_path = run_dir / "01_literature" / "papers.json"
    datasets_path = run_dir / "03_data_sources" / "datasets.json"
    payload: dict[str, Any] = {
        "mode": "supervised",
        "runner": "claude-code-computer-use",
        "run_dir": str(run_dir),
        "allowlisted_domains": [],
        "credential_resources": [],
        "institutional_proxy_url": institutional_proxy_url or "",
        "checkpoint_policy": checkpoint_policy,
        "human_approval_required_for": (
            [
                "CAPTCHA or consent walls",
                "payments or subscription changes",
                "terms-of-service acceptance",
            ]
            if strict_checkpoints
            else [
                "Only when blocked by CAPTCHA/MFA/consent or payment walls",
            ]
        ),
        "tasks": [],
        "flow": [],
    }

    flow: list[dict[str, Any]] = []
    logged_resources: set[str] = set()

    if literature_path.exists():
        literature = json.loads(literature_path.read_text(encoding="utf-8"))
        for paper in literature.get("papers", [])[:5]:
            url = paper.get("url") or ""
            pdf_url = paper.get("pdf_url")
            resource = _extract_domain(url) or _extract_domain(pdf_url or "")
            credential = credential_lookup(resource) if (resource and credential_lookup) else None
            if resource:
                payload["allowlisted_domains"].append(resource)
            if credential and resource:
                payload["credential_resources"].append(resource)
                if resource not in logged_resources:
                    flow.extend(_build_login_flow(resource, credential))
                    logged_resources.add(resource)

            task = {
                "type": "inspect-paper",
                "title": paper.get("title"),
                "url": url,
                "pdf_url": pdf_url,
                "doi": paper.get("doi"),
                "credential_resource": resource if credential else None,
                "institutional_proxy_url": institutional_proxy_url or "",
            }
            payload["tasks"].append(task)
            if url:
                flow.append({"action": "open", "url": url})
                flow.append({"action": "screenshot"})
            if pdf_url:
                flow.append({"action": "download_url", "url": pdf_url, "sleep_seconds": 5})
            elif strict_checkpoints and resource and credential:
                flow.append(
                    {
                        "action": "manual_checkpoint",
                        "message": f"Paper download for {paper.get('title', 'untitled')} requires gated navigation on {resource}.",
                    }
                )

    if datasets_path.exists():
        datasets = json.loads(datasets_path.read_text(encoding="utf-8"))
        for dataset in datasets.get("datasets", [])[:5]:
            url = dataset.get("url") or ""
            resource = _extract_domain(url)
            credential = credential_lookup(resource) if (resource and credential_lookup) else None
            if resource:
                payload["allowlisted_domains"].append(resource)
            if credential and resource:
                payload["credential_resources"].append(resource)
                if resource not in logged_resources:
                    flow.extend(_build_login_flow(resource, credential))
                    logged_resources.add(resource)

            task = {
                "type": "inspect-dataset",
                "title": dataset.get("title"),
                "url": url,
                "access": dataset.get("access"),
                "credential_resource": resource if credential else None,
            }
            payload["tasks"].append(task)
            if url:
                dataset_url = url
                if institutional_proxy_url and resource:
                    dataset_url = institutional_proxy_url.rstrip("/") + "/" + quote(url, safe="")
                flow.append({"action": "open", "url": dataset_url})
                flow.append({"action": "screenshot"})
            if strict_checkpoints and dataset.get("access") not in {"open", "public"}:
                flow.append(
                    {
                        "action": "manual_checkpoint",
                        "message": f"Dataset access for {dataset.get('title', 'untitled')} may require consent or institution confirmation.",
                    }
                )

    payload["allowlisted_domains"] = sorted(set(filter(None, payload["allowlisted_domains"])))
    payload["credential_resources"] = sorted(set(filter(None, payload["credential_resources"])))
    payload["flow"] = flow
    output_path = run_dir / "computer_use_task.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path
