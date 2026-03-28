from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from du_research.ai_backend import ClaudeCodeBackend
from du_research.utils import iso_now


class ManualCheckpointRequired(RuntimeError):
    def __init__(self, message: str, checkpoint_path: Path):
        super().__init__(message)
        self.checkpoint_path = checkpoint_path


class BrowserAutomationRunner:
    def __init__(
        self,
        *,
        runner: str,
        browser: str,
        download_dir: Path,
        screenshot_dir: Path,
        headless: bool,
        timeout_seconds: int,
    ):
        self.runner = runner
        self.browser = browser
        self.download_dir = download_dir
        self.screenshot_dir = screenshot_dir
        self.headless = headless
        self.timeout_seconds = timeout_seconds
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    def execute_task_pack(
        self,
        task_path: Path,
        *,
        credential_lookup: callable | None = None,
    ) -> dict[str, Any]:
        task = json.loads(task_path.read_text(encoding="utf-8"))
        try:
            if self.runner == "claude_code":
                return self._execute_with_claude_code(task)
            return self._execute_with_selenium(task, credential_lookup=credential_lookup)
        except ManualCheckpointRequired as exc:
            return {
                "runner": self.runner,
                "ok": False,
                "status": "manual_checkpoint_required",
                "message": str(exc),
                "checkpoint_path": str(exc.checkpoint_path),
            }

    def _execute_with_claude_code(self, task: dict[str, Any]) -> dict[str, Any]:
        backend = ClaudeCodeBackend(timeout_seconds=self.timeout_seconds)
        prompt = (
            "Use Chrome/computer tools to inspect and download the resources in this JSON task pack. "
            "Never accept terms, solve CAPTCHA, or submit credentials without stopping and stating that "
            "a manual checkpoint is required.\n\n"
            + json.dumps(task, indent=2, ensure_ascii=False)
        )
        response = backend.call(
            prompt,
            mode="strict",
            model="sonnet",
            allowed_tools=["default"],
            max_tokens=4000,
        )
        return {
            "runner": "claude_code",
            "ok": response.ok,
            "session_id": response.session_id,
            "text": response.text,
            "raw": response.raw,
        }

    def _build_driver(self) -> webdriver.Chrome:
        options = ChromeOptions()
        prefs = {
            "download.default_directory": str(self.download_dir.resolve()),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
        }
        options.add_experimental_option("prefs", prefs)
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1440,1200")
        return webdriver.Chrome(options=options)

    def _execute_with_selenium(
        self,
        task: dict[str, Any],
        *,
        credential_lookup: callable | None = None,
    ) -> dict[str, Any]:
        if "flow" in task:
            return self._execute_flow(task["flow"], credential_lookup=credential_lookup)
        return {
            "runner": "selenium",
            "ok": False,
            "error": "Task pack does not contain a selenium flow.",
        }

    def _execute_flow(
        self,
        flow: list[dict[str, Any]],
        *,
        credential_lookup: callable | None = None,
    ) -> dict[str, Any]:
        screenshots = []
        downloaded_files = []
        driver = self._build_driver()
        try:
            wait = WebDriverWait(driver, self.timeout_seconds)
            for index, step in enumerate(flow, start=1):
                action = step.get("action")
                if action == "open":
                    driver.get(step["url"])
                elif action == "click_css":
                    wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, step["selector"]))).click()
                elif action == "type_css":
                    element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, step["selector"])))
                    element.clear()
                    value = step.get("value", "")
                    if value.startswith("credential:"):
                        resource, field = value.split(":", 2)[1:]
                        credential = credential_lookup(resource) if credential_lookup else None
                        value = (credential or {}).get(field, "")
                    element.send_keys(value)
                elif action == "press_enter_css":
                    element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, step["selector"])))
                    element.send_keys(Keys.ENTER)
                elif action == "wait_css":
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, step["selector"])))
                elif action == "download_url":
                    driver.get(step["url"])
                    time.sleep(step.get("sleep_seconds", 5))
                elif action == "screenshot":
                    target = self.screenshot_dir / f"step_{index:02d}.png"
                    driver.save_screenshot(str(target))
                    screenshots.append(str(target))
                elif action == "manual_checkpoint":
                    checkpoint_path = self.screenshot_dir / f"checkpoint_{index:02d}.json"
                    checkpoint_path.write_text(
                        json.dumps(
                            {
                                "timestamp": iso_now(),
                                "message": step.get("message", "Manual action required"),
                                "current_url": driver.current_url,
                            },
                            indent=2,
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    raise ManualCheckpointRequired(step.get("message", "Manual action required"), checkpoint_path)
            downloaded_files = [str(path) for path in self.download_dir.glob("*") if path.is_file()]
            return {
                "runner": "selenium",
                "ok": True,
                "screenshots": screenshots,
                "downloaded_files": downloaded_files,
            }
        finally:
            driver.quit()
