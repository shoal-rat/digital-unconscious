"""Render every page of the app to PNG, offscreen, from the demo memory.

    python scripts/snapshots.py [out_dir] [--lang zh] [--dark]

Used to check the interface without clicking through it, and to refresh the
screenshots in docs/assets.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = Path(args[0] if args else "snapshots").resolve()
    out.mkdir(parents=True, exist_ok=True)
    language = "zh" if "--lang=zh" in sys.argv or "--zh" in sys.argv else "en"
    dark = "--dark" in sys.argv
    home = Path(tempfile.mkdtemp(prefix="dun-snap-"))
    os.environ["DUN_HOME"] = str(home)

    from PySide6.QtWidgets import QApplication

    from unconscious.demo import seed_demo

    seed_demo(home, language=language)
    app = QApplication(sys.argv)
    from unconscious.app import App
    from unconscious.jobs import Jobs
    from unconscious.ui import theme
    from unconscious.ui.i18n import set_language
    from unconscious.ui.window import MainWindow

    theme.load_fonts()
    ctx = App(home)
    ctx.update_settings({"ui": {"theme": "dark" if dark else "light"}})
    set_language(ctx.settings.language)
    theme.apply(app, dark)
    window = MainWindow(ctx, Jobs(ctx), demo=True)
    width, height = 1320, int(os.environ.get("SNAP_HEIGHT", "1000"))
    window.resize(width, height)
    window.show()
    suffix = ("-zh" if language == "zh" else "") + ("-dark" if dark else "")
    spark = ctx.store.sparks(status="pursuing")[0]["id"]
    thread = next(t["id"] for t in ctx.store.threads() if t["name"].startswith(("Bike", "自行车")))
    routes = [
        ("today", {}), ("threads", {}), ("thread", {"id": thread}), ("sparks", {"status": "new"}),
        ("spark", {"id": spark}), ("journal", {}), ("settings", {}),
    ]
    for name, params in routes:
        window.go(name, **params)
        for _ in range(5):
            app.processEvents()
        window.grab().save(str(out / f"{name}{suffix}.png"))
        page = window.page.widget()
        page.resize(window.page.viewport().width(), page.sizeHint().height())
        app.processEvents()
        page.grab().save(str(out / f"{name}{suffix}-full.png"))
    window.close()
    shutil.rmtree(home, ignore_errors=True)  # the borrowed sea leaves nothing behind
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
