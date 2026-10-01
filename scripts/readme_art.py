"""Paint the README's pictures from the sample sea, offscreen.

    python scripts/readme_art.py

Writes docs/assets/en/ and docs/assets/zh/ (pages, dialogs and a day/night hero)
and docs/assets/sea-day.gif / sea-night.gif (a short loop of the moving waterline).
Each language and theme is painted in its own process, because the interface
language and theme are chosen once per run. Needs ffmpeg for the loops;
pngquant and oxipng, when present, keep the pictures small.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
WIDTH, HEIGHT = 1320, 1000
PAGES = ["today", "threads", "thread", "sparks", "spark", "journal", "settings"]
BOTTLE = {
    "en": "Restaurant menus plant a decoy dish. Do pricing pages plant a decoy plan for people buying at 11 p.m.?",
    "zh": "餐厅菜单会放一道诱饵菜。深夜买会员的人，是不是也被定价页上的诱饵套餐推着走？",
}
LOOP_FRAMES, LOOP_FPS, LOOP_BLEND = 96, 12, 20
LOOP_STEP = 0.0625  # phase per frame: twice the app's own pace, so eight seconds show some swimming


def worker(language: str, theme_name: str, out: Path) -> None:
    """Paint every page, both dialogs and (in English) the sea loop frames for one language and theme."""
    home = Path(tempfile.mkdtemp(prefix="dun-art-"))
    os.environ["DUN_HOME"] = str(home)
    from PySide6.QtCore import QRect
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtWidgets import QApplication

    from unconscious.demo import seed_demo

    seed_demo(home, language=language)
    qt = QApplication(sys.argv[:1])
    from unconscious.app import App
    from unconscious.jobs import Jobs
    from unconscious.ui import theme
    from unconscious.ui.charts import Pebbles
    from unconscious.ui.dialogs import JotDialog, SharkDialog
    from unconscious.ui.i18n import set_language, t
    from unconscious.ui.motion import ticker
    from unconscious.ui.widgets import NightPanel
    from unconscious.ui.window import MainWindow

    dark = theme_name == "dark"
    theme.load_fonts()
    ctx = App(home)
    ctx.update_settings({"ui": {"theme": theme_name}})
    set_language(ctx.settings.language)
    theme.apply(qt, dark)
    window = MainWindow(ctx, Jobs(ctx), demo=True)
    window.resize(WIDTH, HEIGHT)
    window.show()
    prefix = f"{language}-{theme_name}"

    def settle() -> None:
        for _ in range(6):
            qt.processEvents()

    spark = ctx.store.sparks(status="pursuing")[0]["id"]
    thread = next(row["id"] for row in ctx.store.threads() if row["name"].startswith(("Bike", "自行车")))
    params = {"thread": {"id": thread}, "sparks": {"status": "new"}, "spark": {"id": spark}}
    for name in PAGES:
        window.go(name, **params.get(name, {}))
        settle()
        window.grab().save(str(out / f"{prefix}-{name}.png"))

    window.go("today")
    settle()
    shark = SharkDialog(window, t("shark.title", day=ctx.store.latest_dream()["day"]), t("shark.body"))
    shark.show()
    settle()
    shark.grab().save(str(out / f"{prefix}-shark.png"))
    shark.close()
    bottle = JotDialog(window)
    bottle.text.setPlainText(BOTTLE[language])
    bottle.show()
    settle()
    bottle.grab().save(str(out / f"{prefix}-bottle.png"))
    bottle.close()

    if language != "en":
        return
    # The loop: a strip of the dream with no words in it, so both READMEs can share it.
    today_pebbles = next(w for w in window.page.findChildren(Pebbles))

    ticker().set_mode("off")  # the loop sets each frame's phase itself
    panel = NightPanel(padding=(52, 0, 52, 34))
    panel.body.addSpacing(150)
    panel.set_fish(5)
    panel.set_shore(Pebbles(today_pebbles.threads, today_pebbles.seed, height=118, labels=False))
    panel.resize(980, panel.sizeHint().height())
    panel.show()
    settle()
    inner = QRect(0, 26, panel.width(), panel.height() - 52)  # square off the rounded corners

    def frame(phase: float) -> QImage:
        panel._phase = phase
        settle()
        return panel.grab(inner).toImage().convertToFormat(QImage.Format.Format_RGB32)

    start = 11.0
    span = LOOP_FRAMES * LOOP_STEP
    for k in range(LOOP_FRAMES):
        image = frame(start + k * LOOP_STEP)
        blend_from = LOOP_FRAMES - LOOP_BLEND
        if k >= blend_from:  # fade into the opening frames so the loop has no seam
            weight = (k - blend_from + 1) / (LOOP_BLEND + 1)
            early = frame(start + k * LOOP_STEP - span)
            painter = QPainter(image)
            painter.setOpacity(weight)
            painter.drawImage(0, 0, early)
            painter.end()
        image.save(str(out / f"sea-{theme_name}-{k:03d}.png"))


def window_frame(light, dark, title: str):
    """Both themes in one window, split along a diagonal: morning on the left, evening on the right."""
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPolygonF

    bar, radius, margin = 34, 16, 56
    w, h = light.width(), light.height() + bar
    canvas = QImage(w + margin * 2, h + margin * 2, QImage.Format.Format_ARGB32_Premultiplied)
    canvas.fill(Qt.GlobalColor.transparent)
    p = QPainter(canvas)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    p.translate(margin, margin)
    for i in range(28, 0, -1):  # a soft shadow, as if the window floated over the sand
        shade = QPainterPath()
        shade.addRoundedRect(QRectF(-i * 0.9, -i * 0.5 + 16, w + i * 1.8, h + i * 1.4), radius + i, radius + i)
        p.fillPath(shade, QColor(20, 40, 70, 4))
    window = QPainterPath()
    window.addRoundedRect(QRectF(0, 0, w, h), radius, radius)
    p.setClipPath(window)
    cut = QPolygonF([QPointF(w * 0.63, 0), QPointF(w + 1, 0), QPointF(w + 1, h + 1), QPointF(w * 0.41, h + 1)])
    for image, side in ((light, None), (dark, cut)):
        p.save()
        if side is not None:
            clip = QPainterPath()
            clip.addPolygon(side)
            p.setClipPath(clip, Qt.ClipOperation.IntersectClip)
        top = QColor(image.pixel(8, 8)).darker(104)
        p.fillRect(QRectF(0, 0, w, bar), top)
        p.drawImage(QPointF(0, bar), image)
        p.setPen(QColor(178, 190, 210) if side is not None else QColor(118, 104, 88))
        p.setFont(QFont("Helvetica Neue", 12))
        p.drawText(QRectF(0, 0, w, bar), Qt.AlignmentFlag.AlignCenter, title)
        p.restore()
    for i, colour in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(colour))
        p.drawEllipse(QPointF(20 + i * 20, bar / 2), 6, 6)
    p.setPen(QPen(QColor(255, 255, 255, 150), 1.6))
    p.drawLine(QPointF(w * 0.63, 0), QPointF(w * 0.41, h))
    p.setClipping(False)
    p.setPen(QPen(QColor(0, 0, 0, 30), 1))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(window)
    p.end()
    return canvas


def compose(raw: Path) -> None:
    from PySide6.QtGui import QGuiApplication, QImage

    QGuiApplication(sys.argv[:1])
    for language in ("en", "zh"):
        target = ASSETS / language
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)
        light = QImage(str(raw / f"{language}-light-today.png"))
        dark = QImage(str(raw / f"{language}-dark-today.png"))
        window_frame(light, dark, "Digital Unconscious").save(str(target / "hero.png"))
        for name in PAGES:
            shutil.copy(raw / f"{language}-light-{name}.png", target / f"{name}.png")
        for name in ("today", "threads", "spark", "journal"):
            shutil.copy(raw / f"{language}-dark-{name}.png", target / f"{name}-dark.png")
        for name in ("shark", "bottle"):
            shutil.copy(raw / f"{language}-light-{name}.png", target / f"{name}.png")
            shutil.copy(raw / f"{language}-dark-{name}.png", target / f"{name}-dark.png")


def loops(raw: Path) -> None:
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found: skipping the sea loops")
        return
    for theme_name, name in (("light", "sea-day"), ("dark", "sea-night")):
        palette = "split[a][b];[a]palettegen=stats_mode=diff:max_colors=256[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(LOOP_FPS),
             "-i", str(raw / f"sea-{theme_name}-%03d.png"), "-vf", palette, "-loop", "0", str(ASSETS / f"{name}.gif")],
            check=True,
        )


def squeeze() -> None:
    pictures = sorted(str(p) for p in ASSETS.glob("*/*.png"))
    if shutil.which("pngquant"):
        subprocess.run(["pngquant", "--quality=82-96", "--speed=1", "--force", "--ext", ".png", *pictures], check=False)
    if shutil.which("oxipng"):
        subprocess.run(["oxipng", "-q", "-o", "3", "--strip", "safe", *pictures], check=False)


def main() -> int:
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3], Path(sys.argv[4]))
        return 0
    raw = Path(tempfile.mkdtemp(prefix="dun-art-raw-"))
    for language in ("en", "zh"):
        for theme_name in ("light", "dark"):
            subprocess.run([sys.executable, __file__, "--worker", language, theme_name, str(raw)], check=True)
    compose(raw)
    loops(raw)
    squeeze()
    for path in sorted(ASSETS.rglob("*")):
        if path.is_file() and path.suffix in {".png", ".gif"}:
            print(f"{path.relative_to(ROOT)}  {path.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
