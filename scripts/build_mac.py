"""Build Digital Unconscious.app and a drag-to-install disk image.

    python -m pip install -e ".[mac]"
    python scripts/build_mac.py

Paints the app icon and the disk image's background with Qt, bundles the app with
PyInstaller, keeps only this Mac's architecture (PySide6 ships Intel and Apple Silicon
in every library), signs it ad hoc, and writes build/Digital-Unconscious-<version>.dmg.
macOS only. Without a Developer ID the app is not notarized: a copy downloaded from the
internet opens the first time with right-click → Open.
"""

from __future__ import annotations

import json
import math
import os
import platform
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
NAME = "Digital Unconscious"
BUNDLE_ID = "com.digital-unconscious.app"
sys.path.insert(0, str(ROOT / "src"))

from unconscious import __version__  # noqa: E402

# --------------------------------------------------------------------------- pictures


def paint_background(scale: int):
    """The disk image's window: linen, a strip of the bay, and the way to Applications."""
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPainterPath, QPen

    from unconscious.ui import theme

    w, h = 640 * scale, 420 * scale
    image = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QColor("#f6f1e8"))
    p = QPainter(image)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    sky = QLinearGradient(QPointF(0, 0), QPointF(0, h))
    sky.setColorAt(0, QColor("#fbf7f0"))
    sky.setColorAt(1, QColor("#efe5d4"))
    p.fillRect(QRectF(0, 0, w, h), sky)
    sea = QPainterPath()
    base = h - 70 * scale
    sea.moveTo(0, h)
    sea.lineTo(0, base)
    x = 0.0
    while x <= w:
        sea.lineTo(x, base + math.sin(x / (46 * scale)) * 5 * scale)
        x += 4 * scale
    sea.lineTo(w, h)
    sea.closeSubpath()
    water = QLinearGradient(QPointF(0, base), QPointF(0, h))
    water.setColorAt(0, QColor("#55c3cf"))
    water.setColorAt(1, QColor("#1a77c2"))
    p.fillPath(sea, water)
    theme.load_fonts()
    title = QFont("Fraunces Soft")
    title.setPixelSize(30 * scale)
    title.setWeight(QFont.Weight.Light)
    p.setPen(QColor("#1f2d3d"))
    p.setFont(title)
    p.drawText(QRectF(0, 34 * scale, w, 44 * scale), Qt.AlignmentFlag.AlignCenter, NAME)
    small = QFont("Figtree")
    small.setPixelSize(14 * scale)
    p.setFont(small)
    p.setPen(QColor("#7a6f62"))
    p.drawText(QRectF(0, 80 * scale, w, 24 * scale), Qt.AlignmentFlag.AlignCenter,
               "Drag into Applications  ·  拖进「应用程序」")
    arrow = QPainterPath()
    start, end = QPointF(250 * scale, 205 * scale), QPointF(390 * scale, 205 * scale)
    arrow.moveTo(start)
    arrow.cubicTo(QPointF(290 * scale, 190 * scale), QPointF(350 * scale, 220 * scale), end)
    p.setPen(QPen(QColor("#2f6db1"), 3 * scale, Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap))
    p.drawPath(arrow)
    p.setPen(QPen(QColor("#2f6db1"), 3 * scale, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    p.drawLine(end, QPointF(end.x() - 12 * scale, end.y() - 9 * scale))
    p.drawLine(end, QPointF(end.x() - 12 * scale, end.y() + 9 * scale))
    p.end()
    return image


def paint_icon_layers(folder: Path) -> None:
    """The layers of a Liquid Glass icon, each on a full-bleed 1024 canvas: the sea (the lower
    half of the sun's disc, with foam) and the horizon (the sun's outline and the sea's edge).
    macOS draws the tile, its corners, its depth and its light; we only give it the drawing."""
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QColor, QImage, QLinearGradient, QPainter, QPainterPath, QPen

    centre, radius = QPointF(512, 540), 300.0

    def canvas() -> tuple[QImage, QPainter]:
        image = QImage(1024, 1024, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        return image, painter

    image, p = canvas()
    sea = QPainterPath()
    sea.moveTo(centre.x() - radius, centre.y())
    sea.arcTo(QRectF(centre.x() - radius, centre.y() - radius, 2 * radius, 2 * radius), 180, 180)
    sea.closeSubpath()
    water = QLinearGradient(QPointF(0, centre.y()), QPointF(0, centre.y() + radius))
    water.setColorAt(0, QColor("#5a92cf"))
    water.setColorAt(0.5, QColor("#2f6db1"))
    water.setColorAt(1, QColor("#1d4f86"))
    p.fillPath(sea, water)
    p.setPen(QPen(QColor("#f6f1e8"), 24, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    for offset, half in ((114, 180), (198, 126)):
        wave = QPainterPath()
        y, x = centre.y() + offset, centre.x() - half
        wave.moveTo(x, y)
        step = 2 * half / 4
        for _ in range(4):
            wave.cubicTo(QPointF(x + step * 0.3, y - 19), QPointF(x + step * 0.7, y + 19), QPointF(x + step, y))
            x += step
        p.drawPath(wave)
    p.end()
    image.save(str(folder / "sea.png"))

    image, p = canvas()
    p.setPen(QPen(QColor("#1f2d3d"), 36, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(centre, radius, radius)
    p.drawLine(QPointF(centre.x() - radius - 92, centre.y()), QPointF(centre.x() + radius + 92, centre.y()))
    p.end()
    image.save(str(folder / "horizon.png"))


def make_icon() -> tuple[Path, Path]:
    """An Icon Composer icon, compiled by actool into Assets.car (the Liquid Glass icon newer macOS
    draws in the Dock, the same size as Apple's own) and AppIcon.icns (for macOS before 26)."""
    source = BUILD / "AppIcon.icon"
    shutil.rmtree(source, ignore_errors=True)
    (source / "Assets").mkdir(parents=True)
    paint_icon_layers(source / "Assets")
    (source / "icon.json").write_text(json.dumps({
        "fill": {"automatic-gradient": "extended-srgb:0.97647,0.95294,0.91765,1.00000"},  # linen
        "groups": [
            {"layers": [{"image-name": "horizon.png", "name": "horizon"}]},
            {"layers": [{"image-name": "sea.png", "name": "sea"}]},
        ],
        "supported-platforms": {"squares": "shared"},
    }, indent=2))
    compiled = BUILD / "icon"
    shutil.rmtree(compiled, ignore_errors=True)
    compiled.mkdir()
    subprocess.run(["xcrun", "actool", str(source), "--compile", str(compiled), "--platform", "macosx",
                    "--minimum-deployment-target", "12.0", "--app-icon", "AppIcon",
                    "--output-partial-info-plist", str(BUILD / "icon-info.plist")],
                   check=True, capture_output=True, cwd=str(BUILD))
    return compiled / "Assets.car", compiled / "AppIcon.icns"


def make_background(target: Path) -> Path:
    one, two = BUILD / "background.png", BUILD / "background@2x.png"
    paint_background(1).save(str(one))
    paint_background(2).save(str(two))
    subprocess.run(["tiffutil", "-cathidpicheck", str(one), str(two), "-out", str(target)], check=True,
                   capture_output=True)
    return target


# --------------------------------------------------------------------------- the app


SPEC = """\
# Generated by scripts/build_mac.py
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

a = Analysis(
    [{launcher!r}],
    pathex=[{src!r}],
    datas=collect_data_files("unconscious"),
    hiddenimports=collect_submodules("unconscious"),
    excludes=["tkinter", "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtWebEngineCore", "PySide6.QtMultimedia"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name={name!r}, console=False, argv_emulation=False,
          codesign_identity=None, entitlements_file=None)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name={name!r})
app = BUNDLE(coll, name={app!r}, icon={icon!r}, bundle_identifier={bundle_id!r}, version={version!r},
             info_plist={plist!r})
"""


def info_plist() -> dict:
    return {
        "CFBundleName": NAME,
        "CFBundleDisplayName": NAME,
        "CFBundleShortVersionString": __version__,
        "CFBundleIconName": "AppIcon",  # Assets.car: the Liquid Glass icon
        "CFBundleVersion": __version__,
        "LSMinimumSystemVersion": "12.0",
        "LSApplicationCategoryType": "public.app-category.productivity",
        "NSHighResolutionCapable": True,
        "NSAppleEventsUsageDescription": (
            "Digital Unconscious asks your browser for the address of the front tab, so it knows which page you "
            "are reading. Nothing else is read."
        ),
        "NSHumanReadableCopyright": "MIT licensed",
    }


def bundle(icon: Path) -> Path:
    spec = BUILD / "digital-unconscious.spec"
    spec.write_text(SPEC.format(
        launcher=str(ROOT / "packaging" / "macos" / "launcher.py"), src=str(ROOT / "src"), name=NAME,
        app=f"{NAME}.app", icon=str(icon), bundle_id=BUNDLE_ID, version=__version__, plist=info_plist(),
    ))
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--log-level", "WARN",
                    "--distpath", str(BUILD / "dist"), "--workpath", str(BUILD / "work"), str(spec)], check=True)
    return BUILD / "dist" / f"{NAME}.app"


def thin(app: Path) -> int:
    """Keep only this Mac's half of every universal binary."""
    arch = platform.machine()  # arm64 or x86_64
    saved = 0
    for path in app.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        with path.open("rb") as handle:
            magic = handle.read(4)
        if magic not in (b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"):  # universal (fat) Mach-O
            continue
        archs = subprocess.run(["lipo", "-archs", str(path)], capture_output=True, text=True).stdout.split()
        if arch not in archs or len(archs) < 2:
            continue
        before = path.stat().st_size
        subprocess.run(["lipo", str(path), "-thin", arch, "-output", str(path)], check=True)
        saved += before - path.stat().st_size
    return saved


def sign(app: Path) -> None:
    subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(app)], check=True, capture_output=True)
    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app)], check=True)


def disk_image(app: Path, background: Path, icon: Path) -> Path:
    import dmgbuild

    target = BUILD / f"Digital-Unconscious-{__version__}.dmg"
    target.unlink(missing_ok=True)
    settings = {
        "files": [str(app)],
        "symlinks": {"Applications": "/Applications"},
        "badge_icon": None,
        "icon": str(icon),
        "format": "ULFO",
        "filesystem": "APFS",
        "background": str(background),
        "window_rect": ((200, 160), (640, 420)),
        "icon_size": 112,
        "text_size": 13,
        "show_status_bar": False,
        "show_tab_view": False,
        "show_toolbar": False,
        "show_pathbar": False,
        "show_sidebar": False,
        "icon_locations": {f"{NAME}.app": (170, 210), "Applications": (470, 210)},
    }
    dmgbuild.build_dmg(str(target), NAME, settings=settings)
    return target


def main() -> int:
    if sys.platform != "darwin":
        print("The disk image is built on macOS.")
        return 1
    from PySide6.QtGui import QGuiApplication

    QGuiApplication(sys.argv[:1])
    BUILD.mkdir(exist_ok=True)
    catalog, icon = make_icon()
    background = make_background(BUILD / "background.tiff")
    app = bundle(icon)
    shutil.copy(catalog, app / "Contents" / "Resources" / "Assets.car")
    saved = thin(app)
    sign(app)
    image = disk_image(app, background, icon)
    size = sum(f.stat().st_size for f in app.rglob("*") if f.is_file() and not f.is_symlink())
    print(f"{app.relative_to(ROOT)}  {size / 1e6:.0f} MB ({saved / 1e6:.0f} MB of other architectures removed)")
    print(f"{image.relative_to(ROOT)}  {image.stat().st_size / 1e6:.0f} MB")
    plistlib.loads((app / "Contents" / "Info.plist").read_bytes())  # sanity: the bundle is well-formed
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
