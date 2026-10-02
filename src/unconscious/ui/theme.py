"""The visual system: a slow morning in Nice.

You wake in a small villa, walk along the Baie des Anges, come back, and
drift off; what surfaces is the dream. So: sun-washed linen and sand by day,
the Mediterranean for the dream — azure in the morning theme, deep blue in the
evening one — and colours borrowed from the town,
faded by the sun — chair blue, sea glass, ochre façades, terracotta, lemon,
lavender. Soft Fraunces for the voice, Figtree for the rest, rounded edges,
unhurried motion. Nothing shouts.

A little of the year 2000 runs through it: gel buttons like Aqua, pebbles
that catch the light like the candy-coloured iMacs, a sheen of glass on the
evening sea. Enough to feel touched by hand and time, not enough to be a theme.

Fonts are vendored static instances (SIL OFL), registered from bytes because
macOS CoreText refuses to register some subset files by path.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QGuiApplication, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import QApplication

FONT_DIR = Path(__file__).parent / "fonts"

# The dream is always the sea; its colour follows the hour. By day it is the
# Baie des Anges in full sun: deep azure far out, clear turquoise over the
# pebbles. By night it is the same bay, deep blue under a moon.
SEA_DAY = {
    "sea_far": "#0f5aa8", "sea_mid": "#1a77c2", "sea_near": "#2694cc", "sea_shore": "#55c3cf",
    "foam": "#ffffff", "glint": "#fff3cf",
    "night_ink": "#fdfaf3", "night_body": "#eaf4fb", "night_muted": "#cfe7f6", "night_rule": "#4d93cf",
    "night_accent": "#ffe08f",  # sunlight on the water
}
SEA_NIGHT = {
    "sea_far": "#08162d", "sea_mid": "#0b1f3d", "sea_near": "#0f2a4e", "sea_shore": "#16406a",
    "foam": "#dfe8f5", "glint": "#d6e4ff",
    "night_ink": "#f3efe7", "night_body": "#d3dce8", "night_muted": "#8fa3bd", "night_rule": "#22395a",
    "night_accent": "#f2c690",  # lamplight along the Promenade
}
LIGHT = {  # matin — morning light on linen
    **SEA_DAY,
    "paper": "#f6f1e8", "paper2": "#efe7da", "card": "#fcf9f4", "ink": "#1f2d3d", "ink2": "#4b5a69",
    "muted": "#8b9198", "faint": "#bdb7ad", "rule": "#e3d9ca", "rule2": "#ece4d7", "accent": "#2f6db1",
    "accent_ink": "#24578f", "accent_soft": "#dfeaf5", "sand": "#ecdfca", "sand_shadow": "#dccbb1",
    "ok": "#4f8f6d", "warn": "#c48a2c", "error_bg": "#f7e1d8", "error_ink": "#a24a33", "none": "#d3cbbe",
    "glow": "#fff6e4", "sidebar": "#f1eadf", "gel": "#2f6db1", "frost_top": "#fffdf9", "frost_bottom": "#f1eadf",
    "card_top": "#fffdf9", "card_bottom": "#f8f2e8",
}
DARK = {  # soir — the same rooms after dinner
    **SEA_NIGHT,
    "paper": "#101b29", "paper2": "#162436", "card": "#16263a", "ink": "#f3ede4", "ink2": "#c3ccd5",
    "muted": "#8494a6", "faint": "#55647a", "rule": "#25374d", "rule2": "#1c2d42", "accent": "#7fb2e8",
    "accent_ink": "#a9cdf2", "accent_soft": "#1d3350", "sand": "#2a3442", "sand_shadow": "#1f2835",
    "ok": "#8fcaa6", "warn": "#e6b85c", "error_bg": "#3a2320", "error_ink": "#f2a58f", "none": "#3c4b5e",
    "glow": "#1a2a3e", "sidebar": "#0d1724", "gel": "#3a78c0", "frost_top": "#1f3149", "frost_bottom": "#17263a",
    "card_top": "#1b2c42", "card_bottom": "#152438",
}
# Sun-faded Riviera colours for threads.
HUES_LIGHT = ["#3c7dbf", "#dda13f", "#5bb0ab", "#d47a60", "#8f7fc2", "#97b455", "#e3c04a", "#c78aa6", "#6f9cc4", "#b08b68"]
HUES_DARK = ["#7fb2e8", "#f0c070", "#86d0ca", "#ef9f86", "#b6a8e6", "#bed482", "#f2d77c", "#e3aec6", "#9fc2e2", "#d3b08d"]


@dataclass
class Theme:
    dark: bool = False

    @property
    def c(self) -> dict[str, str]:
        return DARK if self.dark else LIGHT

    def color(self, name: str, alpha: float = 1.0) -> QColor:
        value = QColor(self.c.get(name, name))
        if alpha < 1:
            value.setAlphaF(alpha)
        return value

    def hue(self, index: int | None, night: bool = False) -> QColor:
        if index is None:
            return self.color("night_muted" if night else "none")
        palette = HUES_DARK if (self.dark or night) else HUES_LIGHT
        return QColor(palette[int(index) % 10])

    def mechanism(self, kind: str, night: bool = False) -> QColor:
        palette = HUES_DARK if (self.dark or night) else HUES_LIGHT
        return QColor({
            "collision": palette[3], "orbit": palette[0], "return": palette[4], "surge": palette[1],
            "seed": palette[5], "gap": palette[2], "steady": self.c["muted"], "fade": self.c["faint"],
        }.get(kind, palette[0]))


THEME = Theme()


def load_fonts() -> None:
    for path in sorted(FONT_DIR.glob("*.ttf")):
        QFontDatabase.addApplicationFontFromData(path.read_bytes())


ROLES: dict[str, tuple[str, int, int, bool, float]] = {
    # role: (family, pixel size, weight, italic, letter spacing %)
    "display-xl": ("Fraunces Soft", 52, 300, False, 99),
    "display-l": ("Fraunces Soft", 42, 300, False, 99),
    "display-m": ("Fraunces Soft", 30, 300, False, 99.5),
    "display-s": ("Fraunces Soft", 24, 300, False, 100),
    "title": ("Fraunces Soft", 22, 400, False, 99.5),
    "title-s": ("Fraunces Soft", 18, 400, False, 100),
    "quote": ("Fraunces Soft", 30, 300, True, 100),
    "quote-s": ("Fraunces Soft", 21, 300, True, 100),
    "reading": ("Fraunces Read", 18, 400, False, 100),
    "question": ("Fraunces Read", 15, 400, True, 100),
    "serif": ("Fraunces Read", 15, 400, False, 100),
    "caption": ("Fraunces Read", 13, 400, True, 100),
    "caption-l": ("Fraunces Read", 15, 400, True, 100),
    "num": ("Fraunces Soft", 30, 300, False, 100),
    "body": ("Figtree", 14, 400, False, 100),
    "body-strong": ("Figtree", 14, 500, False, 100),
    "small": ("Figtree", 12, 400, False, 100),
    "nav": ("Figtree", 14, 500, False, 100),
    "button": ("Figtree", 13, 500, False, 100),
    "mono": ("Fraunces Read", 13, 400, True, 100),  # captions are italic serif, not shouting mono
    "mono-s": ("Fraunces Read", 12, 400, True, 100),
    "mono-n": ("JetBrains Mono", 11, 400, False, 100),
    "wordmark": ("Fraunces Soft", 23, 300, True, 100),
}


CJK_SERIF = ["Songti SC", "STSong", "Noto Serif CJK SC", "Source Han Serif SC", "SimSun"]
CJK_SANS = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC"]
_present: dict[bool, list[str]] = {}


def _fallbacks(serif: bool) -> list[str]:
    """Only the Chinese fallbacks this machine has: asking Qt for a missing family makes
    it build its whole alias table (tens of milliseconds) and warn about it."""
    if serif not in _present:
        installed = set(QFontDatabase.families())
        wanted = CJK_SERIF if serif else CJK_SANS
        _present[serif] = [name for name in wanted if name in installed] or wanted[:1]
    return _present[serif]


def font(role: str) -> QFont:
    family, size, weight, italic, spacing = ROLES.get(role, ROLES["body"])
    f = QFont(family)
    # Chinese falls back to a serif beside Fraunces and a sans beside Figtree.
    f.setFamilies([family, *_fallbacks(family.startswith("Fraunces"))])
    f.setPixelSize(size)
    f.setWeight(QFont.Weight(weight))
    f.setItalic(italic)
    if spacing != 100:
        f.setLetterSpacing(QFont.SpacingType.PercentageSpacing, spacing)
    f.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    return f


def detect_dark(preference: str) -> bool:
    if preference == "dark":
        return True
    if preference == "light":
        return False
    return QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark


def apply(app: QApplication, dark: bool) -> None:
    THEME.dark = dark
    c = THEME.c
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: c["paper"], QPalette.ColorRole.Base: c["card"],
        QPalette.ColorRole.AlternateBase: c["paper2"], QPalette.ColorRole.Text: c["ink"],
        QPalette.ColorRole.WindowText: c["ink"], QPalette.ColorRole.ButtonText: c["ink"],
        QPalette.ColorRole.Button: c["card"], QPalette.ColorRole.Highlight: c["accent"],
        QPalette.ColorRole.HighlightedText: "#ffffff", QPalette.ColorRole.ToolTipBase: c["card"],
        QPalette.ColorRole.ToolTipText: c["ink"], QPalette.ColorRole.PlaceholderText: c["faint"],
        QPalette.ColorRole.Link: c["accent"], QPalette.ColorRole.Mid: c["rule"],
    }
    for role, value in roles.items():
        palette.setColor(role, QColor(value))
    app.setPalette(palette)
    app.setFont(font("body"))
    app.setStyleSheet(stylesheet(c))


_TICK: str | None = None


def tick_path() -> str:
    """Qt stylesheets can only draw an indicator image from a file, so the
    checkbox tick is painted once into a temporary PNG."""
    global _TICK
    if _TICK is None:
        pixmap = QPixmap(30, 30)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor("#ffffff"), 3.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawPolyline([QPointF(8, 15.5), QPointF(13, 20.5), QPointF(22.5, 10)])
        painter.end()
        path = Path(tempfile.gettempdir()) / "digital-unconscious-tick-soft.png"
        pixmap.save(str(path))
        _TICK = path.as_posix()
    return _TICK


def gel(base: str) -> str:
    """An Aqua-style gel: a glossy upper half, a crisp seam, a glow underneath."""
    color = QColor(base)
    return (
        "qlineargradient(x1:0, y1:0, x2:0, y2:1, "
        f"stop:0 {color.lighter(158).name()}, stop:0.46 {color.lighter(122).name()}, "
        f"stop:0.5 {color.name()}, stop:1 {color.lighter(132).name()})"
    )


def frost(top: str, bottom: str) -> str:
    """A frosted, slightly domed pill for secondary actions."""
    return (
        "qlineargradient(x1:0, y1:0, x2:0, y2:1, "
        f"stop:0 {QColor(top).lighter(104).name()}, stop:0.48 {top}, stop:0.52 {bottom}, stop:1 {QColor(bottom).lighter(103).name()})"
    )


GLASS = (
    "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 rgba(255,255,255,0.30), stop:0.48 rgba(255,255,255,0.12), "
    "stop:0.52 rgba(255,255,255,0.04), stop:1 rgba(255,255,255,0.14))"
)


def stylesheet(c: dict[str, str]) -> str:
    return f"""
    QWidget {{ color: {c['ink']}; }}
    QMainWindow, #Page {{ background: {c['paper']}; }}
    QScrollArea {{ border: none; background: {c['paper']}; }}
    QToolTip {{ background: {c['card']}; color: {c['ink']}; border: 1px solid {c['rule']}; border-radius: 8px; padding: 7px 10px; }}

    QLabel {{ background: transparent; }}
    QLabel[tone="muted"] {{ color: {c['muted']}; }}
    QLabel[tone="ink2"] {{ color: {c['ink2']}; }}
    QLabel[tone="accent"] {{ color: {c['accent']}; }}
    QLabel[tone="faint"] {{ color: {c['faint']}; }}
    QLabel[tone="error"] {{ color: {c['error_ink']}; background: {c['error_bg']}; border-radius: 12px; padding: 10px 14px; }}
    #Night QLabel {{ color: {c['night_ink']}; }}
    #Night QLabel[tone="muted"] {{ color: {c['night_muted']}; }}
    #Night QLabel[tone="ink2"], #Night QLabel[tone="body"] {{ color: {c['night_body']}; }}
    #Night QLabel[tone="accent"] {{ color: {c['night_accent']}; }}
    #Night QLabel[tone="faint"] {{ color: {c['night_muted']}; }}
    #Night QLabel[tone="error"] {{ color: #ffd2c4; background: rgba(244,167,127,0.18); }}

    QPushButton {{
        background: {frost(c['frost_top'], c['frost_bottom'])}; color: {c['ink']}; border: 1px solid {c['rule']}; border-radius: 16px;
        padding: 7px 18px; min-height: 18px;
    }}
    QPushButton:hover {{ border-color: {c['ink2']}; }}
    QPushButton:pressed {{ background: {c['paper2']}; }}
    QPushButton:disabled {{ color: {c['faint']}; border-color: {c['rule2']}; }}
    QPushButton[kind="ink"], QPushButton[kind="accent"] {{ background: {gel(c['gel'])}; border: 1px solid {QColor(c['gel']).darker(125).name()}; color: #ffffff; }}
    QPushButton[kind="ink"]:hover, QPushButton[kind="accent"]:hover {{ background: {gel(QColor(c['gel']).lighter(112).name())}; }}
    QPushButton[kind="ink"]:pressed, QPushButton[kind="accent"]:pressed {{ background: {gel(QColor(c['gel']).darker(112).name())}; }}
    QPushButton[kind="line"] {{ border-color: {c['rule']}; color: {c['ink2']}; }}
    QPushButton[kind="ghost"], QPushButton[kind="quiet"], QPushButton[kind="link"] {{ background: transparent; }}
    QPushButton[kind="line"]:hover {{ border-color: {c['ink2']}; color: {c['ink']}; }}
    QPushButton[kind="ghost"] {{ border-color: transparent; color: {c['muted']}; padding: 7px 10px; }}
    QPushButton[kind="quiet"], QPushButton[kind="link"] {{ min-height: 0; }}
    QPushButton[kind="ghost"]:hover {{ background: transparent; color: {c['ink']}; }}
    QPushButton[kind="on"] {{ background: {gel(c['gel'])}; border: 1px solid {QColor(c['gel']).darker(125).name()}; color: #ffffff; }}
    QPushButton[kind="chip"] {{
        border: 1px solid {c['rule']}; border-radius: 10px; padding: 3px 11px 3px 8px; background: {frost(c['frost_top'], c['paper2'])}; color: {c['ink2']}; text-align: left; min-height: 14px;
    }}
    QPushButton[kind="chip"]:hover {{ background: {c['accent_soft']}; color: {c['ink']}; }}
    QPushButton[kind="reason"] {{ border: 1px solid {c['rule']}; border-radius: 11px; padding: 3px 12px; color: {c['ink2']}; min-height: 16px; background: {frost(c['frost_top'], c['frost_bottom'])}; }}
    QPushButton[kind="reason"]:hover {{ border-color: {c['accent']}; color: {c['accent']}; background: transparent; }}
    QPushButton[kind="link"] {{ border: none; padding: 0; color: {c['ink']}; text-align: left; background: transparent; border-radius: 0; }}
    QPushButton[kind="link"]:hover {{ color: {c['accent']}; }}
    QPushButton[kind="fold"] {{ background: transparent; border: 1px solid {c['rule']}; border-radius: 12px; padding: 3px 13px; min-height: 16px; color: {c['muted']}; }}
    QPushButton[kind="fold"]:hover {{ border-color: {c['ink2']}; color: {c['ink']}; }}
    QPushButton[kind="quiet"] {{ border: none; padding: 4px 2px; color: {c['muted']}; background: transparent; border-radius: 0; }}
    QPushButton[kind="quiet"]:hover {{ color: {c['ink']}; }}
    QPushButton[size="s"] {{ padding: 5px 13px; border-radius: 13px; min-height: 16px; }}
    QPushButton[kind="tab"] {{ border: 1px solid transparent; border-radius: 14px; padding: 5px 14px; color: {c['muted']}; min-height: 18px; background: transparent; }}
    QPushButton[kind="tab"]:hover {{ background: transparent; color: {c['ink']}; }}
    QPushButton[kind="tab"][active="true"] {{ color: #ffffff; background: {gel(c['gel'])}; border: 1px solid {QColor(c['gel']).darker(125).name()}; }}
    #Night QPushButton {{ border: 1px solid rgba(247,241,230,0.42); border-radius: 16px; color: {c['night_ink']}; background: {GLASS}; padding: 7px 18px; }}
    #Night QPushButton:hover {{ border-color: {c['night_ink']}; }}
    #Night QPushButton[kind="fold"] {{ background: rgba(255,255,255,0.07); border: 1px solid rgba(247,241,230,0.30); border-radius: 12px; padding: 3px 13px; min-height: 16px; color: {c['night_muted']}; }}
    #Night QPushButton[kind="fold"]:hover {{ border-color: {c['night_ink']}; color: {c['night_ink']}; background: rgba(255,255,255,0.12); }}
    #Night QPushButton[kind="accent"] {{ background: {c['night_ink']}; border: 1px solid {c['night_ink']}; border-radius: 16px; color: {c['sea_far']}; }}
    #Night QPushButton[kind="accent"]:hover {{ background: #ffffff; }}

    #Sidebar {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {QColor(c['sidebar']).lighter(103).name()}, stop:1 {c['sidebar']}); }}
    #Sidebar QPushButton[kind="nav"] {{
        border: none; border-radius: 16px; padding: 7px 14px; text-align: left; color: {c['ink2']}; background: transparent; min-height: 20px;
    }}
    #Sidebar QPushButton[kind="nav"]:hover {{ color: {c['ink']}; background: {c['paper2']}; }}
    #Sidebar QPushButton[kind="nav"][active="true"] {{ color: #ffffff; background: {gel(c['gel'])}; border: 1px solid {QColor(c['gel']).darker(125).name()}; }}
    #Sidebar QPushButton[kind="jot"] {{ background: {gel(c['gel'])}; border: 1px solid {QColor(c['gel']).darker(125).name()}; color: #ffffff; padding: 9px 14px; }}
    #Sidebar QPushButton[kind="jot"]:hover {{ background: {gel(QColor(c['gel']).lighter(112).name())}; }}

    QLineEdit {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {c['paper2']}, stop:0.25 {c['card']}, stop:1 {c['card']}); color: {c['ink']}; border: 1px solid {c['rule']}; border-radius: 16px;
        padding: 8px 14px; selection-background-color: {c['accent']}; selection-color: #ffffff;
    }}
    QLineEdit:focus {{ border-color: {c['accent']}; }}
    QTextEdit, QPlainTextEdit, QSpinBox, QTimeEdit, QDateEdit, QComboBox {{
        background: {c['card']}; color: {c['ink']}; border: 1px solid {c['rule']}; border-radius: 12px;
        padding: 7px 10px; selection-background-color: {c['accent']}; selection-color: #ffffff;
    }}
    QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QTimeEdit:focus, QDateEdit:focus, QComboBox:focus {{
        border-color: {c['accent']};
    }}
    QComboBox::drop-down {{ border: none; width: 24px; }}
    QComboBox QAbstractItemView {{ background: {c['card']}; color: {c['ink']}; selection-background-color: {c['accent_soft']}; selection-color: {c['ink']}; border: 1px solid {c['rule']}; }}
    QCheckBox {{ spacing: 10px; }}
    QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {c['muted']}; border-radius: 5px; background: {c['card']}; }}
    QCheckBox::indicator:checked {{ background: {gel(c['gel'])}; border-color: {QColor(c['gel']).darker(125).name()}; image: url({tick_path()}); }}

    QScrollBar:vertical {{ background: transparent; width: 9px; margin: 4px 2px; }}
    QScrollBar::handle:vertical {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {QColor(c['rule']).lighter(106).name()}, stop:0.5 {c['faint']}, stop:1 {QColor(c['rule']).lighter(104).name()}); border-radius: 3px; min-height: 40px; }}
    QScrollBar::handle:vertical:hover {{ background: {c['faint']}; }}
    QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{ height: 0; background: none; }}
    QScrollBar:horizontal {{ height: 0; }}

    QMenu {{ background: {c['card']}; color: {c['ink']}; border: 1px solid {c['rule']}; border-radius: 10px; padding: 6px; }}
    QMenu::item {{ padding: 7px 18px; border-radius: 6px; }}
    QMenu::item:selected {{ background: {c['accent_soft']}; color: {c['ink']}; }}
    QMenu::item:disabled {{ color: {c['muted']}; }}
    QMenu::separator {{ height: 1px; background: {c['rule']}; margin: 5px 10px; }}
    QDialog {{ background: {c['paper']}; }}
    #Objection {{ border: none; border-left: 2px solid {c['sand']}; }}
    #Evidence {{ border: none; }}
    #Drop {{ border: 1.5px dashed {c['faint']}; border-radius: 22px; background: transparent; }}
    #Drop[over="true"] {{ border-color: {c['accent']}; background: {c['accent_soft']}; }}
    #Tile {{ border: 1px solid rgba(247,241,230,0.18); border-radius: 18px; background: rgba(247,241,230,0.05); }}
    """
