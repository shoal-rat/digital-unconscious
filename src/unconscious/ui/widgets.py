"""Painted primitives: pebble swatches, soft mood tags, the dusk sea, the horizon mark.
Rounded, unhurried, sun-faded: character comes from drawing a few things gently."""

from __future__ import annotations

import math

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFontMetrics,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
    QTextLayout,
    QTextOption,
    QTransform,
)
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QWidgetItem,
)

from unconscious.ui.i18n import t
from unconscious.ui.theme import THEME, font

INSTRUMENT_KINDS = {"", "ink", "accent", "line", "ghost", "on", "jot", "tab"}

# ------------------------------------------------------------------ text


def label(text: str, role: str = "body", tone: str | None = None, *, wrap: bool = True, selectable: bool = False) -> QLabel:
    widget = QLabel(text or "")
    widget.setTextFormat(Qt.TextFormat.PlainText)  # model output is never interpreted as markup
    widget.setFont(font(role))
    widget.setWordWrap(wrap)
    if tone:
        widget.setProperty("tone", tone)
    if selectable:
        widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        widget.setCursor(Qt.CursorShape.IBeamCursor)
    return widget


def in_night(widget: QWidget) -> bool:
    node = widget.parentWidget() if widget is not None else None
    while node is not None:
        if node.objectName() == "Night":
            return True
        node = node.parentWidget()
    return False


DAY_TONES = {None: "ink", "ink2": "ink2", "muted": "muted", "accent": "accent", "body": "ink2", "faint": "faint"}
NIGHT_TONES = {None: "night_ink", "ink2": "night_body", "muted": "night_muted", "accent": "night_accent", "body": "night_body", "faint": "night_muted"}


class Para(QWidget):
    """A wrapped paragraph with real leading. QLabel cannot set line height in
    plain text, and its rich-text mode under-reports height (clipping the last
    line), so paragraphs are laid out here with QTextLayout."""

    def __init__(self, text: str, role: str = "body", tone: str | None = None, leading: float = 1.5):
        super().__init__()
        self._text = (text or "").strip()
        self._font = font(role)
        self.tone = tone
        self.leading = leading
        self._cache: tuple[int, list, float] | None = None
        policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def text(self) -> str:
        return self._text

    def _layout(self, width: int):
        width = max(40, width)
        if self._cache and self._cache[0] == width:
            return self._cache[1], self._cache[2]
        metrics = QFontMetrics(self._font)
        step = max(metrics.height(), self._font.pixelSize() * self.leading)
        offset = (step - metrics.height()) / 2
        layouts, y = [], 0.0
        option = QTextOption()
        option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        for block in self._text.split("\n") or [""]:
            layout = QTextLayout(block, self._font)
            layout.setTextOption(option)
            layout.beginLayout()
            while True:
                line = layout.createLine()
                if not line.isValid():
                    break
                line.setLineWidth(width)
                line.setPosition(QPointF(0, y + offset))
                y += step
            layout.endLayout()
            layouts.append(layout)
        self._cache = (width, layouts, y)
        return layouts, y

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        # Box layouts offer the full column width even when a maximum width
        # will clamp the paragraph; measure at the width it will paint at.
        return int(self._layout(min(width, self.maximumWidth()))[1] + 1)

    def sizeHint(self) -> QSize:
        width = self.width() if self.width() > 40 else 480
        return QSize(min(width, 680), self.heightForWidth(width))

    def minimumSizeHint(self) -> QSize:
        return QSize(60, self.heightForWidth(max(self.width(), 60)))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.updateGeometry()

    def _color(self) -> QColor:
        table = NIGHT_TONES if in_night(self) else DAY_TONES
        return THEME.color(table.get(self.tone, table[None]))

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        painter.setPen(self._color())
        layouts, _ = self._layout(self.width())
        for layout in layouts:
            layout.draw(painter, QPointF(0, 0))


def para(text: str, role: str = "body", tone: str | None = None, line_height: int = 150) -> Para:
    return Para(text, role, tone, line_height / 100)




def eyebrow(text: str, tone: str = "muted") -> QLabel:
    """A caption: lowercase italic, said in passing rather than announced."""
    text = text or ""
    if text.isascii() and not text[:1].isdigit():
        text = text[:1].lower() + text[1:] if not text.isupper() else text.lower()
    return label(text, "caption", tone, wrap=False)


def button(text: str, kind: str = "", on_click=None, *, tip: str = "") -> QPushButton:
    """Pill buttons in a quiet sans; links and chips keep their own type."""
    instrument = kind in INSTRUMENT_KINDS
    widget = QPushButton(text)
    if kind:
        widget.setProperty("kind", kind)
    widget.setCursor(Qt.CursorShape.PointingHandCursor)
    widget.setFont(font("button" if instrument else "body"))
    if tip:
        widget.setToolTip(tip)
    if on_click:
        widget.clicked.connect(lambda _checked=False: on_click())
    return widget


def hbox(*items, spacing: int = 8, margins=(0, 0, 0, 0)) -> QHBoxLayout:
    layout = QHBoxLayout()
    layout.setSpacing(spacing)
    layout.setContentsMargins(*margins)
    for item in items:
        _add(layout, item)
    return layout


def vbox(*items, spacing: int = 8, margins=(0, 0, 0, 0)) -> QVBoxLayout:
    layout = QVBoxLayout()
    layout.setSpacing(spacing)
    layout.setContentsMargins(*margins)
    for item in items:
        _add(layout, item)
    return layout


def _add(layout, item) -> None:
    if item is None:
        return
    if item == "stretch":
        layout.addStretch(1)
    elif isinstance(item, int):
        layout.addSpacing(item)
    elif isinstance(item, QLayout):
        layout.addLayout(item)
    else:
        layout.addWidget(item)


def wrap(layout: QLayout) -> QWidget:
    widget = QWidget()
    widget.setLayout(layout)
    return widget


def clear(layout: QLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
        elif item.layout():
            clear(item.layout())


# ------------------------------------------------------------------ icons


def glyph(kind: str) -> QPainterPath:
    """Mechanism and evidence glyphs on a 16×16 grid."""
    p = QPainterPath()
    if kind == "collision":
        p.addEllipse(QPointF(5.5, 8), 4, 4)
        p.addEllipse(QPointF(10.5, 8), 4, 4)
    elif kind == "orbit":
        p.addEllipse(QPointF(8, 8), 2.0, 2.0)
        ring = QPainterPath()
        ring.addEllipse(QPointF(8, 8), 6.5, 3.4)
        p.addPath(QTransform().translate(8, 8).rotate(-20).translate(-8, -8).map(ring))
    elif kind == "return":
        p.arcMoveTo(QRectF(3, 3, 10, 10), 20)
        p.arcTo(QRectF(3, 3, 10, 10), 20, 290)
        p.moveTo(13.2, 3.0)
        p.lineTo(13.2, 6.6)
        p.lineTo(9.6, 6.6)
    elif kind == "surge":
        p.moveTo(1.5, 12)
        p.cubicTo(3.5, 12, 4, 9, 5.5, 9)
        p.cubicTo(7, 9, 7.3, 11, 8.7, 11)
        p.cubicTo(10.5, 11, 11, 4, 14.5, 3)
    elif kind == "seed":
        p.moveTo(8, 14)
        p.lineTo(8, 7.5)
        p.moveTo(8, 8.5)
        p.cubicTo(8, 5.5, 5.5, 4, 3, 4)
        p.cubicTo(3, 7, 5.5, 8.5, 8, 8.5)
        p.moveTo(8, 7)
        p.cubicTo(8, 4.5, 10, 3.4, 12.5, 3.4)
        p.cubicTo(12.5, 5.9, 10.5, 7, 8, 7)
    elif kind == "gap":
        p.moveTo(5, 3)
        p.lineTo(3, 3)
        p.lineTo(3, 13)
        p.lineTo(5, 13)
        p.moveTo(11, 3)
        p.lineTo(13, 3)
        p.lineTo(13, 13)
        p.lineTo(11, 13)
        p.moveTo(7, 8)
        p.lineTo(9, 8)
    elif kind == "steady":
        for y in (5, 8, 11):
            p.moveTo(2, y)
            p.lineTo(14, y)
    elif kind == "fade":
        p.moveTo(2, 8)
        p.lineTo(5, 8)
        p.moveTo(7, 8)
        p.lineTo(9, 8)
        p.moveTo(11, 8)
        p.lineTo(12, 8)
    elif kind == "search":
        p.addEllipse(QPointF(7, 7), 4.2, 4.2)
        p.moveTo(10.2, 10.2)
        p.lineTo(13.5, 13.5)
    elif kind == "jot":
        p.moveTo(3, 13)
        p.lineTo(4, 9.5)
        p.lineTo(11, 2.5)
        p.lineTo(13.5, 5)
        p.lineTo(6.5, 12)
        p.closeSubpath()
    elif kind == "reading":
        p.moveTo(2, 3.5)
        p.cubicTo(4.2, 2.7, 6.2, 2.9, 8, 4.3)
        p.cubicTo(9.8, 2.9, 11.8, 2.7, 14, 3.5)
        p.lineTo(14, 12.5)
        p.cubicTo(11.8, 11.7, 9.8, 11.9, 8, 13.3)
        p.cubicTo(6.2, 11.9, 4.2, 11.7, 2, 12.5)
        p.closeSubpath()
        p.moveTo(8, 4.3)
        p.lineTo(8, 13.3)
    else:
        p.addRect(QRectF(4, 3, 8, 10))
    return p


def draw_glyph(painter: QPainter, kind: str, rect: QRectF, color: QColor, width: float = 1.3) -> None:
    painter.save()
    painter.translate(rect.topLeft())
    scale = rect.width() / 16
    painter.scale(scale, scale)
    pen = QPen(color, width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    pen.setCosmetic(True)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPath(glyph(kind))
    if kind == "orbit":
        painter.setBrush(color)
        painter.drawEllipse(QPointF(8, 8), 2.0, 2.0)
    painter.restore()




def mark_pixmap(size: int, *, mono: bool = False, state: str = "observing") -> QPixmap:
    """The horizon mark: the sun on the sea's edge — above the line open air,
    below it the water, where the unconscious lives."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    p = QPainter(pixmap)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = size / 32
    if mono:
        line, sea = QColor("#000000"), QColor("#000000")
    else:
        line, sea = QColor("#1f2d3d"), QColor("#2f6db1")
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#f6f1e8"))
        p.drawRoundedRect(QRectF(0, 0, size, size), 9 * s, 9 * s)
    radius = (10.5 if not mono else 12.5) * s
    center = QPointF(16 * s, 16.5 * s)
    if state != "paused":
        lower = QPainterPath()
        lower.moveTo(center.x() - radius, center.y())
        lower.arcTo(QRectF(center.x() - radius, center.y() - radius, 2 * radius, 2 * radius), 180, 180)
        lower.closeSubpath()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(sea)
        p.drawPath(lower)
        if not mono:
            gloss = QLinearGradient(QPointF(0, center.y()), QPointF(0, center.y() + radius))
            gloss.setColorAt(0, QColor(255, 255, 255, 120))
            gloss.setColorAt(0.5, QColor(255, 255, 255, 0))
            p.setBrush(gloss)
            p.drawPath(lower)
            pen = QPen(QColor("#f6f1e8"), 1.3 * s)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            for y, inset in ((20.6, 3.2), (24.2, 6.4)):
                wave = QPainterPath()
                x0, x1 = center.x() - radius + inset * s, center.x() + radius - inset * s
                wave.moveTo(x0, y * s)
                steps = 12
                for i in range(1, steps + 1):
                    x = x0 + (x1 - x0) * i / steps
                    wave.lineTo(x, y * s + math.sin(i / steps * math.pi * 3) * 0.7 * s)
                p.drawPath(wave)
    pen = QPen(line, 1.5 * s)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(center, radius, radius)
    p.drawLine(QPointF(center.x() - radius - 3 * s, center.y()), QPointF(center.x() + radius + 3 * s, center.y()))
    if state == "dreaming":
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(line if mono else QColor("#f4a77f"))
        p.drawEllipse(QPointF(25.5 * s, 6.5 * s), 2.4 * s, 2.4 * s)
    p.end()
    return pixmap


def app_icon() -> QIcon:
    icon = QIcon()
    for size in (16, 32, 64, 128, 256, 512):
        icon.addPixmap(mark_pixmap(size))
    return icon


def swatch_icon(hue: int | None, size: int = 9) -> QIcon:
    pixmap = QPixmap(size * 2, size * 2)
    pixmap.setDevicePixelRatio(2)
    pixmap.fill(Qt.GlobalColor.transparent)
    p = QPainter(pixmap)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(THEME.hue(hue))
    p.drawEllipse(QRectF(0.5, 1, size - 1, size - 2))
    p.end()
    return QIcon(pixmap)


# ------------------------------------------------------------------ small painted widgets


def orb(p: QPainter, center: QPointF, rx: float, ry: float, color: QColor) -> None:
    """A small glossy bead: lit from the upper left, with a speck of light."""
    body = QRadialGradient(QPointF(center.x() - rx * 0.35, center.y() - ry * 0.4), max(rx, ry) * 1.6)
    body.setColorAt(0, color.lighter(155))
    body.setColorAt(0.5, color)
    body.setColorAt(1, color.darker(130))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(body)
    p.drawEllipse(center, rx, ry)
    p.setBrush(QColor(255, 255, 255, 200))
    p.drawEllipse(QPointF(center.x() - rx * 0.36, center.y() - ry * 0.42), rx * 0.28, ry * 0.2)


class Stamp(QWidget):
    """A mechanism, said softly: a small coloured pebble and an italic word."""

    def __init__(self, kind: str, small: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.kind = kind
        self.text = t(f"mech.{kind}").lower() if t(f"mech.{kind}").isascii() else t(f"mech.{kind}")
        self._font = font("mono-s" if small else "mono")
        metrics = QFontMetrics(self._font)
        self.setFixedSize(metrics.horizontalAdvance(self.text) + 18, 18 if small else 20)
        self.setToolTip(t(f"mech.{kind}"))

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        night = in_night(self)
        color = THEME.mechanism(self.kind, night)
        orb(p, QPointF(5, self.height() / 2), 5.2, 4.4, color)
        p.setFont(self._font)
        p.setPen(THEME.color("night_body" if night else "ink2"))
        p.drawText(QRectF(15, 0, self.width(), self.height()), Qt.AlignmentFlag.AlignVCenter, self.text)


class Score(QWidget):
    """The critic's score, quietly: a number and a little tide line filled to it."""

    def __init__(self, score: float, parent: QWidget | None = None):
        super().__init__(parent)
        self.score = max(0.0, min(100.0, float(score or 0)))
        self.setFixedSize(70, 22)
        self.setToolTip(t("spark.score"))

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setFont(font("caption"))
        p.setPen(THEME.color("muted"))
        p.drawText(QRectF(0, 0, 26, self.height()), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{self.score:.0f}")
        x0, x1, mid = 33.0, float(self.width() - 2), self.height() / 2 + 1
        def wave(until: float) -> QPainterPath:
            path = QPainterPath()
            path.moveTo(x0, mid)
            x = x0
            while x <= until:
                path.lineTo(x, mid + math.sin((x - x0) / 3.2) * 2.0)
                x += 1.0
            return path
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(THEME.color("rule"), 1.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawPath(wave(x1))
        p.setPen(QPen(THEME.color("accent"), 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawPath(wave(x0 + (x1 - x0) * self.score / 100))


ScoreRing = Score


class Swatch(QWidget):
    def __init__(self, hue: int | None, size: int = 10, parent: QWidget | None = None):
        super().__init__(parent)
        self.hue = hue
        self.setFixedSize(size, size)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(THEME.hue(self.hue, in_night(self)))
        p.drawEllipse(QRectF(0, self.height() * 0.1, self.width(), self.height() * 0.8))


class Rule(QWidget):
    def __init__(self, color: str = "rule", weight: int = 1, parent: QWidget | None = None):
        super().__init__(parent)
        self.color_name = color
        self.setFixedHeight(weight)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, _event) -> None:
        name = {"ink": "rule"}.get(self.color_name, self.color_name)
        QPainter(self).fillRect(self.rect(), THEME.color(name))


class MiniBar(QWidget):
    def __init__(self, fraction: float, hue: int | None, parent: QWidget | None = None):
        super().__init__(parent)
        self.fraction = max(0.0, min(1.0, fraction))
        self.hue = hue
        self.setFixedHeight(5)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(THEME.color("rule2"))
        p.drawRoundedRect(QRectF(self.rect()), 2.5, 2.5)
        color = THEME.hue(self.hue)
        fill = QLinearGradient(QPointF(0, 0), QPointF(0, self.height()))
        fill.setColorAt(0, color.lighter(150))
        fill.setColorAt(0.5, color)
        fill.setColorAt(1, color.darker(115))
        p.setBrush(fill)
        p.drawRoundedRect(QRectF(0, 0, max(5.0, self.width() * self.fraction), self.height()), 2.5, 2.5)


def chip(text: str, hue: int | None, on_click=None) -> QPushButton:
    widget = button(text, "chip", on_click)
    widget.setIcon(swatch_icon(hue))
    widget.setIconSize(QSize(9, 9))
    widget.setFont(font("small"))
    return widget


class WaveMark(QWidget):
    """A short hand-drawn wave, used where other designs would draw a rule."""

    def __init__(self, width: int = 34, color: str = "accent", parent: QWidget | None = None):
        super().__init__(parent)
        self.color_name = color
        self.setFixedSize(width, 10)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(THEME.color(self.color_name), 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        path = QPainterPath()
        path.moveTo(1, 5)
        x = 1.0
        while x < self.width() - 1:
            path.lineTo(x, 5 + math.sin(x / self.width() * math.pi * 3) * 2.6)
            x += 0.8
        p.drawPath(path)


class SectionHead(QWidget):
    """A small wave, a soft serif title, an italic aside."""

    def __init__(self, title: str, count: int | None = None, note: str = "", index: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        layout = vbox(spacing=6)
        layout.addWidget(WaveMark())
        row = hbox(spacing=10)
        row.addWidget(label(title, "display-s", wrap=False), 0, Qt.AlignmentFlag.AlignBaseline)
        if count is not None:
            row.addWidget(label(str(count), "caption-l", "muted", wrap=False), 0, Qt.AlignmentFlag.AlignBaseline)
        row.addStretch(1)
        if note:
            row.addWidget(label(note, "caption", "muted", wrap=False), 0, Qt.AlignmentFlag.AlignBaseline)
        layout.addLayout(row)
        self.setLayout(layout)


class Clickable(QFrame):
    clicked = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


def sea_gradient(rect: QRectF) -> QLinearGradient:
    """Far water at the top, the shallows at the bottom, as seen from the Promenade."""
    gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
    gradient.setColorAt(0.0, THEME.color("sea_far"))
    gradient.setColorAt(0.5, THEME.color("sea_mid"))
    gradient.setColorAt(0.84, THEME.color("sea_near"))
    gradient.setColorAt(1.0, THEME.color("sea_shore"))
    return gradient


class NightPanel(QFrame):
    """The dream: the Baie des Anges after sunset. A soft wash from dusk sky to
    deep water and a low warm glow where the sun went down. With a shore, the
    sea ends in a slow wave over wet sand, and the day's pebbles lie below it."""

    def __init__(self, parent: QWidget | None = None, padding: tuple[int, int, int, int] = (52, 46, 52, 40)):
        super().__init__(parent)
        self.setObjectName("Night")
        self.body = vbox(spacing=0, margins=padding)
        self.setLayout(self.body)
        self.shore: QWidget | None = None
        self.fish = 0
        self._phase = 9.0  # the shoal is already crossing when the panel opens
        self._timer: QTimer | None = None

    def set_fish(self, count: int) -> None:
        """A small shoal crossing the shallows: one fish for each idea that surfaced."""
        self.fish = max(0, min(int(count), 7))

    def set_shore(self, widget: QWidget) -> None:
        self.body.addSpacing(70)
        self.body.addWidget(widget)
        self.shore = widget
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._drift)
        self._timer.start(80)

    def _drift(self) -> None:
        if self.isVisible() and self.shore is not None:
            self._phase += 0.03
            edge = self._edge()
            self.update(0, int(edge - 130), self.width(), 160)

    def _edge(self) -> float:
        return float(self.shore.geometry().top() - 18) if self.shore is not None else float(self.height())

    def _wave_y(self, x: float, base: float, amplitude: float = 5.0, speed: float = 1.0) -> float:
        phase = self._phase * speed
        return base + math.sin(x / 92 + phase) * amplitude + math.sin(x / 37 + phase * 1.6) * amplitude * 0.32

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())
        frame = QPainterPath()
        frame.addRoundedRect(rect, 26, 26)
        if self.shore is None:
            p.fillPath(frame, sea_gradient(rect))
            self._glow(p, frame, rect)
            return
        edge = self._edge()
        p.fillPath(frame, THEME.color("sand"))
        rng = __import__("random").Random(11)
        speck = QColor(THEME.c["sand_shadow"])
        for _ in range(int(rect.width() * (rect.height() - edge) / 260)):
            speck.setAlphaF(rng.uniform(0.25, 0.7))
            p.fillRect(QRectF(rng.uniform(0, rect.width()), rng.uniform(edge, rect.height()), 1.2, 1.2), speck)
        wet = QPainterPath()
        wet.moveTo(0, edge)
        x = 0.0
        while x <= rect.width():
            wet.lineTo(x, self._wave_y(x, edge + 14, 6, 0.7))
            x += 4
        wet.lineTo(rect.width(), edge - 20)
        wet.lineTo(0, edge - 20)
        wet.closeSubpath()
        damp = QColor(THEME.c["sand_shadow"])
        damp.setAlphaF(0.55)
        p.fillPath(wet.intersected(frame), damp)
        sea = QPainterPath()
        sea.moveTo(0, 0)
        sea.lineTo(0, edge)
        x = 0.0
        while x <= rect.width():
            sea.lineTo(x, self._wave_y(x, edge))
            x += 4
        sea.lineTo(rect.width(), 0)
        sea.closeSubpath()
        sea = sea.intersected(frame)
        p.fillPath(sea, sea_gradient(QRectF(0, 0, rect.width(), edge)))
        self._glow(p, sea, rect)
        foam = QColor(THEME.c["foam"])
        for offset, alpha, width in ((0, 0.75, 1.6), (-14, 0.14, 1.1), (-30, 0.08, 1.0)):
            foam.setAlphaF(alpha)
            p.setPen(QPen(foam, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            line = QPainterPath()
            x = 8.0
            line.moveTo(x, self._wave_y(x, edge + offset, 5 if offset == 0 else 3.4, 1.0 if offset == 0 else 0.6))
            while x <= rect.width() - 8:
                line.lineTo(x, self._wave_y(x, edge + offset, 5 if offset == 0 else 3.4, 1.0 if offset == 0 else 0.6))
                x += 4
            p.drawPath(line)
        if self.fish:
            self._shoal(p, rect, edge)

    def _shoal(self, p: QPainter, rect: QRectF, edge: float) -> None:
        span = rect.width() + 160
        lead = (self._phase * 38) % span - 80
        colour = QColor(THEME.c["foam"])
        colour.setAlphaF(0.34 if not THEME.dark else 0.3)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(colour)
        for i in range(self.fish):
            x = lead - (i % 3) * 17 - (i // 3) * 9
            y = edge - 30 + ((i % 3) - 1) * 7 + math.sin(self._phase * 1.6 + i) * 2.4
            p.save()
            p.translate(x, y)
            p.rotate(math.sin(self._phase * 1.6 + i) * 4)
            p.drawEllipse(QPointF(0, 0), 5.2, 1.9)
            tail = QPainterPath()
            tail.moveTo(-4.5, 0)
            tail.lineTo(-8.5, -2.6)
            tail.lineTo(-8.5, 2.6)
            tail.closeSubpath()
            p.drawPath(tail)
            p.restore()

    def _glow(self, p: QPainter, shape: QPainterPath, rect: QRectF) -> None:
        # The light source: the sun in the morning theme, the moon in the evening one.
        glow = QRadialGradient(QPointF(rect.width() * 0.86, rect.height() * 0.02), rect.width() * 0.46)
        light = THEME.color("glint")
        light.setAlphaF(0.34 if not THEME.dark else 0.16)
        glow.setColorAt(0, light)
        light.setAlphaF(0.0)
        glow.setColorAt(1, light)
        p.fillPath(shape, glow)
        rng = __import__("random").Random(5)
        p.setPen(Qt.PenStyle.NoPen)
        if THEME.dark:
            for _ in range(9):  # a few faint stars over the far water
                star = QColor(THEME.c["foam"])
                star.setAlphaF(rng.uniform(0.25, 0.6))
                p.setBrush(star)
                p.drawEllipse(QPointF(rng.uniform(0.05, 0.97) * rect.width(), rng.uniform(0.04, 0.3) * rect.height()), 0.9, 0.9)
        elif self.shore is not None:
            edge = self._edge()
            for i in range(26):  # sun glitter on the shallows, twinkling slowly
                x = rng.uniform(0.04, 0.96) * rect.width()
                y = edge - rng.uniform(26, 120)
                twinkle = 0.5 + 0.5 * math.sin(self._phase * 2.2 + i * 1.7)
                spark = QColor(THEME.c["glint"])
                spark.setAlphaF(0.18 + 0.42 * twinkle)
                p.setBrush(spark)
                width = rng.uniform(3, 9)
                p.drawRoundedRect(QRectF(x, y, width, 1.4), 0.7, 0.7)
        # a sheet of glass over the water: one broad, curved reflection
        sheen = QPainterPath()
        sheen.addEllipse(QPointF(rect.width() * 0.30, -rect.height() * 0.62), rect.width() * 0.95, rect.height() * 0.95)
        reflect = QLinearGradient(QPointF(0, 0), QPointF(0, rect.height() * 0.34))
        reflect.setColorAt(0, QColor(255, 255, 255, 34))
        reflect.setColorAt(1, QColor(255, 255, 255, 4))
        p.fillPath(sheen.intersected(shape), reflect)
        edge_line = QPainterPath()
        edge_line.addRoundedRect(rect.adjusted(1, 1, -1, -1), 25, 25)
        p.setPen(QPen(QColor(255, 255, 255, 40), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.save()
        p.setClipRect(QRectF(0, 0, rect.width(), 40))
        p.drawPath(edge_line)
        p.restore()


class TideLine(QWidget):
    """The sidebar's pulse: a small wave that moves with the watcher's state —
    rolling while it watches, flat at anchor, deeper while it dives."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.state = "never"
        self._phase = 0.0
        self.setFixedHeight(14)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(90)

    def set_state(self, state: str) -> None:
        self.state = state
        self.update()

    def _tick(self) -> None:
        if self.isVisible() and self.state in {"observing", "dreaming", "idle"}:
            self._phase += 0.12 if self.state == "dreaming" else 0.06
            self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        amplitude = {"observing": 2.6, "dreaming": 4.2, "idle": 1.0}.get(self.state, 0.0)
        colour = THEME.color("accent" if self.state in {"observing", "dreaming"} else "faint")
        pen = QPen(colour, 1.5, Qt.PenStyle.SolidLine if amplitude else Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        path = QPainterPath()
        mid = self.height() / 2
        x = 1.0
        path.moveTo(x, mid)
        while x < self.width() - 1:
            path.lineTo(x, mid + math.sin(x / 9 + self._phase) * amplitude)
            x += 1.5
        p.drawPath(path)


class GridSurface(QWidget):
    """The page: sun-washed linen with a soft morning light from the window."""

    _grain: QPixmap | None = None
    _grain_dark: bool | None = None

    def _texture(self) -> QPixmap:
        if GridSurface._grain is None or GridSurface._grain_dark != THEME.dark:
            import random

            rng = random.Random(3)
            tile = QPixmap(160, 160)
            tile.fill(THEME.color("paper"))
            p = QPainter(tile)
            for _ in range(900):
                shade = THEME.color("ink", rng.uniform(0.012, 0.035))
                p.fillRect(QRectF(rng.uniform(0, 160), rng.uniform(0, 160), 1, 1), shade)
            p.end()
            GridSurface._grain, GridSurface._grain_dark = tile, THEME.dark
        return GridSurface._grain

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.drawTiledPixmap(event.rect(), self._texture(), event.rect().topLeft())
        light = QRadialGradient(QPointF(self.width() * 0.9, -80), max(self.width(), 900) * 0.75)
        glow = THEME.color("glow")
        glow.setAlphaF(0.85 if not THEME.dark else 0.6)
        light.setColorAt(0, glow)
        glow.setAlphaF(0.0)
        light.setColorAt(1, glow)
        p.fillRect(event.rect(), light)


class Steps(QWidget):
    """Progress through named steps; the current one breathes, slowly."""

    def __init__(self, steps, current: str | None, done: bool = False, night: bool = True, parent=None):
        super().__init__(parent)
        self.steps = list(steps)
        self.index = self.steps.index(current) if current in self.steps else (len(self.steps) if done else -1)
        self.night = night
        self._phase = 0.0
        self.setFixedHeight(len(self.steps) * 28)
        self.setMinimumWidth(320)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(70)

    def set_current(self, current: str | None) -> None:
        if current in self.steps:
            self.index = self.steps.index(current)
            self.update()

    def _tick(self) -> None:
        self._phase = (self._phase + 0.05) % (2 * math.pi)
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        ink = THEME.color("night_ink" if self.night else "ink")
        muted = THEME.color("night_muted" if self.night else "muted")
        accent = THEME.color("night_accent" if self.night else "accent")
        p.setFont(font("caption-l"))
        for i, step in enumerate(self.steps):
            y = i * 28 + 14
            if i < self.index:
                color = ink
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(ink)
                p.drawEllipse(QPointF(6, y), 4, 3.4)
            elif i == self.index:
                color = accent
                breath = 0.5 + 0.5 * math.sin(self._phase)
                halo = QColor(accent)
                halo.setAlphaF(0.25 * breath)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(halo)
                p.drawEllipse(QPointF(6, y), 6 + 4 * breath, 5 + 3.4 * breath)
                p.setBrush(accent)
                p.drawEllipse(QPointF(6, y), 4, 3.4)
            else:
                color = muted
                p.setPen(QPen(muted, 1))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(QPointF(6, y), 3.6, 3)
            p.setPen(color)
            p.drawText(QRect(24, y - 12, self.width() - 24, 24), Qt.AlignmentFlag.AlignVCenter, t(f"steps.{step}"))


# ------------------------------------------------------------------ layouts


class FlowLayout(QLayout):
    """Left-to-right wrapping layout (chips, stamps, buttons)."""

    def __init__(self, parent=None, spacing: int = 6):
        super().__init__(parent)
        self._items: list = []
        self._spacing = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item) -> None:
        self._items.append(item)

    def addWidget(self, widget) -> None:  # noqa: N802 - Qt naming
        self.addChildWidget(widget)
        self.addItem(QWidgetItem(widget))

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._layout(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._layout(rect, apply=True)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        return size

    def _layout(self, rect: QRect, apply: bool) -> int:
        x, y, line = rect.x(), rect.y(), 0
        for item in self._items:
            hint = item.sizeHint()
            if x + hint.width() > rect.right() + 1 and line > 0:
                x = rect.x()
                y += line + self._spacing
                line = 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self._spacing
            line = max(line, hint.height())
        return y + line - rect.y()


class CardGrid(QWidget):
    """Responsive columns of cards; re-flows only when the column count changes."""

    def __init__(self, cards: list[QWidget], min_width: int = 310, spacing: int = 20, parent=None):
        super().__init__(parent)
        self.cards = cards
        self.min_width = min_width
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(spacing)
        self.grid.setVerticalSpacing(spacing)
        self.spacing = spacing
        self.columns = 0
        self._place(3)

    def _place(self, columns: int) -> None:
        if columns == self.columns:
            return
        self.columns = columns
        for card in self.cards:
            self.grid.removeWidget(card)
        for col in range(6):
            self.grid.setColumnStretch(col, 1 if col < columns else 0)
        for index, card in enumerate(self.cards):
            self.grid.addWidget(card, index // columns, index % columns, Qt.AlignmentFlag.AlignTop)

    def resizeEvent(self, event) -> None:
        width = event.size().width()
        self._place(max(1, min(3, (width + self.spacing) // (self.min_width + self.spacing))))
        super().resizeEvent(event)
