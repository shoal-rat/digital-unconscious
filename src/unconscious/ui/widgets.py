"""Painted primitives: pebble swatches, soft mood tags, the dusk sea, the horizon mark.
Rounded, unhurried, sun-faded: character comes from drawing a few things gently."""

from __future__ import annotations

import math

from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFontMetrics,
    QIcon,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QRadialGradient,
    QTextLayout,
    QTextOption,
    QTransform,
)
from PySide6.QtGui import Qt as GuiQt
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


class WrapLabel(QLabel):
    """A QLabel that measures its wrapped height at the width it will really have.

    A vertical box layout asks every child for its height at the full column width, even
    when the child carries a maximum width; QLabel then reports fewer lines than it will
    paint, the box comes out short, and the layout squeezes the difference out of whatever
    is tallest (a dream's reflection, cut off halfway). Clamping here keeps the sums true."""

    def heightForWidth(self, width: int) -> int:
        return super().heightForWidth(min(width, self.maximumWidth()) if width >= 0 else width)


class TitleLabel(WrapLabel):
    """A wrapping line of text that opens something when clicked; underlined on hover."""

    clicked = Signal()

    def __init__(self, text: str, role: str):
        super().__init__(text)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setWordWrap(True)
        self.setFont(font(role))
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def enterEvent(self, _event) -> None:
        f = self.font()
        f.setUnderline(True)
        self.setFont(f)

    def leaveEvent(self, _event) -> None:
        f = self.font()
        f.setUnderline(False)
        self.setFont(f)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()


def plain_tip(text: str) -> str:
    """A tooltip shown as plain text: Qt guesses rich text from anything that looks like a tag, and
    window titles, model names and jots are never markup."""
    return GuiQt.convertFromPlainText(text, GuiQt.WhiteSpaceMode.WhiteSpaceNormal) if text else ""


def label(text: str, role: str = "body", tone: str | None = None, *, wrap: bool = True, selectable: bool = False) -> QLabel:
    widget = WrapLabel(text or "")
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

    overflowChanged = Signal(bool)  # noqa: N815 - Qt naming: the text no longer fits its fold, or fits again

    def __init__(self, text: str, role: str = "body", tone: str | None = None, leading: float = 1.5):
        super().__init__()
        self._text = (text or "").strip()
        self._font = font(role)
        self.tone = tone
        self.leading = leading
        self._cache: tuple[int, list, float, list[float]] | None = None
        self.fold_lines: int | None = None  # set by Fold: show only this many lines while folded
        self.folded = True
        self._overflowing: bool | None = None  # unknown until the first real width, which always tells the Fold
        policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def text(self) -> str:
        return self._text

    def _layout(self, width: int):
        width = max(40, width)
        if self._cache and self._cache[0] == width:
            return self._cache[1], self._cache[2]
        bottoms: list[float] = []
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
                bottoms.append(y)
            layout.endLayout()
            layouts.append(layout)
        self._cache = (width, layouts, y, bottoms)
        return layouts, y

    def exceeds(self, width: int) -> bool:
        """Does the text run past its fold at this width? One extra line is not worth folding."""
        if not self.fold_lines:
            return False
        self._layout(min(width, self.maximumWidth()))
        return len(self._cache[3]) > self.fold_lines + 1

    def _shown_height(self, width: int) -> float:
        width = min(width, self.maximumWidth())
        _, full = self._layout(width)
        if self.folded and self.exceeds(width):
            return self._cache[3][self.fold_lines - 1]
        return full

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        # Box layouts offer the full column width even when a maximum width
        # will clamp the paragraph; measure at the width it will paint at.
        return int(self._shown_height(width) + 1)

    def sizeHint(self) -> QSize:
        width = self.width() if self.width() > 40 else 480
        return QSize(min(width, 680), self.heightForWidth(width))

    def minimumSizeHint(self) -> QSize:
        return QSize(60, self.heightForWidth(max(self.width(), 60)))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.updateGeometry()
        overflowing = self.exceeds(self.width())
        if overflowing != self._overflowing:
            self._overflowing = overflowing
            self.overflowChanged.emit(overflowing)

    def _color(self) -> QColor:
        table = NIGHT_TONES if in_night(self) else DAY_TONES
        return THEME.color(table.get(self.tone, table[None]))

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        painter.setPen(self._color())
        layouts, _ = self._layout(self.width())
        if not (self.folded and self.exceeds(self.width())):
            for layout in layouts:
                layout.draw(painter, QPointF(0, 0))
            return
        # Folded: the shown lines, the last one fading into the page, so it reads as "there is more".
        bottoms = self._cache[3]
        shown = bottoms[self.fold_lines - 1]
        fade_from = bottoms[self.fold_lines - 2] if self.fold_lines > 1 else 0.0
        ratio = self.devicePixelRatioF()
        image = QImage(int(self.width() * ratio) + 1, int(shown * ratio) + 1, QImage.Format.Format_ARGB32_Premultiplied)
        image.setDevicePixelRatio(ratio)
        image.fill(Qt.GlobalColor.transparent)
        inner = QPainter(image)
        inner.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        inner.setPen(painter.pen())
        for layout in layouts:
            layout.draw(inner, QPointF(0, 0))
        fade = QLinearGradient(QPointF(0, fade_from), QPointF(0, shown))
        fade.setColorAt(0, QColor(0, 0, 0, 255))
        fade.setColorAt(1, QColor(0, 0, 0, 30))
        inner.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
        inner.fillRect(QRectF(0, fade_from, self.width(), shown - fade_from + 1), fade)
        inner.end()
        painter.drawImage(QPointF(0, 0), image)


# Folds a person has opened stay open while the app runs, through page rebuilds.
UNFOLDED: set[str] = set()


class Fold(QWidget):
    """Long text shown in its first lines, the last one fading, with a translucent pill that
    unfolds it in place and folds it back. The pill appears only when the text is longer than
    the fold at the width it has, so nothing is ever cut without a way to read it all."""

    def __init__(self, text: Para, lines: int = 6, key: str | None = None):
        super().__init__()
        self.text, self.key = text, key
        text.fold_lines = max(1, lines)
        text.folded = not (key and key in UNFOLDED)
        self.toggle = button("", "fold", self._toggle)
        self.toggle.setFont(font("caption-l"))
        row = hbox(self.toggle, "stretch")
        box = vbox(text, row, spacing=10)
        self.setLayout(box)
        text.overflowChanged.connect(lambda _overflowing: self._sync())
        self._sync()

    def _sync(self) -> None:
        over = self.text.exceeds(self.text.width() if self.text.width() > 40 else self.text.maximumWidth())
        self.text._overflowing = over  # what the pill shows; a later width that changes it emits again
        self.toggle.setVisible(over)
        self.toggle.setText(t("fold.more") + "  ↓" if self.text.folded else t("fold.less") + "  ↑")

    def _toggle(self) -> None:
        self.text.folded = not self.text.folded
        if self.key:
            (UNFOLDED.discard if self.text.folded else UNFOLDED.add)(self.key)
        self.text.updateGeometry()
        self.text.update()
        self._sync()


class FoldList(QWidget):
    """A list that shows its first ``keep`` rows and a translucent pill for the rest."""

    def __init__(self, rows: list, keep: int, key: str | None = None, spacing: int = 0):
        super().__init__()
        self.rows, self.keep, self.key = rows, keep, key
        box = vbox(spacing=spacing)
        for row in rows:
            _add(box, row)
        self.toggle = button("", "fold", self._toggle)
        self.toggle.setFont(font("caption-l"))
        box.addSpacing(8 if len(rows) > keep else 0)
        box.addLayout(hbox(self.toggle, "stretch"))
        self.setLayout(box)
        self.toggle.setVisible(len(rows) > keep)  # after setLayout: shown without a parent it would be a window
        self.open = bool(key and key in UNFOLDED)
        self._apply()

    def _apply(self) -> None:
        for index, row in enumerate(self.rows):
            target = row if isinstance(row, QWidget) else None
            if target is not None:
                target.setVisible(self.open or index < self.keep)
        hidden = len(self.rows) - self.keep
        self.toggle.setText(t("fold.fewer") + "  ↑" if self.open else t("fold.all", n=len(self.rows)) + "  ↓")
        self.toggle.setToolTip("" if self.open else t("fold.hidden", n=hidden))

    def _toggle(self) -> None:
        self.open = not self.open
        if self.key:
            (UNFOLDED.add if self.open else UNFOLDED.discard)(self.key)
        self._apply()


class Unfold(QWidget):
    """Something kept folded away under a translucent pill, opened in place (an earlier dive of the day)."""

    def __init__(self, body: QWidget, more: str, less: str, key: str | None = None):
        super().__init__()
        self.body, self.more, self.less, self.key = body, more, less, key
        self.toggle = button("", "fold", self._toggle)
        self.toggle.setFont(font("caption-l"))
        box = vbox(hbox(self.toggle, "stretch"), body, spacing=14)
        self.setLayout(box)
        self.open = bool(key and key in UNFOLDED)
        self._apply()

    def _apply(self) -> None:
        self.body.setVisible(self.open)
        self.toggle.setText(self.less + "  ↑" if self.open else self.more + "  ↓")

    def _toggle(self) -> None:
        self.open = not self.open
        if self.key:
            (UNFOLDED.add if self.open else UNFOLDED.discard)(self.key)
        self._apply()


class ElidedLabel(QLabel):
    """One line that ends in an ellipsis at whatever width it gets, with the whole text in a
    tooltip: never wider than its place, never silently cut."""

    def __init__(self, text: str, role: str = "body", tone: str | None = None, tip: str | None = None,
                 longest: int = 420):
        super().__init__()
        self.full = text or ""
        self.longest = longest
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setFont(font(role))
        if tone:
            self.setProperty("tone", tone)
        self.setToolTip(plain_tip(tip if tip is not None else self.full))
        self.setText(self.full)

    def sizeHint(self) -> QSize:
        m = self.contentsMargins()  # a style's padding lands here
        width = min(self.fontMetrics().horizontalAdvance(self.full) + 4, self.longest) + m.left() + m.right()
        return QSize(width, super().sizeHint().height())

    def minimumSizeHint(self) -> QSize:
        m = self.contentsMargins()
        return QSize(min(self.sizeHint().width(), 40 + m.left() + m.right()), super().minimumSizeHint().height())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        room = max(10, self.contentsRect().width())
        self.setText(self.fontMetrics().elidedText(self.full, Qt.TextElideMode.ElideRight, room))


def capped(widget: QWidget, width: int) -> QHBoxLayout:
    """A widget no wider than ``width``, added to a vertical layout through a row so the layout
    measures it at that width (see WrapLabel). Use it for holders that have a layout, whose
    height Qt asks of the layout directly and never of the widget."""
    widget.setMaximumWidth(width)
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(0)
    row.addWidget(widget, 1)
    row.addStretch(0)
    return row


def para(text: str, role: str = "body", tone: str | None = None, line_height: int = 150) -> Para:
    return Para(text, role, tone, line_height / 100)




def eyebrow(text: str, tone: str = "muted", *, wrap: bool = False) -> QLabel:
    """A caption: lowercase italic, said in passing rather than announced."""
    text = text or ""
    if text.isascii() and not text[:1].isdigit():
        text = text[:1].lower() + text[1:] if not text.isupper() else text.lower()
    return label(text, "caption", tone, wrap=wrap)


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


def mac_icon_image(size: int = 1024) -> QImage:
    """The app icon on Apple's grid: the horizon mark on a linen tile of 824 on a 1024 canvas,
    with a soft shadow and a little gloss. The Dock draws it the size of every other app."""

    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    p = QPainter(image)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    k = size / 1024
    tile = QRectF(100 * k, 92 * k, 824 * k, 824 * k)  # Apple's icon grid: a 824 pt tile on a 1024 canvas
    for i in range(18, 0, -1):  # a soft shadow under the tile
        shade = QPainterPath()
        grow = i * 1.6 * k
        shade.addRoundedRect(tile.adjusted(-grow, -grow + 10 * k, grow, grow + 14 * k), 185 * k + grow, 185 * k + grow)
        p.fillPath(shade, QColor(25, 40, 60, 6))
    body = QPainterPath()
    body.addRoundedRect(tile, 185 * k, 185 * k)
    linen = QLinearGradient(tile.topLeft(), tile.bottomLeft())
    linen.setColorAt(0, QColor("#fbf7f0"))
    linen.setColorAt(1, QColor("#efe4d2"))
    p.fillPath(body, linen)
    glow = QRadialGradient(QPointF(tile.center().x(), tile.top() + 250 * k), 420 * k)
    glow.setColorAt(0, QColor(255, 214, 160, 90))  # the morning over the bay
    glow.setColorAt(1, QColor(255, 214, 160, 0))
    p.fillPath(body, glow)

    centre, radius = QPointF(512 * k, 520 * k), 250 * k
    sea = QPainterPath()
    sea.moveTo(centre.x() - radius, centre.y())
    sea.arcTo(QRectF(centre.x() - radius, centre.y() - radius, 2 * radius, 2 * radius), 180, 180)
    sea.closeSubpath()
    water = QLinearGradient(QPointF(0, centre.y()), QPointF(0, centre.y() + radius))
    water.setColorAt(0, QColor("#5a92cf"))
    water.setColorAt(0.5, QColor("#2f6db1"))
    water.setColorAt(1, QColor("#1d4f86"))
    p.fillPath(sea, water)
    gloss = QLinearGradient(QPointF(0, centre.y()), QPointF(0, centre.y() + radius * 0.6))
    gloss.setColorAt(0, QColor(255, 255, 255, 140))
    gloss.setColorAt(1, QColor(255, 255, 255, 0))
    p.fillPath(sea, gloss)
    foam = QPen(QColor("#f6f1e8"), 20 * k, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
    p.setPen(foam)
    for offset, half in ((95, 150), (165, 105)):
        wave = QPainterPath()
        y = centre.y() + offset * k
        x = centre.x() - half * k
        wave.moveTo(x, y)
        steps = 4
        width = 2 * half * k / steps
        for _ in range(steps):
            wave.cubicTo(QPointF(x + width * 0.3, y - 16 * k), QPointF(x + width * 0.7, y + 16 * k), QPointF(x + width, y))
            x += width
        p.drawPath(wave)
    ink = QPen(QColor("#1f2d3d"), 30 * k, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
    p.setPen(ink)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(centre, radius, radius)
    p.drawLine(QPointF(centre.x() - radius - 75 * k, centre.y()), QPointF(centre.x() + radius + 75 * k, centre.y()))
    shine = QPainterPath()
    shine.addRoundedRect(QRectF(tile.left() + 30 * k, tile.top() + 22 * k, tile.width() - 60 * k, 300 * k), 160 * k, 160 * k)
    top = QLinearGradient(QPointF(0, tile.top()), QPointF(0, tile.top() + 300 * k))
    top.setColorAt(0, QColor(255, 255, 255, 120))
    top.setColorAt(1, QColor(255, 255, 255, 0))
    p.setPen(Qt.PenStyle.NoPen)
    p.fillPath(shine.intersected(body), top)
    p.setPen(QPen(QColor(255, 255, 255, 170), 3 * k))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(body)
    p.end()
    return image


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


CHIP_TEXT_WIDTH = 220


def chip(text: str, hue: int | None, on_click=None) -> QPushButton:
    widget = button(text, "chip", on_click)
    widget.setIcon(swatch_icon(hue))
    widget.setIconSize(QSize(9, 9))
    widget.setFont(font("small"))
    shown = widget.fontMetrics().elidedText(text or "", Qt.TextElideMode.ElideRight, CHIP_TEXT_WIDTH)
    if shown != (text or ""):
        widget.setText(shown)
        widget.setToolTip(plain_tip(text))
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


class _Waterline(QWidget):
    """The one strip of the dream that moves. It is opaque, so a frame repaints only
    this strip: not the words above it, the pebbles below or the page behind."""

    def __init__(self, panel: NightPanel):
        super().__init__(panel)
        self.panel = panel
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.translate(0, -self.y())
        self.panel.paint_waterline(p)


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
        self._layers_key: tuple | None = None
        self._sand_layer: QPixmap | None = None
        self._sea_layer: QPixmap | None = None
        self._band_sand: QPixmap | None = None
        self._band_sea: QBrush | None = None
        self._waterline: _Waterline | None = None

    def set_fish(self, count: int) -> None:
        """A small shoal crossing the shallows: one fish for each idea that surfaced."""
        self.fish = max(0, min(int(count), 7))

    def set_shore(self, widget: QWidget) -> None:
        self.body.addSpacing(70)
        self.body.addWidget(widget)
        self.shore = widget
        self._waterline = _Waterline(self)
        self._waterline.lower()
        widget.installEventFilter(self)
        from unconscious.ui.motion import SEA_MS, ticker

        ticker().subscribe(self, self._drift, every=SEA_MS)

    # Only the waterline band moves: from just under the words to just above the pebbles.
    BAND_ABOVE, BAND_BELOW = 50, 18

    def _drift(self, step: float = 1.0) -> None:
        if self._waterline is not None:
            self._phase += 0.03 * step
            self._waterline.update()

    def _place_waterline(self) -> None:
        if self._waterline is not None:
            edge = int(self._edge())
            self._waterline.setGeometry(0, edge - self.BAND_ABOVE, self.width(), self.BAND_ABOVE + self.BAND_BELOW)

    def eventFilter(self, watched, event) -> bool:
        if watched is self.shore and event.type() in (QEvent.Type.Move, QEvent.Type.Resize):
            self._place_waterline()
        return super().eventFilter(watched, event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._place_waterline()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        from unconscious.ui.motion import ticker

        ticker().refresh()

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
        sand, sea = self._layers(rect, frame, edge)
        p.drawPixmap(0, 0, sand)
        p.setClipRect(QRectF(0, 0, rect.width(), edge - self.BAND_ABOVE))
        p.drawPixmap(0, 0, sea)  # the waterline strip paints the band below this

    def paint_waterline(self, p: QPainter) -> None:
        """The moving band, called by the waterline strip with panel coordinates.
        Only cheap operations each frame: one opaque copy, two unsmoothed fills,
        and hairline strokes, which Qt rasterises far faster than wide antialiased paths."""
        rect = QRectF(self.rect())
        frame = QPainterPath()
        frame.addRoundedRect(rect, 26, 26)
        edge = self._edge()
        self._layers(rect, frame, edge)
        if self._band_sand is None or self._band_sea is None:
            return
        top = edge - self.BAND_ABOVE
        width = rect.width()
        xs = [float(x) for x in range(0, int(width) + 8, 8)]
        wave = self._wave_y
        ratio = p.device().devicePixelRatioF() if p.device() is not None else 1.0

        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        p.drawPixmap(QPointF(0, top), self._band_sand)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        p.setPen(Qt.PenStyle.NoPen)
        damp = QColor(THEME.c["sand_shadow"])
        damp.setAlphaF(0.55)
        wet_edge = [QPointF(x, wave(x, edge + 11, 4.5, 0.7)) for x in xs]
        p.setBrush(damp)
        p.drawPolygon(QPolygonF([QPointF(0, edge - 20)] + wet_edge + [QPointF(width, edge - 20)]))
        p.setBrush(self._band_sea)
        p.drawPolygon(QPolygonF([QPointF(0, top - 1)] + [QPointF(x, wave(x, edge)) for x in xs] + [QPointF(width, top - 1)]))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        damp.setAlphaF(0.3)
        self._hairline(p, QPolygonF(wet_edge), 1.2, damp, ratio)  # softens the steps of the fill
        if not THEME.dark:
            self._glitter(p, rect, edge, moving=True)
        foam = QColor(THEME.c["foam"])
        for offset, alpha, line_width in ((0, 0.75, 1.6), (-14, 0.14, 1.1), (-30, 0.08, 1.0)):
            foam.setAlphaF(alpha)
            amplitude, speed = (5, 1.0) if offset == 0 else (3.4, 0.6)
            line = QPolygonF([QPointF(x, wave(x, edge + offset, amplitude, speed)) for x in xs if 8 <= x <= width - 8])
            self._hairline(p, line, line_width, foam, ratio)  # the main one hides the fill's steps
        if self.fish:
            self._shoal(p, rect, edge)

    @staticmethod
    def _hairline(p: QPainter, line: QPolygonF, width: float, colour: QColor, ratio: float) -> None:
        """A line ``width`` logical pixels wide, drawn as stacked one-device-pixel strokes."""
        rows = max(1, round(width * ratio))
        alpha = colour.alphaF()
        colour = QColor(colour)
        if rows > 1:  # neighbouring hairlines overlap; keep the core near the asked-for alpha
            colour.setAlphaF(1 - math.sqrt(max(0.0, 1 - alpha)))
        pen = QPen(colour, 0)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for row in range(rows):
            shift = (row - (rows - 1) / 2) / ratio
            p.drawPolyline(line.translated(0, shift))

    def _layers(self, rect: QRectF, frame: QPainterPath, edge: float) -> tuple[QPixmap, QPixmap]:
        """Everything that does not move, painted once per size and theme: the sand with
        its specks, and the sea with its light, stars and glass, both cut to the panel."""
        ratio = self.devicePixelRatioF()
        key = (self.width(), self.height(), round(edge), THEME.dark, ratio)
        if key != self._layers_key or self._sand_layer is None or self._sea_layer is None:

            def canvas() -> tuple[QPixmap, QPainter]:
                pixmap = QPixmap(max(1, round(rect.width() * ratio)), max(1, round(rect.height() * ratio)))
                pixmap.setDevicePixelRatio(ratio)
                pixmap.fill(Qt.GlobalColor.transparent)
                painter = QPainter(pixmap)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                return pixmap, painter

            sand, p = canvas()
            p.fillPath(frame, THEME.color("sand"))
            rng = __import__("random").Random(11)
            speck = QColor(THEME.c["sand_shadow"])
            for _ in range(int(rect.width() * (rect.height() - edge) / 260)):
                speck.setAlphaF(rng.uniform(0.25, 0.7))
                p.fillRect(QRectF(rng.uniform(0, rect.width()), rng.uniform(edge, rect.height()), 1.2, 1.2), speck)
            p.end()

            sea, p = canvas()
            reach = QPainterPath()
            reach.addRect(QRectF(0, 0, rect.width(), edge + 12))
            reach = reach.intersected(frame)
            p.fillPath(reach, sea_gradient(QRectF(0, 0, rect.width(), edge)))
            self._glow(p, reach, rect)
            p.end()
            self._sand_layer, self._sea_layer, self._layers_key = sand, sea, key
            # the waterline band, cut out once: opaque sand to copy, sea as a texture to fill with
            top = edge - self.BAND_ABOVE
            band = QRect(0, round(top * ratio), sand.width(), round((self.BAND_ABOVE + self.BAND_BELOW) * ratio))
            self._band_sand = sand.copy(band)
            self._band_sand.setDevicePixelRatio(ratio)
            texture = sea.copy(band)
            texture.setDevicePixelRatio(1.0)
            self._band_sea = QBrush(texture)
            self._band_sea.setTransform(QTransform().translate(0, top).scale(1 / ratio, 1 / ratio))
        return self._sand_layer, self._sea_layer

    def _glitter(self, p: QPainter, rect: QRectF, edge: float, moving: bool) -> None:
        """Sun glitter on the shallows. Glints inside the waterline band twinkle;
        the ones higher up are part of the still layer."""
        rng = __import__("random").Random(5)
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(26):
            x = rng.uniform(0.04, 0.96) * rect.width()
            y = edge - rng.uniform(26, 120)
            width = rng.uniform(3, 9)
            in_band = y > edge - self.BAND_ABOVE + 4
            if in_band != moving:
                continue
            twinkle = 0.5 + 0.5 * math.sin(self._phase * 2.2 + i * 1.7) if moving else 0.5
            spark = QColor(THEME.c["glint"])
            spark.setAlphaF(0.18 + 0.42 * twinkle)
            p.setBrush(spark)
            p.drawRoundedRect(QRectF(x, y, width, 1.4), 0.7, 0.7)

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
            self._glitter(p, rect, self._edge(), moving=False)
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
        from unconscious.ui.motion import SLOW_MS, ticker

        ticker().subscribe(self, self._tick, every=SLOW_MS)

    def set_state(self, state: str) -> None:
        self.state = state
        self.update()

    def _tick(self, step: float = 1.0) -> None:
        if self.state in {"observing", "dreaming", "idle"}:
            self._phase += (0.12 if self.state == "dreaming" else 0.06) * step
            self.update()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        from unconscious.ui.motion import ticker

        ticker().refresh()

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
            x += 3.0  # ~19 points a wavelength; antialiasing does the rest
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
        from unconscious.ui.motion import SLOW_MS, ticker

        ticker().subscribe(self, self._tick, every=SLOW_MS)

    def set_current(self, current: str | None) -> None:
        if current in self.steps:
            self.index = self.steps.index(current)
            self.update()

    def _tick(self, step: float = 1.0) -> None:
        self._phase = (self._phase + 0.06 * step) % (2 * math.pi)
        self.update()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        from unconscious.ui.motion import ticker

        ticker().refresh()

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
            name = p.fontMetrics().elidedText(t(f"steps.{step}"), Qt.TextElideMode.ElideRight, max(10, self.width() - 26))
            p.drawText(QRect(24, y - 12, self.width() - 24, 24), Qt.AlignmentFlag.AlignVCenter, name)


# ------------------------------------------------------------------ layouts


class FlowLayout(QLayout):
    """Left-to-right wrapping layout (chips, stamps, buttons).

    With ``one_line`` it asks for the width of all its items on one line, so beside other
    things in a row it keeps them on one line while there is room and wraps when there is not."""

    def __init__(self, parent=None, spacing: int = 6, *, one_line: bool = False):
        super().__init__(parent)
        self._items: list = []
        self._spacing = spacing
        self._one_line = one_line
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
        if not self._one_line or not self._items:
            return self.minimumSize()
        hints = [item.sizeHint() for item in self._items]
        width = sum(h.width() for h in hints) + self._spacing * (len(hints) - 1)
        return QSize(width, max(h.height() for h in hints))

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize().boundedTo(QSize(160, 10_000)))
        return size

    def _layout(self, rect: QRect, apply: bool) -> int:
        x, y, line = rect.x(), rect.y(), 0
        for item in self._items:
            hint = item.sizeHint().boundedTo(QSize(max(1, rect.width()), 10_000))
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

    def _need(self) -> int:
        # A card's buttons and margins set how narrow it can really go; never squeeze below that.
        return max([self.min_width] + [card.minimumSizeHint().width() for card in self.cards])

    def minimumSizeHint(self) -> QSize:
        # One column always fits, so ask the page for one card's width, not the current columns':
        # otherwise three columns hold the page wider than a small window and never get to re-flow.
        return QSize(self._need(), super().minimumSizeHint().height())

    def resizeEvent(self, event) -> None:
        width = event.size().width()
        need = self._need()
        self._place(max(1, min(3, (width + self.spacing) // (need + self.spacing))))
        super().resizeEvent(event)
