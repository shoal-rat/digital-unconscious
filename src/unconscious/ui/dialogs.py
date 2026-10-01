"""Small dialogs with a little life in them: a bottle for thoughts, a shark for forgetting."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPainterPath, QPen, QShortcut
from PySide6.QtWidgets import QDialog, QHBoxLayout, QPlainTextEdit, QSizePolicy, QVBoxLayout, QWidget

from unconscious.ui.i18n import t
from unconscious.ui.motion import ticker
from unconscious.ui.theme import THEME, font
from unconscious.ui.widgets import button, eyebrow, label, para, sea_gradient


class _Strip(QWidget):
    """A strip of sea with something moving on it; subclasses paint the something."""

    def __init__(self, height: int = 72):
        super().__init__()
        self.phase = 2.6  # start with the fin (or bottle) already in view
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        ticker().subscribe(self, self._tick)

    def _tick(self, step: float = 1.0) -> None:
        self.phase += 0.08 * step
        self.update()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        ticker().refresh()

    def wave_y(self, x: float) -> float:
        return self.height() * 0.58 + math.sin(x / 26 + self.phase * 2.2) * 2.6

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())
        frame = QPainterPath()
        frame.addRoundedRect(rect, 16, 16)
        p.fillPath(frame, sea_gradient(rect))
        self.draw(p, rect)
        foam = QColor(THEME.c["foam"])
        foam.setAlphaF(0.55)
        p.setPen(QPen(foam, 1.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        line = QPainterPath()
        x = 6.0
        line.moveTo(x, self.wave_y(x))
        while x < rect.width() - 6:
            line.lineTo(x, self.wave_y(x))
            x += 3
        p.drawPath(line)

    def draw(self, p: QPainter, rect: QRectF) -> None:  # override
        pass


class FinStrip(_Strip):
    """A dorsal fin gliding along the surface, unhurried."""

    def draw(self, p: QPainter, rect: QRectF) -> None:
        x = (self.phase * 35) % (rect.width() + 120) - 60
        base = self.wave_y(x) + 1
        fin = QPainterPath()
        fin.moveTo(x - 16, base)
        fin.cubicTo(x - 10, base - 10, x - 2, base - 26, x + 8, base - 30)
        fin.cubicTo(x + 4, base - 20, x + 6, base - 8, x + 14, base)
        fin.closeSubpath()
        p.fillPath(fin, QColor("#24364c") if not THEME.dark else QColor("#0a1220"))
        wake = QColor(THEME.c["foam"])
        wake.setAlphaF(0.5)
        p.setPen(QPen(wake, 1.2))
        for i in range(3):
            p.drawLine(QPointF(x - 20 - i * 12, base + 2 + i), QPointF(x - 30 - i * 12, base + 2 + i))


class BottleStrip(_Strip):
    """A corked bottle with a rolled note, bobbing on the swell."""

    def draw(self, p: QPainter, rect: QRectF) -> None:
        x = rect.width() * 0.5 + math.sin(self.phase * 0.7) * 26
        y = self.wave_y(x) - 4
        p.save()
        p.translate(x, y)
        p.rotate(-14 + math.sin(self.phase * 1.7) * 8)
        glass = QColor("#e8f6f4")
        glass.setAlphaF(0.75)
        body = QPainterPath()
        body.addRoundedRect(QRectF(-20, -8, 32, 16), 7, 7)
        neck = QPainterPath()
        neck.addRoundedRect(QRectF(10, -4, 12, 8), 3, 3)
        p.fillPath(body.united(neck), glass)
        p.fillRect(QRectF(21, -3.5, 5, 7), QColor("#b08b68"))
        p.fillRect(QRectF(-13, -4, 16, 8), QColor("#f6efe0"))
        p.setPen(QPen(QColor("#c9b89a"), 0.8))
        p.drawLine(QPointF(-11, -1.5), QPointF(1, -1.5))
        p.drawLine(QPointF(-11, 1.5), QPointF(-2, 1.5))
        p.setPen(QPen(QColor(255, 255, 255, 200), 1.2))
        p.drawLine(QPointF(-15, -5.5), QPointF(4, -5.5))
        p.restore()


class SharkDialog(QDialog):
    """Forgetting, said plainly: the shark eats it."""

    def __init__(self, parent: QWidget, title: str, body: str):
        super().__init__(parent)
        self.setWindowTitle(t("settings.forgetAll"))
        self.setModal(True)
        self.setMinimumWidth(460)
        box = QVBoxLayout(self)
        box.setContentsMargins(24, 22, 24, 18)
        box.setSpacing(12)
        box.addWidget(FinStrip())
        box.addSpacing(4)
        box.addWidget(label(title, "display-s"))
        box.addWidget(para(body, "body", "ink2", 150))
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(button(t("common.cancel"), "ghost", self.reject))
        eat = button(t("shark.go"), "accent", self.accept)
        eat.setStyleSheet("QPushButton { background: #c9573e; border-color: #a8442f; color: #ffffff; }")
        row.addWidget(eat)
        box.addLayout(row)


class JotDialog(QDialog):
    """A message in a bottle for tonight's dream."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setWindowTitle(t("jot.title"))
        self.setModal(True)
        self.setMinimumWidth(540)
        box = QVBoxLayout(self)
        box.setContentsMargins(26, 22, 26, 20)
        box.setSpacing(10)
        box.addWidget(BottleStrip(64))
        box.addSpacing(6)
        box.addWidget(eyebrow(t("jot.eyebrow")))
        box.addWidget(label(t("jot.title"), "display-m"))
        self.text = QPlainTextEdit()
        self.text.setPlaceholderText(t("jot.placeholder"))
        self.text.setFont(font("reading"))
        self.text.setFixedHeight(130)
        box.addWidget(self.text)
        box.addWidget(label(t("jot.hint"), "caption", "muted"))
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(button(t("common.cancel"), "ghost", self.reject))
        keep = button(t("jot.save"), "accent", self.accept)
        keep.setDefault(True)
        row.addWidget(keep)
        box.addLayout(row)
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self.accept)

    def value(self) -> str:
        return self.text.toPlainText().strip()
