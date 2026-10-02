"""Painted pictures of attention: the day's tide, the pebbles of a day, the
strata of threads over weeks, and a thread's history."""

from __future__ import annotations

import hashlib
import math
import random
from datetime import date

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFontMetrics,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
    QTransform,
)
from PySide6.QtWidgets import QSizePolicy, QToolTip, QWidget

from unconscious.text import duration
from unconscious.ui.i18n import language, t
from unconscious.ui.theme import THEME, font
from unconscious.ui.widgets import plain_tip


def short_day(day: str) -> str:
    d = date.fromisoformat(day)
    if language() == "zh":
        return f"{d.month}月{d.day}日"
    return f"{d.day} {d.strftime('%b')}"  # %-d is not portable to Windows


def human(seconds: float) -> str:
    seconds = seconds or 0
    if language() == "zh":
        minutes = int(round(seconds / 60))
        if minutes < 60:
            return f"{max(minutes, 1)} 分钟" if seconds else "—"
        hours, rest = divmod(minutes, 60)
        return f"{hours} 小时" + (f" {rest} 分" if rest else "")
    return duration(seconds) if seconds else "—"


def _clock(minutes: float) -> str:
    return f"{int(minutes // 60) % 24:02d}:{int(round(minutes % 60)) % 60:02d}"


# ------------------------------------------------------------------ pebbles


def pebble_path(rx: float, ry: float, seed: int) -> QPainterPath:
    """A smooth, slightly irregular stone, like the galets of the Promenade."""
    rng = random.Random(seed)
    a1, a2, p1, p2 = rng.uniform(0.03, 0.07), rng.uniform(0.02, 0.05), rng.uniform(0, 6.28), rng.uniform(0, 6.28)
    path = QPainterPath()
    steps = 48
    for i in range(steps + 1):
        theta = i / steps * 2 * math.pi
        k = 1 + a1 * math.sin(2 * theta + p1) + a2 * math.sin(3 * theta + p2)
        point = QPointF(rx * k * math.cos(theta), ry * k * math.sin(theta))
        if i == 0:
            path.moveTo(point)
        else:
            path.lineTo(point)
    path.closeSubpath()
    return QTransform().rotate(rng.uniform(-24, 24)).map(path)


def draw_pebble(p: QPainter, center: QPointF, rx: float, ry: float, color: QColor, seed: int, sand_shadow: QColor) -> QPainterPath:
    """A glossy stone, lit from the upper left: part galet, part candy-coloured iMac."""
    shape = QTransform().translate(center.x(), center.y()).map(pebble_path(rx, ry, seed))
    shadow = QRadialGradient(QPointF(center.x() + rx * 0.12, center.y() + ry * 0.78), rx * 1.25)
    shade = QColor(sand_shadow)
    shade.setAlphaF(0.85)
    shadow.setColorAt(0, shade)
    shade.setAlphaF(0.0)
    shadow.setColorAt(1, shade)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(shadow)
    p.drawEllipse(QPointF(center.x() + rx * 0.12, center.y() + ry * 0.78), rx * 1.25, ry * 0.55)
    body = QRadialGradient(QPointF(center.x() - rx * 0.35, center.y() - ry * 0.5), max(rx, ry) * 1.55)
    body.setColorAt(0.0, color.lighter(150))
    body.setColorAt(0.45, color)
    body.setColorAt(1.0, color.darker(135))
    p.fillPath(shape, body)
    rim = QLinearGradient(QPointF(0, center.y() + ry * 0.2), QPointF(0, center.y() + ry))
    glow = QColor(255, 255, 255, 0)
    rim.setColorAt(0, glow)
    glow.setAlpha(46)
    rim.setColorAt(1, glow)
    p.fillPath(shape, rim)
    sheen = QPainterPath()
    sheen.addEllipse(QPointF(center.x() - rx * 0.3, center.y() - ry * 0.42), rx * 0.48, ry * 0.26)
    soft = QRadialGradient(QPointF(center.x() - rx * 0.3, center.y() - ry * 0.46), rx * 0.48)
    soft.setColorAt(0, QColor(255, 255, 255, 150))
    soft.setColorAt(1, QColor(255, 255, 255, 0))
    p.fillPath(sheen.intersected(shape), soft)
    p.setBrush(QColor(255, 255, 255, 215))
    p.drawEllipse(QPointF(center.x() - rx * 0.42, center.y() - ry * 0.46), max(1.2, rx * 0.09), max(0.9, ry * 0.07))
    return shape


def layout_pebbles(threads: list[dict], width: float, height: float, seed: str, scale: float = 1.0) -> list[dict]:
    """Scatter pebbles along a beach without overlap; threads that collided today lie touching."""
    if not threads:
        return []
    rng = random.Random(int(hashlib.sha1(seed.encode()).hexdigest()[:8], 16))
    peak = max(th["seconds"] for th in threads) or 1
    placed: list[dict] = []
    margin = 26 * scale
    for th in sorted(threads, key=lambda x: -x["seconds"]):
        r = (9 + 25 * math.sqrt(max(th["seconds"], 1) / peak)) * scale
        rx, ry = r * 1.28, r * 0.82
        partner = next((q for q in placed if q["id"] in th.get("touch", set())), None)
        best = None
        for attempt in range(220):
            if partner and attempt < 80:
                angle = rng.uniform(-0.9, 0.9) + (0 if rng.random() < 0.5 else math.pi)
                distance = (partner["rx"] + rx) * 0.88
                x = partner["x"] + math.cos(angle) * distance
                y = partner["y"] + math.sin(angle) * distance * 0.6
            else:
                x = rng.uniform(margin + rx, max(margin + rx + 1, width - margin - rx))
                y = rng.uniform(ry + 8 * scale, max(ry + 9 * scale, height - ry - 8 * scale))
            if not (margin * 0.5 + rx <= x <= width - margin * 0.5 - rx and ry * 0.6 <= y <= height - ry * 0.9):
                continue
            clear = all(
                ((x - q["x"]) / ((rx + q["rx"]) * 0.98)) ** 2 + ((y - q["y"]) / ((ry + q["ry"]) * 1.05)) ** 2 >= 1
                for q in placed
            )
            if clear:
                best = (x, y)
                break
        if best is None:
            continue
        placed.append({**th, "x": best[0], "y": best[1], "rx": rx, "ry": ry, "seed": rng.randrange(1 << 30)})
    return placed


def shore_threads(topics: list[dict], signals: list[dict]) -> list[dict]:
    """Collapse a dream's topics into one pebble per thread."""
    threads: dict[int, dict] = {}
    for topic in topics or []:
        tid = topic.get("thread_id")
        if tid is None:
            continue
        entry = threads.setdefault(tid, {"id": tid, "name": topic.get("thread_name") or topic.get("label"),
                                         "hue": topic.get("hue"), "seconds": 0.0, "touch": set()})
        entry["seconds"] += topic.get("seconds") or 0
    for signal in signals or []:
        if signal.get("kind") == "collision" and len(signal.get("threads") or []) == 2:
            a, b = signal["threads"]
            if a in threads and b in threads:
                threads[a]["touch"].add(b)
                threads[b]["touch"].add(a)
    for entry in threads.values():
        entry["seconds"] = max(entry["seconds"], 60)
    return list(threads.values())


class Pebbles(QWidget):
    """The day's threads as pebbles. Size is time; pebbles that touch met today."""

    thread_clicked = Signal(int)

    def __init__(self, threads: list[dict], seed: str, height: int = 150, labels: bool = True, night: bool = True, parent=None):
        super().__init__(parent)
        self.threads = threads
        self.seed = seed
        self.labels = labels
        self.night = night
        self._placed: list[dict] = []
        self._size = (0, 0)
        self.setFixedHeight(height)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _layout(self) -> list[dict]:
        if self._size != (self.width(), self.height()):
            scale = min(1.0, self.height() / 150)
            self._placed = layout_pebbles(self.threads, self.width(), self.height(), self.seed, scale)
            self._size = (self.width(), self.height())
        return self._placed

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        shadow = THEME.color("sand_shadow")
        placed = self._layout()
        for pebble in placed:
            draw_pebble(p, QPointF(pebble["x"], pebble["y"]), pebble["rx"], pebble["ry"], THEME.hue(pebble["hue"], night=False), pebble["seed"], shadow)
        if self.labels:
            p.setFont(font("caption"))
            metrics = QFontMetrics(p.font())
            p.setPen(QColor("#6d5c47") if not THEME.dark else QColor("#aeb9c6"))
            taken = [QRectF(q["x"] - q["rx"], q["y"] - q["ry"], 2 * q["rx"], 2 * q["ry"]) for q in placed]
            for pebble in sorted(placed, key=lambda q: -q["seconds"]):
                text = metrics.elidedText(pebble["name"] or "", Qt.TextElideMode.ElideRight, 170)
                w = metrics.horizontalAdvance(text) + 6
                box = QRectF(pebble["x"] - w / 2, pebble["y"] + pebble["ry"] + 5, w, 17)
                own = QRectF(pebble["x"] - pebble["rx"], pebble["y"] - pebble["ry"], 2 * pebble["rx"], 2 * pebble["ry"])
                if box.bottom() > self.height() or any(box.intersects(r) for r in taken if r != own):
                    continue  # never let a label sit on another stone; the tooltip still names it
                p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
                taken.append(box)

    def _at(self, pos: QPointF) -> dict | None:
        for pebble in self._layout():
            if ((pos.x() - pebble["x"]) / pebble["rx"]) ** 2 + ((pos.y() - pebble["y"]) / pebble["ry"]) ** 2 <= 1.1:
                return pebble
        return None

    def mouseMoveEvent(self, event) -> None:
        pebble = self._at(event.position())
        self.setCursor(Qt.CursorShape.PointingHandCursor if pebble else Qt.CursorShape.ArrowCursor)
        if pebble:
            QToolTip.showText(event.globalPosition().toPoint(), plain_tip(f"{pebble['name']}\n{human(pebble['seconds'])}"), self)
        else:
            QToolTip.hideText()

    def mouseReleaseEvent(self, event) -> None:
        pebble = self._at(event.position())
        if pebble and event.button() == Qt.MouseButton.LeftButton:
            self.thread_clicked.emit(int(pebble["id"]))


class ShoreThumb(QWidget):
    """A small square of the day: a strip of evening sea above, its pebbles below."""

    def __init__(self, threads: list[dict], seed: str, size: QSize | None = None, parent=None):
        super().__init__(parent)
        size = size or QSize(132, 84)
        self.threads = threads
        self.seed = seed
        self.setFixedSize(size)
        self._placed = layout_pebbles(threads, size.width(), size.height() - 24, seed, scale=0.42)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())
        frame = QPainterPath()
        frame.addRoundedRect(rect, 14, 14)
        p.fillPath(frame, THEME.color("sand"))
        sea = QPainterPath()
        sea.moveTo(0, 0)
        sea.lineTo(0, 20)
        x = 0.0
        while x <= rect.width():
            sea.lineTo(x, 20 + math.sin(x / 13 + len(self.seed)) * 2.2)
            x += 2
        sea.lineTo(rect.width(), 0)
        sea.closeSubpath()
        from unconscious.ui.widgets import sea_gradient

        p.fillPath(sea.intersected(frame), sea_gradient(QRectF(0, 0, rect.width(), 24)))
        p.translate(0, 22)
        for pebble in self._placed:
            draw_pebble(p, QPointF(pebble["x"], pebble["y"]), pebble["rx"], pebble["ry"], THEME.hue(pebble["hue"]), pebble["seed"], THEME.color("sand_shadow"))


# ------------------------------------------------------------------ the day's tide


class Ribbon(QWidget):
    """One day as a band of rounded strokes. Colour = thread, gaps = absence,
    a small round mark = a jot or a fed document."""

    def __init__(self, view: dict, parent: QWidget | None = None):
        super().__init__(parent)
        self.segments = view.get("segments") or []
        self.marks = view.get("marks") or []
        self.setMouseTracking(True)
        self.setFixedHeight(78)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        points = [x for s in self.segments for x in (s["start"], s["end"])] + [m["at"] for m in self.marks]
        lo = math.floor(min(points) / 60) * 60 if points else 8 * 60
        hi = math.ceil(max(points) / 60) * 60 if points else 20 * 60
        self.lo = min(lo, 8 * 60)
        self.hi = max(hi, self.lo + 6 * 60)

    def _x(self, minutes: float) -> float:
        return (minutes - self.lo) / (self.hi - self.lo) * (self.width() - 16) + 8

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        top, height = 18, 26
        track = QPainterPath()
        track.addRoundedRect(QRectF(0, top, self.width(), height), height / 2, height / 2)
        p.fillPath(track, THEME.color("paper2"))
        p.setPen(Qt.PenStyle.NoPen)
        for seg in self.segments:
            x0, x1 = self._x(seg["start"]), self._x(seg["end"])
            color = THEME.hue(seg.get("hue")) if seg.get("hue") is not None else THEME.color("none")
            w = max(4.0, x1 - x0)
            capsule = QRectF(x0, top + 5, w, height - 10)
            gel = QLinearGradient(capsule.topLeft(), capsule.bottomLeft())
            gel.setColorAt(0.0, color.lighter(145))
            gel.setColorAt(0.48, color.lighter(112))
            gel.setColorAt(0.52, color)
            gel.setColorAt(1.0, color.lighter(122))
            p.setBrush(gel)
            p.drawRoundedRect(capsule, capsule.height() / 2, capsule.height() / 2)
            if w > 7:
                p.setBrush(QColor(255, 255, 255, 90))
                p.drawRoundedRect(QRectF(x0 + 2, top + 6.2, w - 4, (height - 10) * 0.32), 3, 3)
        for mark in self.marks:
            x = self._x(mark["at"])
            p.setBrush(THEME.mechanism("collision") if mark["kind"] == "jot" else THEME.color("ink2"))
            p.drawEllipse(QPointF(x, 7), 3.4, 3.4)
            p.setPen(QPen(THEME.color("faint"), 1))
            p.drawLine(QPointF(x, 11), QPointF(x, top - 1))
            p.setPen(Qt.PenStyle.NoPen)
        p.setFont(font("caption"))
        p.setPen(THEME.color("muted"))
        step = 3 if (self.hi - self.lo) / 60 > 12 else 2
        hour = self.lo // 60
        while hour * 60 <= self.hi:
            x = self._x(hour * 60)
            p.drawText(QRectF(x - 24, top + height + 8, 48, 16), Qt.AlignmentFlag.AlignCenter, f"{int(hour) % 24}h")
            hour += step

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        text = ""
        if pos.y() < 16:
            for mark in self.marks:
                if abs(self._x(mark["at"]) - pos.x()) <= 6:
                    text = f"{t('mech.' + mark['kind'])} · {_clock(mark['at'])}\n{mark['label']}"
                    break
        else:
            for seg in self.segments:
                if self._x(seg["start"]) - 2 <= pos.x() <= self._x(seg["end"]) + 2:
                    head = seg.get("thread") or seg.get("category") or ""
                    text = f"{head} · {_clock(seg['start'])}–{_clock(seg['end'])}\n" + "\n".join(seg.get("labels") or [])
                    break
        if text:
            QToolTip.showText(event.globalPosition().toPoint(), plain_tip(text), self)
        else:
            QToolTip.hideText()


# ------------------------------------------------------------------ strata


class Strata(QWidget):
    """Threads as rows of pebbles, one per day, sized by time."""

    thread_clicked = Signal(int)
    ROW = 54
    HEAD = 30

    def __init__(self, view: dict, parent: QWidget | None = None):
        super().__init__(parent)
        self.days = view.get("days") or []
        self.rows = [r for r in view.get("threads") or [] if r["state"] != "muted"]
        self.muted = [r for r in view.get("threads") or [] if r["state"] == "muted"]
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._hover: int | None = None
        self.setFixedHeight(self._height())

    def _height(self) -> int:
        extra = 40 if self.muted else 0
        return self.HEAD + (len(self.rows) + len(self.muted)) * self.ROW + extra + 8

    def sizeHint(self) -> QSize:
        return QSize(900, self._height())

    def _geometry(self) -> tuple[int, int, int, int]:
        width = self.width()
        name_w = 240 if width > 820 else 180
        sig_w = 170 if width > 820 else 0
        columns = 28 if width > 900 else 21 if width > 620 else 14
        columns = min(columns, len(self.days)) or 1
        return name_w, sig_w, columns, width - name_w - sig_w - 24

    def _all_rows(self) -> list[tuple[int, dict]]:
        out, y = [], self.HEAD
        for row in self.rows:
            out.append((y, row))
            y += self.ROW
        if self.muted:
            y += 40
            for row in self.muted:
                out.append((y, row))
                y += self.ROW
        return out

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        name_w, sig_w, columns, dots_w = self._geometry()
        days = self.days[-columns:]
        cell = dots_w / max(columns, 1)
        x0 = name_w + 12
        peak = max([1.0] + [v for r in self.rows + self.muted for v in r["series"][-columns:]])
        p.setFont(font("caption"))
        p.setPen(THEME.color("muted"))
        for i, day in enumerate(days):
            d = date.fromisoformat(day)
            last = i == len(days) - 1
            if last or d.weekday() == 0:
                text = t("threads.today") if last else short_day(day)
                p.drawText(QRectF(x0 + i * cell - 30, 0, cell + 60, 18), Qt.AlignmentFlag.AlignCenter, text)
        if self.muted:
            y_label = self.HEAD + len(self.rows) * self.ROW + 22
            p.drawText(QRectF(0, y_label - 9, 300, 18), Qt.AlignmentFlag.AlignVCenter, t("threads.muted"))
        for y, row in self._all_rows():
            muted = row["state"] == "muted"
            if self._hover == row["id"]:
                band = QPainterPath()
                band.addRoundedRect(QRectF(-6, y + 3, self.width() + 12, self.ROW - 6), 16, 16)
                p.fillPath(band, THEME.color("paper2"))
            p.setOpacity(0.45 if muted else 1.0)
            hue = THEME.hue(row.get("hue"))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(hue)
            p.drawEllipse(QRectF(2, y + 15, 10, 8))
            title_font = font("title-s")
            p.setFont(title_font)
            p.setPen(THEME.color("ink"))
            name = QFontMetrics(title_font).elidedText(row["name"], Qt.TextElideMode.ElideRight, name_w - 22)
            p.drawText(QRectF(20, y + 6, name_w - 20, 24), Qt.AlignmentFlag.AlignVCenter, name)
            p.setFont(font("caption"))
            p.setPen(THEME.color("muted"))
            facts = f"{t('threads.days', n=row['active_14'])} · {human(row['total'])}"
            facts = QFontMetrics(p.font()).elidedText(facts, Qt.TextElideMode.ElideRight, name_w - 22)
            p.drawText(QRectF(20, y + 29, name_w - 20, 16), Qt.AlignmentFlag.AlignVCenter, facts)
            series = row["series"][-columns:]
            for i, value in enumerate(series):
                cx = x0 + i * cell + cell / 2
                cy = y + self.ROW / 2
                if value <= 0:
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(THEME.color("rule"))
                    p.drawEllipse(QPointF(cx, cy), 1.4, 1.4)
                    continue
                s = math.sqrt(value / peak)
                r = min(cell / 2 - 1.5, 3 + s * 10)
                color = QColor(hue)
                orb = QRadialGradient(QPointF(cx - r * 0.35, cy - r * 0.35), r * 1.5)
                orb.setColorAt(0, color.lighter(150))
                orb.setColorAt(0.5, color)
                orb.setColorAt(1, color.darker(128))
                p.setOpacity((0.55 + 0.45 * s) * (0.45 if muted else 1.0))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(orb)
                p.drawEllipse(QPointF(cx, cy), r * 1.12, r * 0.88)
                if r > 4:
                    p.setBrush(QColor(255, 255, 255, 170))
                    p.drawEllipse(QPointF(cx - r * 0.4, cy - r * 0.38), r * 0.22, r * 0.14)
                p.setOpacity(0.45 if muted else 1.0)
                if i == len(series) - 1:
                    p.setPen(QPen(THEME.color("accent"), 1.3))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawEllipse(QPointF(cx, cy), r * 1.12 + 3.5, r * 0.88 + 3.5)
            if sig_w:
                p.setFont(font("caption"))
                x = self.width() - 6
                for kind, word in reversed(self._signal_words(row)[:3]):
                    w = QFontMetrics(p.font()).horizontalAdvance(word) + 16
                    if x - w < self.width() - sig_w:  # the rest stay in the row's tooltip, off the dots
                        break
                    x -= w
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(THEME.mechanism(kind))
                    p.drawEllipse(QRectF(x, y + self.ROW / 2 - 3, 8, 6))
                    p.setPen(THEME.color("ink2"))
                    p.drawText(QRectF(x + 12, y, w, self.ROW), Qt.AlignmentFlag.AlignVCenter, word)
                    x -= 10
            p.setOpacity(1.0)

    @staticmethod
    def _signal_words(row: dict) -> list[tuple[str, str]]:
        words = [(kind, t(f"mech.{kind}")) for kind in row["signals"] if kind != "collision"]
        return [(kind, word.lower() if word.isascii() else word) for kind, word in words]

    def _row_at(self, y: float) -> dict | None:
        for top, row in self._all_rows():
            if top <= y < top + self.ROW:
                return row
        return None

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        row = self._row_at(pos.y())
        hover = row["id"] if row else None
        if hover != self._hover:
            self._hover = hover
            self.setCursor(Qt.CursorShape.PointingHandCursor if row else Qt.CursorShape.ArrowCursor)
            self.update()
        if not row:
            QToolTip.hideText()
            return
        name_w, _sig_w, columns, dots_w = self._geometry()
        cell = dots_w / max(columns, 1)
        index = int((pos.x() - name_w - 12) // cell)
        days = self.days[-columns:]
        if 0 <= index < len(days):
            value = row["series"][-columns:][index]
            QToolTip.showText(event.globalPosition().toPoint(), plain_tip(f"{row['name']}\n{short_day(days[index])} · {human(value)}"), self)
        else:
            tip = [row["name"], row.get("gist") or "", " · ".join(word for _, word in self._signal_words(row))]
            QToolTip.showText(event.globalPosition().toPoint(), plain_tip("\n".join(x for x in tip if x)), self)

    def leaveEvent(self, _event) -> None:
        self._hover = None
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        row = self._row_at(event.position().y())
        if row and event.button() == Qt.MouseButton.LeftButton:
            self.thread_clicked.emit(int(row["id"]))


class HistoryBars(QWidget):
    def __init__(self, series: list[float], days: list[str], hue: int | None, parent: QWidget | None = None):
        super().__init__(parent)
        self.series = series
        self.days = days
        self.hue = hue
        self.setFixedHeight(150)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        n = max(1, len(self.series))
        base = self.height() - 24
        peak = max([1.0] + self.series)
        bar = self.width() / n
        p.setPen(Qt.PenStyle.NoPen)
        for i, value in enumerate(self.series):
            if value:
                h = max(5.0, (value / peak) * (base - 8))
                w = max(3.0, bar - 4)
                color = THEME.hue(self.hue)
                tube = QLinearGradient(QPointF(i * bar + 2, 0), QPointF(i * bar + 2 + w, 0))
                tube.setColorAt(0.0, color.darker(110))
                tube.setColorAt(0.35, color.lighter(138))
                tube.setColorAt(1.0, color.darker(112))
                p.setBrush(tube)
                p.drawRoundedRect(QRectF(i * bar + 2, base - h, w, h + 3), min(w / 2, 5), min(w / 2, 5))
            else:
                p.setBrush(THEME.color("rule"))
                p.drawEllipse(QPointF(i * bar + bar / 2, base - 2), 1.3, 1.3)
        p.fillRect(QRectF(0, base + 2, self.width(), 1), THEME.color("rule"))
        p.setFont(font("caption"))
        p.setPen(THEME.color("muted"))
        if self.days:
            p.drawText(QRectF(0, base + 8, 160, 16), Qt.AlignmentFlag.AlignLeft, short_day(self.days[0]))
            p.drawText(QRectF(self.width() - 160, base + 8, 160, 16), Qt.AlignmentFlag.AlignRight, t("threads.today"))

    def mouseMoveEvent(self, event) -> None:
        n = max(1, len(self.series))
        index = int(event.position().x() // (self.width() / n))
        if 0 <= index < len(self.series) and index < len(self.days):
            QToolTip.showText(event.globalPosition().toPoint(), f"{short_day(self.days[index])} · {human(self.series[index])}", self)
