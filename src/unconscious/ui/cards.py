"""The spark card: a soft card with the mechanism said quietly, the question in
italics, a first small step, when to let it go, footnoted evidence, and the
critic's gentler objection."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QWidget,
)

from unconscious.ui.charts import human
from unconscious.ui.i18n import t
from unconscious.ui.theme import THEME, font
from unconscious.ui.widgets import (
    FlowLayout,
    Fold,
    ScoreRing,
    Stamp,
    TitleLabel,
    button,
    chip,
    eyebrow,
    hbox,
    label,
    para,
    plain_tip,
    vbox,
    wrap,
)

KIND_MARK = {"search": "⌕", "jot": "✎", "reading": "❡", "note": "·"}
SHADOW = (12, 4, 20)  # side, top, bottom room for the painted shadow
REASONS = ["generic", "known", "off_field", "infeasible", "wrong"]


class SparkCard(QFrame):
    open_spark = Signal(int)
    open_thread = Signal(int)
    feedback = Signal(int, str, str)  # spark id, status, reason

    def __init__(self, spark: dict, *, compact: bool = False, linked: bool = True, parent: QWidget | None = None):
        super().__init__(parent)
        self.spark = spark
        self.status = spark.get("status") or "new"
        self.compact = compact
        # The soft shadow is painted inside these margins rather than with a
        # QGraphicsEffect, which re-composites the textured page behind it.
        sx, st, sb = SHADOW
        body = vbox(spacing=10, margins=(26 + sx, 22 + st, 26 + sx, 18 + sb))
        top = hbox(Stamp(spark["mechanism"]), "stretch", spacing=8)
        if self.status != "new":
            tag = label(t(f"spark.{self.status}"), "caption", "accent", wrap=False)
            top.addWidget(tag)
            top.addSpacing(6)
        top.addWidget(ScoreRing(spark.get("score", 0)))
        body.addLayout(top)
        body.addSpacing(10)

        title = TitleLabel(spark["title"], "title-s" if compact else "title")
        if linked:
            title.clicked.connect(lambda: self.open_spark.emit(int(spark["id"])))
        else:
            title.setCursor(Qt.CursorShape.ArrowCursor)
        body.addWidget(title)
        if spark.get("question"):
            body.addWidget(para(spark["question"], "question", "ink2", 145))
        if not compact and spark.get("insight"):
            insight = para(spark["insight"], "body", "ink2", 150)
            # in a list the card folds a long insight; the fish's own page shows it whole
            body.addWidget(Fold(insight, lines=5, key=f"insight:{spark['id']}") if linked else insight)

        if not compact and (spark.get("first_step") or spark.get("kill")):
            grid = QGridLayout()
            grid.setHorizontalSpacing(12)
            grid.setVerticalSpacing(8)
            row = 0
            for key, value in (("spark.firstStep", spark.get("first_step")), ("spark.kill", spark.get("kill"))):
                if not value:
                    continue
                tag = eyebrow(t(key))
                grid.addWidget(tag, row, 0, Qt.AlignmentFlag.AlignTop)
                grid.addWidget(para(value, "body", None, 145), row, 1)
                row += 1
            grid.setColumnStretch(1, 1)
            body.addLayout(grid)

        threads = spark.get("threads") or []
        if threads:
            flow = FlowLayout(spacing=6)
            for th in threads:
                flow.addWidget(chip(th["name"], th.get("hue"), lambda tid=th["id"]: self.open_thread.emit(int(tid))))
            holder = QWidget()
            holder.setLayout(flow)
            body.addWidget(holder)

        evidence = spark.get("evidence") or []
        if evidence and not compact:
            body.addSpacing(2)
            box = vbox(spacing=5)
            shown = evidence[:4] if linked else evidence
            for index, item in enumerate(shown, 1):
                text = f"“{item.get('body') or item['label']}”" if item.get("kind") in {"jot", "note"} else item["label"]
                line = hbox(spacing=8)
                number = label(str(index), "caption", "accent", wrap=False)
                number.setFixedWidth(10)
                line.addWidget(number, 0, Qt.AlignmentFlag.AlignTop)
                mark = label(KIND_MARK.get(item.get("kind"), ""), "small", "muted", wrap=False)
                mark.setFixedWidth(12)
                line.addWidget(mark, 0, Qt.AlignmentFlag.AlignTop)
                content = label(text, "small", "ink2")
                tip = item["label"]
                if item.get("domain"):
                    tip += f"\n{item['domain']}"
                if item.get("seconds"):
                    tip += f" · {human(item['seconds'])}"
                content.setToolTip(plain_tip(tip))
                line.addWidget(content, 1)
                if item.get("seconds"):
                    line.addWidget(label(human(item["seconds"]), "caption", "muted", wrap=False), 0, Qt.AlignmentFlag.AlignTop)
                box.addLayout(line)
            if len(evidence) > len(shown):
                more = button(t("spark.moreEvidence", n=len(evidence) - len(shown)), "link",
                              lambda: self.open_spark.emit(int(spark["id"])))
                more.setFont(font("caption"))
                box.addLayout(hbox(more, "stretch"))
            body.addWidget(_dashed(box))

        if not compact and spark.get("objection"):
            block = QFrame()
            block.setObjectName("Objection")
            inner = vbox(spacing=3, margins=(12, 0, 0, 0))
            inner.addWidget(eyebrow(t("spark.objection")))
            inner.addWidget(para(spark["objection"], "question", "muted", 140))
            block.setLayout(inner)
            body.addWidget(block)

        body.addStretch(1)
        body.addSpacing(4)
        self.actions = self._actions()
        body.addWidget(self.actions)
        self.reasons = self._reasons()
        self.reasons.hide()
        body.addWidget(self.reasons)
        self.setLayout(body)
        self.veil = None
        if self.status == "dismissed":
            self.veil = _Veil(self)
            self.veil.raise_()

    # -- actions -------------------------------------------------------------

    def _actions(self) -> QWidget:
        sid = int(self.spark["id"])
        row = hbox(spacing=6, margins=(0, 10, 0, 0))

        def small(widget):
            widget.setProperty("size", "s")
            return widget

        def quiet(text, handler):
            widget = button(text, "quiet", handler)
            widget.setFont(font("caption-l"))
            return widget

        if self.status == "new":
            row.addWidget(small(button(t("spark.keep"), "line", lambda: self.feedback.emit(sid, "kept", ""))))
            row.addWidget(small(button(t("spark.pursue"), "ink", lambda: self.feedback.emit(sid, "pursuing", ""))))
            row.addStretch(1)
            row.addWidget(quiet(t("spark.dismiss"), self._toggle_reasons))
        else:
            if self.status != "pursuing":
                row.addWidget(small(button(t("spark.pursue"), "line", lambda: self.feedback.emit(sid, "pursuing", ""))))
            else:
                row.addWidget(small(button(t("spark.open"), "line", lambda: self.open_spark.emit(sid))))
            row.addStretch(1)
            row.addWidget(quiet(t("spark.undo"), lambda: self.feedback.emit(sid, "new", "")))
        return wrap(row)

    def _reasons(self) -> QWidget:
        sid = int(self.spark["id"])
        flow = FlowLayout(spacing=6)
        for reason in REASONS:
            flow.addWidget(button(t(f"reason.{reason}"), "reason", lambda r=reason: self.feedback.emit(sid, "dismissed", r)))
        holder = QWidget()
        holder.setLayout(flow)
        return holder

    def _toggle_reasons(self) -> None:
        self.reasons.setVisible(not self.reasons.isVisible())

    # -- paint ---------------------------------------------------------------

    def body_rect(self) -> QRectF:
        sx, st, sb = SHADOW
        return QRectF(self.rect()).adjusted(sx + 0.5, st + 0.5, -sx - 0.5, -sb - 0.5)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.body_rect()
        p.setPen(Qt.PenStyle.NoPen)
        tint = (120, 92, 54) if not THEME.dark else (0, 0, 0)
        layers = 10
        for i in range(layers, 0, -1):
            spread = i * 1.6
            alpha = (9 if not THEME.dark else 22) * (1 - i / (layers + 1)) ** 1.6
            p.setBrush(QColor(*tint, int(alpha)))
            p.drawRoundedRect(rect.adjusted(-spread * 0.6, -spread * 0.2 + 6, spread * 0.6, spread * 0.55 + 6), 22 + spread, 22 + spread)
        surface = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        surface.setColorAt(0, THEME.color("card_top"))
        surface.setColorAt(1, THEME.color("card_bottom"))
        p.setPen(QPen(THEME.color("rule"), 1))
        p.setBrush(surface)
        p.drawRoundedRect(rect, 22, 22)
        p.save()
        p.setClipRect(QRectF(0, 0, self.width(), rect.top() + 24))
        p.setPen(QPen(QColor(255, 255, 255, 230 if not THEME.dark else 40), 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(rect.adjusted(1.2, 1.2, -1.2, -1.2), 21, 21)
        p.restore()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if getattr(self, "veil", None) is not None:
            self.veil.setGeometry(self.rect())


class _Veil(QWidget):
    """A translucent wash of paper over a dismissed card (no graphics effect needed)."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(THEME.color("paper", 0.48))
        p.drawRoundedRect(self.parentWidget().body_rect(), 22, 22)


def _dashed(layout) -> QWidget:
    holder = QFrame()
    holder.setObjectName("Evidence")
    layout.setContentsMargins(0, 10, 0, 0)
    holder.setLayout(layout)
    return holder
